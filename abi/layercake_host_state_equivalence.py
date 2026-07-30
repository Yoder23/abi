"""Paired exact-state verification for native LayerCake graph rewrites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .layercake_host_runtime import (
    LayerCakeHostRuntimeError,
    NativeHostRuntime,
    _canonical_sha,
    _sha256_file,
    _write_json,
    generate_native_host,
)


def _array_check(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    exact = bool(np.array_equal(left, right))
    return {
        "exact": exact,
        "maximum_absolute_difference": (
            0.0
            if exact
            else float(
                np.max(
                    np.abs(
                        left.astype(np.float64)
                        - right.astype(np.float64)
                    )
                )
            )
        ),
    }


def verify_state_equivalence(
    *,
    reference_artifact: str | Path,
    candidate_artifact: str | Path,
    validation_evidence_path: str | Path,
    output_path: str | Path,
    threads: int = 16,
    decode_steps: int = 16,
) -> dict[str, Any]:
    """Require bit-exact public state and generation on one prompt per route."""

    reference_artifact = Path(reference_artifact).resolve()
    candidate_artifact = Path(candidate_artifact).resolve()
    validation_evidence_path = Path(validation_evidence_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"state-equivalence evidence is immutable: {output_path}"
        )
    source_evidence = json.loads(
        validation_evidence_path.read_text(encoding="utf-8")
    )
    selected: dict[int, dict[str, Any]] = {}
    for row in source_evidence["observations"]:
        route = int(row["automatic_route"])
        if row.get("symbolic_handler_used") is False:
            selected.setdefault(route, row)
    for row in source_evidence["observations"]:
        selected.setdefault(int(row["automatic_route"]), row)
    if set(selected) != set(range(9)):
        raise LayerCakeHostRuntimeError(
            "validation evidence does not cover every English route"
        )
    reference = NativeHostRuntime(reference_artifact, threads=threads)
    candidate = NativeHostRuntime(candidate_artifact, threads=threads)
    if (
        reference.metadata["tokenizer"]["sha256"]
        != candidate.metadata["tokenizer"]["sha256"]
    ):
        raise LayerCakeHostRuntimeError("tokenizers differ")

    observations = []
    for expected_route, row in sorted(selected.items()):
        prompt = str(row["prompt"])
        max_new_tokens = int(row["max_new_tokens"])
        reference_generation = generate_native_host(
            reference, prompt, max_new_tokens=max_new_tokens
        )
        candidate_generation = generate_native_host(
            candidate, prompt, max_new_tokens=max_new_tokens
        )
        prompt_ids = reference.encode(prompt + "\n")
        reference_logits, reference_state = reference.prefill(prompt_ids)
        candidate_logits, candidate_state = candidate.prefill(prompt_ids)
        prefill = {
            "logits": _array_check(
                reference_logits, candidate_logits
            ),
            "route": _array_check(
                reference_state.route, candidate_state.route
            ),
            "abi_state": _array_check(
                reference_state.abi_state,
                candidate_state.abi_state,
            ),
            "cache": [
                _array_check(left, right)
                for left, right in zip(
                    reference_state.cache, candidate_state.cache
                )
            ],
            "output_token_ids": _array_check(
                reference_state.output_token_ids,
                candidate_state.output_token_ids,
            ),
        }
        decode = []
        generated_ids = reference_generation[
            "authoritative_generated_token_ids"
        ][:decode_steps]
        for token_id in generated_ids:
            reference_logits, reference_state = reference.decode_step(
                int(token_id), reference_state
            )
            candidate_logits, candidate_state = candidate.decode_step(
                int(token_id), candidate_state
            )
            decode.append(
                {
                    "token_id": int(token_id),
                    "logits": _array_check(
                        reference_logits, candidate_logits
                    ),
                    "route": _array_check(
                        reference_state.route, candidate_state.route
                    ),
                    "abi_state": _array_check(
                        reference_state.abi_state,
                        candidate_state.abi_state,
                    ),
                    "cache": [
                        _array_check(left, right)
                        for left, right in zip(
                            reference_state.cache,
                            candidate_state.cache,
                        )
                    ],
                }
            )
        generation_exact = (
            reference_generation["output"]
            == candidate_generation["output"]
            and reference_generation[
                "authoritative_generated_token_ids"
            ]
            == candidate_generation[
                "authoritative_generated_token_ids"
            ]
            and reference_generation["route"]
            == candidate_generation["route"]
        )
        all_checks = [
            prefill["logits"]["exact"],
            prefill["route"]["exact"],
            prefill["abi_state"]["exact"],
            prefill["output_token_ids"]["exact"],
            *(item["exact"] for item in prefill["cache"]),
            generation_exact,
        ]
        for step in decode:
            all_checks.extend(
                [
                    step["logits"]["exact"],
                    step["route"]["exact"],
                    step["abi_state"]["exact"],
                    *(item["exact"] for item in step["cache"]),
                ]
            )
        observations.append(
            {
                "probe_id": row["probe_id"],
                "capability": row["capability"],
                "expected_route": expected_route,
                "prompt_sha256": _canonical_sha({"prompt": prompt}),
                "generation_exact": generation_exact,
                "reference_output_sha256": reference_generation[
                    "output_sha256"
                ],
                "candidate_output_sha256": candidate_generation[
                    "output_sha256"
                ],
                "prefill": prefill,
                "decode_steps_checked": len(decode),
                "decode": decode,
                "pass": bool(all(all_checks)),
            }
        )
    evidence = {
        "format": "abi-layercake-native-state-equivalence/1",
        "status": (
            "PASS"
            if len(observations) == 9
            and all(row["pass"] for row in observations)
            else "FAIL"
        ),
        "reference_artifact": str(reference_artifact),
        "reference_graph_sha256": reference.metadata["runtime"][
            "graph_sha256"
        ],
        "candidate_artifact": str(candidate_artifact),
        "candidate_graph_sha256": candidate.metadata["runtime"][
            "graph_sha256"
        ],
        "validation_evidence_sha256": _sha256_file(
            validation_evidence_path
        ),
        "threads": int(threads),
        "routes_checked": sorted(selected),
        "decode_steps_per_route_max": int(decode_steps),
        "all_arrays_bit_exact": all(
            row["pass"] for row in observations
        ),
        "observations": observations,
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-artifact", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--validation-evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=16)
    args = parser.parse_args(argv)
    result = verify_state_equivalence(
        reference_artifact=args.reference_artifact,
        candidate_artifact=args.candidate_artifact,
        validation_evidence_path=args.validation_evidence,
        output_path=args.output,
        threads=args.threads,
        decode_steps=args.decode_steps,
    )
    display = {
        key: value
        for key, value in result.items()
        if key != "observations"
    }
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
