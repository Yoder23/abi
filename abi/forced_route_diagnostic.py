"""Measure LayerCake generation with the disclosed expected capability route."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .english_generalization_evaluation import (
    _collapse_metrics,
    _normalize_decoding_contract,
)
from .hf_extraction import evaluate_output, load_probe_catalog
from .layercake_core_loader import (
    CAPABILITY_CAKE_ORDER,
    load_layercake_core,
)
from .layercake_host import (
    _canonical_json_bytes,
    _generate_host,
    _sha256_file,
    _write_json,
)


def run_forced_route_diagnostic(
    *,
    catalog_path: str | Path,
    split: str,
    layercake_root: str | Path,
    standalone_core_path: str | Path,
    output_path: str | Path,
    device_name: str,
    limit_per_capability: int,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    """Force only the expected internal route; leave generation unchanged."""

    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise RuntimeError(f"diagnostic evidence is immutable: {output_path}")
    catalog_path = Path(catalog_path).resolve()
    catalog = load_probe_catalog(catalog_path)
    counts: Counter[str] = Counter()
    probes = []
    for probe in catalog["probes"]:
        capability = str(probe["capability"])
        if (
            probe["split"] == split
            and counts[capability] < limit_per_capability
        ):
            probes.append(probe)
            counts[capability] += 1
    device = torch.device(device_name)
    core_path = Path(standalone_core_path).resolve()
    model, tokenizer, manifest = load_layercake_core(
        core_path,
        layercake_root=Path(layercake_root).resolve(),
        device=device,
    )
    model.eval()
    decoding = _normalize_decoding_contract(
        getattr(model, "_abi_decoding", None)
    )
    decoding["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    model._abi_decoding = decoding
    capability_index = {
        capability: index
        for index, capability in enumerate(CAPABILITY_CAKE_ORDER)
    }
    observations = []
    with torch.no_grad():
        model.capability_router.weight.zero_()
        for probe in probes:
            capability = str(probe["capability"])
            internal_route = capability_index[capability]
            model.capability_router.bias.fill_(-100.0)
            model.capability_router.bias[internal_route] = 100.0
            output, token_ids, canonical_route, seconds = _generate_host(
                model,
                tokenizer,
                str(probe["prompt"]),
                max_new_tokens=int(probe["max_new_tokens"]),
                device=device,
            )
            passed, score = evaluate_output(output, probe["evaluator"])
            observations.append(
                {
                    "probe_id": str(probe["probe_id"]),
                    "capability": capability,
                    "forced_internal_route": internal_route,
                    "forced_canonical_route": canonical_route,
                    "output": output,
                    "output_sha256": hashlib.sha256(
                        output.encode("utf-8")
                    ).hexdigest(),
                    "generated_tokens": len(token_ids),
                    "passed": bool(passed),
                    "score": float(score),
                    "generation_seconds": seconds,
                    "collapse": _collapse_metrics(
                        token_ids,
                        output,
                        tokenizer.encode(str(probe["prompt"]) + "\n"),
                        str(probe["prompt"]),
                    ),
                }
            )
    passes = sum(row["passed"] for row in observations)
    collapses = sum(
        row["collapse"]["collapse_detected"] for row in observations
    )
    evidence: dict[str, Any] = {
        "format": "abi-layercake-forced-capability-route-diagnostic/1",
        "status": "DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "split": split,
        },
        "candidate": {
            "path": str(core_path),
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "intervention": {
            "router_weight_zeroed_in_memory": True,
            "router_bias_forced_to_expected_internal_capability": True,
            "checkpoint_changed": False,
            "generation_weights_changed": False,
            "decoding_no_repeat_ngram_size": no_repeat_ngram_size,
        },
        "observation_count": len(observations),
        "passes": passes,
        "collapse_count": collapses,
        "observations": observations,
        "claim_boundary": (
            "This isolates the upper bound from correct capability routing. "
            "It is not autonomous-routing or promotion evidence."
        ),
        "contaminated_final_v4_accessed": False,
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--standalone-core", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-per-capability", type=int, default=2)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    args = parser.parse_args(argv)
    evidence = run_forced_route_diagnostic(
        catalog_path=args.catalog,
        split=args.split,
        layercake_root=args.layercake_root,
        standalone_core_path=args.standalone_core,
        output_path=args.output,
        device_name=args.device,
        limit_per_capability=args.limit_per_capability,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(
        json.dumps(
            {
                "passes": evidence["passes"],
                "collapse_count": evidence["collapse_count"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
