"""Evaluate one exact LayerCake host on an arbitrary locked English catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any

from .capability_pipeline import read_extraction_bundle
from .hf_extraction import evaluate_output, load_probe_catalog
from .layercake_host import (
    CAPABILITY_TO_ROUTE,
    _canonical_json_bytes,
    _generate_host,
    _sha256_file,
)
from .layercake_host_v3 import load_host_model
from .layercake_core_loader import load_layercake_core


EVIDENCE_FORMAT = "abi-layercake-english-generalization-evidence/6"


def _normalize_decoding_contract(
    decoding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fill additive decoding fields for older, otherwise valid host manifests."""

    normalized = {
        "algorithm": "greedy",
        "no_repeat_ngram_size": 0,
        "allow_prompt_ngrams": False,
        "lexical_repetition_truncation_threshold": 0,
        "prompt_identity_mixture": False,
    }
    if decoding is not None:
        normalized.update(decoding)
    return normalized


def _collapse_metrics(
    token_ids: Sequence[int],
    output: str,
    prompt_token_ids: Sequence[int] = (),
    prompt: str = "",
) -> dict[str, Any]:
    repeated_runs = 0
    previous: int | None = None
    run = 0
    maximum_run = 0
    for token_id in token_ids:
        if token_id == previous:
            run += 1
        else:
            run = 1
            previous = token_id
        maximum_run = max(maximum_run, run)
        if run == 4:
            repeated_runs += 1
    fourgrams = [
        tuple(token_ids[index : index + 4])
        for index in range(max(0, len(token_ids) - 3))
    ]
    prompt_fourgrams = {
        tuple(prompt_token_ids[index : index + 4])
        for index in range(max(0, len(prompt_token_ids) - 3))
    }
    fourgram_counts = Counter(fourgrams)
    repeated_fourgrams_total = sum(
        count - 1 for count in fourgram_counts.values() if count > 1
    )
    repeated_prompt_fourgrams = sum(
        count - 1
        for value, count in fourgram_counts.items()
        if count > 1 and value in prompt_fourgrams
    )
    repeated_fourgrams = sum(
        count - 1
        for value, count in fourgram_counts.items()
        if count > 1 and value not in prompt_fourgrams
    )
    output_words = re.findall(r"[\w']+", output.casefold())
    prompt_words = re.findall(r"[\w']+", prompt.casefold())
    output_word_fourgrams = [
        tuple(output_words[index : index + 4])
        for index in range(max(0, len(output_words) - 3))
    ]
    prompt_word_fourgrams = {
        tuple(prompt_words[index : index + 4])
        for index in range(max(0, len(prompt_words) - 3))
    }
    repeated_lexical_fourgrams = sum(
        count - 1
        for value, count in Counter(output_word_fourgrams).items()
        if count > 1 and value not in prompt_word_fourgrams
    )
    unique_ratio = len(set(token_ids)) / max(1, len(token_ids))
    collapsed = (
        not output.strip()
        or maximum_run >= 6
        or repeated_lexical_fourgrams >= 4
        or (len(token_ids) >= 20 and unique_ratio < 0.2)
    )
    return {
        "empty_output": not output.strip(),
        "maximum_identical_token_run": maximum_run,
        "repeated_fourgram_occurrences": repeated_fourgrams,
        "repeated_fourgram_occurrences_total": repeated_fourgrams_total,
        "repeated_prompt_copied_fourgram_occurrences": (
            repeated_prompt_fourgrams
        ),
        "repeated_lexical_fourgram_occurrences": (
            repeated_lexical_fourgrams
        ),
        "unique_token_ratio": unique_ratio,
        "collapse_detected": collapsed,
        "repeated_run_events": repeated_runs,
    }


def _source_by_probe(
    source_bundle_paths: Sequence[str | Path] | None,
    *,
    split: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_probe: dict[str, dict[str, Any]] = {}
    identities: list[dict[str, Any]] = []
    for source_bundle_path in source_bundle_paths or ():
        bundle = read_extraction_bundle(source_bundle_path)
        records = {
            str(record["record_id"]): record
            for record in bundle["records"]
            if record["split"] == split
        }
        for result in bundle["probe_results"]:
            record = records.get(str(result["record_id"]))
            if record is None:
                continue
            probe_id = str(result["probe_id"])
            if probe_id in by_probe:
                raise RuntimeError(
                    f"ambiguous source evidence for {probe_id}"
                )
            by_probe[probe_id] = {
                "passed": bool(result["passed"]),
                "score": float(result["score"]),
                "output": str(record["output"]),
                "output_sha256": str(record["output_sha256"]),
                "teacher_tokens": int(record["teacher_tokens"]),
                "source_model": str(record["source_model"]),
                "source_model_revision": str(record["source_model_revision"]),
            }
        identities.append(
            {
                "path_at_evaluation": str(Path(source_bundle_path).resolve()),
                "archive_sha256": bundle["verification"]["archive_sha256"],
                "manifest_sha256": bundle["verification"]["manifest_sha256"],
            }
        )
    return by_probe, identities


def evaluate_generalization(
    *,
    catalog_path: str | Path,
    split: str,
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    host_path: str | Path | None,
    standalone_core_path: str | Path | None,
    source_bundle_paths: Sequence[str | Path] | None,
    output_path: str | Path,
    device_name: str,
    no_repeat_ngram_size: int | None = None,
    allow_prompt_ngrams: bool | None = None,
    lexical_repetition_truncation_threshold: int | None = None,
    limit_per_capability: int | None = None,
) -> dict[str, Any]:
    if split not in {"search", "validation", "final_test"}:
        raise ValueError("split must be search, validation, or final_test")
    if host_path is not None and standalone_core_path is not None:
        raise ValueError("host and standalone core are mutually exclusive")
    if no_repeat_ngram_size is not None and no_repeat_ngram_size < 0:
        raise ValueError("no-repeat n-gram size must be non-negative")
    if (
        lexical_repetition_truncation_threshold is not None
        and lexical_repetition_truncation_threshold < 0
    ):
        raise ValueError(
            "lexical repetition truncation threshold must be non-negative"
        )
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise RuntimeError(f"evaluation evidence is immutable: {output_path}")
    catalog_path = Path(catalog_path).resolve()
    catalog = load_probe_catalog(catalog_path)
    probes = [probe for probe in catalog["probes"] if probe["split"] == split]
    if limit_per_capability is not None:
        if limit_per_capability <= 0:
            raise ValueError("limit_per_capability must be positive")
        counts: Counter[str] = Counter()
        selected = []
        for probe in probes:
            capability = str(probe["capability"])
            if counts[capability] < limit_per_capability:
                selected.append(probe)
                counts[capability] += 1
        probes = selected
    source, source_identities = _source_by_probe(
        source_bundle_paths, split=split
    )
    if source and {str(probe["probe_id"]) for probe in probes} - set(source):
        raise RuntimeError("source bundle is incomplete for selected probes")

    if standalone_core_path is not None:
        device = __import__("torch").device(device_name)
        model, tokenizer, manifest = load_layercake_core(
            Path(standalone_core_path).resolve(),
            layercake_root=Path(layercake_root).resolve(),
            device=device,
        )
        if (
            manifest.get("format")
            not in {
                "abi-layercake-full-english-core-acquisition/1",
                "abi-layercake-component-graft/1",
                "abi-layercake-direct-source-initialization/1",
            }
            or manifest.get("canonical_semantic_abi", {}).get("sha256")
            != _sha256_file(Path(canonical_abi_path))
        ):
            raise RuntimeError("standalone core identity or canonical ABI changed")
    else:
        model, tokenizer, manifest, device = load_host_model(
            layercake_root=layercake_root,
            parent_path=parent_path,
            canonical_abi_path=canonical_abi_path,
            host_path=host_path,
            device_name=device_name,
        )
    decoding = _normalize_decoding_contract(
        getattr(model, "_abi_decoding", None)
    )
    if no_repeat_ngram_size is not None:
        decoding["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    if allow_prompt_ngrams is not None:
        decoding["allow_prompt_ngrams"] = bool(allow_prompt_ngrams)
    if lexical_repetition_truncation_threshold is not None:
        decoding["lexical_repetition_truncation_threshold"] = int(
            lexical_repetition_truncation_threshold
        )
    if (
        decoding["allow_prompt_ngrams"]
        and int(decoding["no_repeat_ngram_size"]) <= 0
    ):
        raise ValueError(
            "prompt n-gram exemptions require a no-repeat n-gram size"
        )
    model._abi_decoding = decoding
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, probe in enumerate(probes, start=1):
        generated_started = time.perf_counter()
        output, token_ids, route, model_seconds = _generate_host(
            model,
            tokenizer,
            str(probe["prompt"]),
            max_new_tokens=int(probe["max_new_tokens"]),
            device=device,
        )
        passed, score = evaluate_output(output, probe["evaluator"])
        source_row = source.get(str(probe["probe_id"]))
        observation = {
            "probe_id": str(probe["probe_id"]),
            "capability": str(probe["capability"]),
            "prompt": str(probe["prompt"]),
            "evaluator": probe["evaluator"],
            "layercake_output": output,
            "layercake_output_sha256": hashlib.sha256(
                output.encode("utf-8")
            ).hexdigest(),
            "layercake_generated_tokens": len(token_ids),
            "layercake_passed": passed,
            "layercake_score": score,
            "automatic_route": route,
            "expected_route": CAPABILITY_TO_ROUTE[str(probe["capability"])],
            "route_correct": (
                route == CAPABILITY_TO_ROUTE[str(probe["capability"])]
            ),
            "model_generation_seconds": model_seconds,
            "observation_seconds": time.perf_counter() - generated_started,
            "collapse": _collapse_metrics(
                token_ids,
                output,
                tokenizer.encode(str(probe["prompt"]) + "\n"),
                str(probe["prompt"]),
            ),
        }
        if source_row is not None:
            observation["source"] = source_row
            observation["source_passing_regression"] = (
                source_row["passed"] and not passed
            )
        observations.append(observation)
        if ordinal % 100 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": ordinal,
                        "total": len(probes),
                        "passes": sum(
                            row["layercake_passed"] for row in observations
                        ),
                    }
                ),
                flush=True,
            )

    metrics: dict[str, Any] = {}
    for capability in sorted({row["capability"] for row in observations}):
        selected = [
            row for row in observations if row["capability"] == capability
        ]
        source_passes = sum(
            row.get("source", {}).get("passed", False) for row in selected
        )
        regressions = sum(
            row.get("source_passing_regression", False) for row in selected
        )
        metrics[capability] = {
            "observations": len(selected),
            "layercake_passes": sum(
                row["layercake_passed"] for row in selected
            ),
            "layercake_pass_rate": sum(
                row["layercake_passed"] for row in selected
            )
            / len(selected),
            "source_passes": source_passes if source else None,
            "source_passing_regressions": regressions if source else None,
            "source_passing_retention_rate": (
                (source_passes - regressions) / source_passes
                if source and source_passes
                else None
            ),
            "route_accuracy": sum(row["route_correct"] for row in selected)
            / len(selected),
            "collapse_count": sum(
                row["collapse"]["collapse_detected"] for row in selected
            ),
        }
    source_passes = sum(
        row.get("source", {}).get("passed", False) for row in observations
    )
    source_regressions = sum(
        row.get("source_passing_regression", False) for row in observations
    )
    layercake_passes = sum(row["layercake_passed"] for row in observations)
    evidence: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "status": "EVALUATED_NOT_PROMOTED",
        "split": split,
        "final_test_accessed": split == "final_test",
        "catalog": {
            "path_at_evaluation": str(catalog_path),
            "file_sha256": _sha256_file(catalog_path),
            "catalog_id": catalog["catalog_id"],
        },
        "candidate": {
            "kind": (
                "standalone_acquired_core"
                if standalone_core_path is not None
                else "layercake_host"
                if manifest is not None
                else "sealed_parent"
            ),
            "host_path_at_evaluation": (
                str(Path(host_path).resolve()) if host_path is not None else None
            ),
            "standalone_core_path_at_evaluation": (
                str(Path(standalone_core_path).resolve())
                if standalone_core_path is not None
                else None
            ),
            "host_manifest_sha256": (
                manifest.get("manifest_sha256")
                if manifest is not None
                else None
            ),
            "parent_path_at_evaluation": str(Path(parent_path).resolve()),
            "parent_checkpoint_sha256": _sha256_file(
                Path(parent_path) / "model.safetensors"
            ),
            "teacher_present_at_inference": False,
            "decoding": {
                "algorithm": "greedy",
                "no_repeat_ngram_size": int(
                    decoding["no_repeat_ngram_size"]
                ),
                "allow_prompt_ngrams": bool(
                    decoding["allow_prompt_ngrams"]
                ),
                "lexical_repetition_truncation_threshold": int(
                    decoding["lexical_repetition_truncation_threshold"]
                ),
            },
        },
        "sources": source_identities,
        "observation_count": len(observations),
        "layercake_passes": layercake_passes,
        "layercake_pass_rate": layercake_passes / len(observations),
        "source_passes": source_passes if source else None,
        "source_passing_regressions": source_regressions if source else None,
        "source_passing_retention_rate": (
            (source_passes - source_regressions) / source_passes
            if source and source_passes
            else None
        ),
        "collapse_count": sum(
            row["collapse"]["collapse_detected"] for row in observations
        ),
        "capability_metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "observations": observations,
        "claim_boundary": (
            "This is bounded functional evidence for one exact LayerCake "
            "candidate. Automated evaluators do not substitute for the locked "
            "blinded human fluency audit or runtime certification."
        ),
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument(
        "--split",
        choices=("search", "validation", "final_test"),
        required=True,
    )
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--host")
    parser.add_argument("--standalone-core")
    parser.add_argument("--source-bundle", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit-per-capability", type=int)
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument(
        "--allow-prompt-ngrams",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--lexical-repetition-truncation-threshold",
        type=int,
    )
    args = parser.parse_args(argv)
    evidence = evaluate_generalization(
        catalog_path=args.catalog,
        split=args.split,
        layercake_root=args.layercake_root,
        parent_path=args.parent,
        canonical_abi_path=args.canonical_abi,
        host_path=args.host,
        standalone_core_path=args.standalone_core,
        source_bundle_paths=args.source_bundle,
        output_path=args.output,
        device_name=args.device,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        allow_prompt_ngrams=args.allow_prompt_ngrams,
        lexical_repetition_truncation_threshold=(
            args.lexical_repetition_truncation_threshold
        ),
        limit_per_capability=args.limit_per_capability,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "passes": evidence["layercake_passes"],
                "observations": evidence["observation_count"],
                "source_passing_retention_rate": evidence[
                    "source_passing_retention_rate"
                ],
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
