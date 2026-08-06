"""Read-only V40 attribution of the sealed V38 BPE-core failure."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_bpe_core import _load_candidate, _json, load_protocol as load_bpe_protocol


FORMAT = "abi-capability-compiler-phase3-bpe-failure-attribution/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY"
        or protocol.get("training_allowed") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("V40 governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V40 binding changed: {relative}")
    base_path = (root / protocol["base_protocol"]["path"]).resolve()
    base, base_sha = load_bpe_protocol(root, base_path)
    if base_sha != protocol["base_protocol"]["sha256"]:
        raise Phase3Error("V40 base protocol changed")
    return protocol, base, sha256_file(path)


def fact_free_mode(output: str) -> bool:
    return "PAVO-" in output and "NORU-" in output


def classify(
    *,
    training_exact_rate: float,
    same_header_delta: float,
    body_only_delta: float,
    thresholds: Mapping[str, Any],
) -> str:
    if training_exact_rate < float(thresholds["training_autonomous_exact_minimum"]):
        return "PRIMARY_MODEL_FIT_OR_AUTONOMOUS_STATE"
    if max(same_header_delta, body_only_delta) >= float(thresholds["header_intervention_improvement_minimum"]):
        return "PRIMARY_ACQUISITION_TO_EVALUATION_HEADER_COVARIATE_SHIFT"
    return "PRIMARY_SEMANTIC_ROUTING_AND_HELDOUT_GENERALIZATION"


def _generate(model: Any, prompt: str, maximum_actions: int) -> tuple[str, str | None]:
    try:
        return model.generate_bytes(prompt, maximum_actions=maximum_actions).decode("utf-8", errors="strict"), None
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def execute(root: Path, protocol_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, base, protocol_sha = load_protocol(root, protocol_path)
    candidate_dir = (root / protocol["candidate_dir"]).resolve()
    metadata = _json(candidate_dir / "metadata.json")
    if (
        sha256_file(candidate_dir / "metadata.json") != protocol["candidate"]["metadata_sha256"]
        or sha256_file(candidate_dir / "model.safetensors") != protocol["candidate"]["checkpoint_sha256"]
        or metadata["checkpoint"]["sha256"] != protocol["candidate"]["checkpoint_sha256"]
    ):
        raise Phase3Error("V40 candidate identity changed")
    model, _ = _load_candidate(root, base, candidate_dir, torch.device("cuda"))
    observations = int(protocol["diagnostic"]["observations_per_capability"])
    maximum_actions = int(protocol["diagnostic"]["maximum_actions"])
    acquisition = load_phase1_ir((root / base["phase1_ir"]).resolve())
    probes = development_probes((root / base["development_catalog"]).resolve())

    header_by_capability: dict[str, str] = {}
    training_rows: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        values = sorted(
            (row for row in acquisition if row["capability"] == capability),
            key=lambda row: str(row["ir_record_id"]),
        )
        header_by_capability[capability] = str(values[0]["normalized_acquisition_prompt"]).splitlines()[0]
        training_rows.extend(values[:observations])

    raw: list[dict[str, Any]] = []
    for index, row in enumerate(training_rows):
        output, error = _generate(model, str(row["normalized_acquisition_prompt"]), maximum_actions)
        raw.append(
            {
                "scope": "training_replay",
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "variant": "original",
                "output": output,
                "generation_error": error,
                "exact_bytes": output.encode("utf-8") == str(row["normalized_output"]).encode("utf-8") and error is None,
                "fact_free_mode": fact_free_mode(output),
                "repetition_collapse": repetition_collapse(output),
            }
        )
        if (index + 1) % 100 == 0:
            print(json.dumps({"training_replay": index + 1}), flush=True)

    selected_probes: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        values = sorted(
            (probe for probe in probes if probe["canonical_capability"] == capability),
            key=lambda probe: str(probe["probe_id"]),
        )
        selected_probes.extend(values[:observations])
    for index, probe in enumerate(selected_probes):
        prompt = str(probe["prompt"])
        lines = prompt.splitlines()
        body = "\n".join(lines[1:])
        variants = {
            "original": prompt,
            "body_only": body,
            "capability_matched_acquisition_header": header_by_capability[str(probe["canonical_capability"])] + "\n" + body,
        }
        for variant, value in variants.items():
            output, error = _generate(model, value, maximum_actions)
            raw.append(
                {
                    "scope": "development_intervention",
                    "record_id": str(probe["probe_id"]),
                    "capability": str(probe["canonical_capability"]),
                    "variant": variant,
                    "output": output,
                    "generation_error": error,
                    "functional_pass": evaluate_functional(output, probe["evaluator"]),
                    "fact_free_mode": fact_free_mode(output),
                    "repetition_collapse": repetition_collapse(output),
                }
            )
        if (index + 1) % 100 == 0:
            print(json.dumps({"development_prompts": index + 1}), flush=True)

    training = [row for row in raw if row["scope"] == "training_replay"]
    development = [row for row in raw if row["scope"] == "development_intervention"]
    variants: dict[str, Any] = {}
    for variant in ("original", "body_only", "capability_matched_acquisition_header"):
        values = [row for row in development if row["variant"] == variant]
        passes = sum(row["functional_pass"] is True for row in values)
        variants[variant] = {
            "observations": len(values),
            "functional_passes": passes,
            "functional_rate": passes / len(values),
            "fact_free_mode": sum(row["fact_free_mode"] is True for row in values),
            "generation_errors": sum(row["generation_error"] is not None for row in values),
            "repetition_collapses": sum(row["repetition_collapse"] is True for row in values),
        }
    original_rate = variants["original"]["functional_rate"]
    training_exact = sum(row["exact_bytes"] is True for row in training) / len(training)
    classification = classify(
        training_exact_rate=training_exact,
        same_header_delta=variants["capability_matched_acquisition_header"]["functional_rate"] - original_rate,
        body_only_delta=variants["body_only"]["functional_rate"] - original_rate,
        thresholds=protocol["classification_thresholds"],
    )
    result: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-bpe-failure-attribution-result/1",
        "status": "PASS_READ_ONLY_ATTRIBUTION",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "candidate_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "training_replay": {
            "observations": len(training),
            "exact_bytes": sum(row["exact_bytes"] is True for row in training),
            "exact_rate": training_exact,
            "fact_free_mode": sum(row["fact_free_mode"] is True for row in training),
            "generation_errors": sum(row["generation_error"] is not None for row in training),
            "repetition_collapses": sum(row["repetition_collapse"] is True for row in training),
        },
        "development_interventions": variants,
        "classification": classification,
        "training_performed": False,
        "checkpoint_changed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "next_rule": protocol["post_diagnostic_rule"],
        "claim_boundary": "Read-only attribution of one sealed failed ABI candidate; no LayerCake regression or quality claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result, raw


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "verify"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_BPE_FAILURE_ATTRIBUTION_PROTOCOL_V40.json")
    parser.add_argument("--result", default="results/abi_capability_compiler_phase3_bpe_failure_attribution/attribution_v40.json")
    parser.add_argument("--rows", default="results/abi_capability_compiler_phase3_bpe_failure_attribution/rows_v40.jsonl")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result, rows = execute(root, (root / args.protocol).resolve())
    result_path = (root / args.result).resolve()
    rows_path = (root / args.rows).resolve()
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    result["rows_sha256"] = hashlib.sha256(payload).hexdigest()
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes({key: value for key, value in result.items() if key != "evidence_sha256"})).hexdigest()
    if args.command == "write":
        _write_immutable(rows_path, payload)
        _write_immutable(result_path, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    elif _json(result_path) != result or rows_path.read_bytes() != payload:
        raise Phase3Error("stored V40 evidence differs from recomputation")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
