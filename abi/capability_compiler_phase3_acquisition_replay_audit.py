"""Paired acquisition-replay and prompt-projection audit for V474/V484."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import zipfile

from safetensors.torch import load_file
import torch

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_targeted_recovery_bridge import _generate_enforced, _load_parent
from .capability_compiler_phase3_weak_residual import SharedWeakResidual, WEAK_CAPABILITIES, _attach
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase3-acquisition-replay-audit/1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_PAIRED_ACQUISITION_REPLAY_ATTRIBUTION"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("acquisition replay governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"acquisition replay binding changed: {relative}")
    return protocol, sha256_file(path)


def _artifact_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as archive:
        return [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line]


def selected_records(rows: list[dict[str, Any]], per_stratum: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["capability"]), int(row["builder"]))].append(row)
    selected = []
    for capability in WEAK_CAPABILITIES:
        for builder in range(4):
            values = sorted(grouped[(capability, builder)], key=lambda row: str(row["record_id"]))
            if len(values) < per_stratum:
                raise Phase3Error("acquisition replay depth changed")
            selected.extend(values[:per_stratum])
    return selected


def _required_literals(evaluator: Mapping[str, Any]) -> list[str]:
    kind = evaluator["kind"]
    if kind in {"contains_all", "ordered_contains"}:
        return [str(value) for value in evaluator["values"]]
    if kind == "exact":
        return [str(evaluator["value"])]
    if kind == "all_of":
        return [value for rule in evaluator["rules"] for value in _required_literals(rule)]
    return []


def literal_recall(output: str, evaluator: Mapping[str, Any]) -> float | None:
    values = _required_literals(evaluator)
    if not values:
        return None
    lowered = output.casefold()
    return sum(value.casefold() in lowered for value in values) / len(values)


def _load_candidate(root: Path, protocol: dict[str, Any], spec: Mapping[str, Any], device: torch.device):
    model, tokenizer, _ = _load_parent(root, protocol, device)
    residual = SharedWeakResidual().to(device)
    residual.load_state_dict(load_file(str(root / spec["checkpoint"]), device="cuda"), strict=True)
    residual.eval()
    handles = _attach(model, residual)
    return model, tokenizer, handles


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable acquisition replay output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("acquisition replay CUDA unavailable")
    artifact_rows = _artifact_rows(root / protocol["supervision"]["artifact"])
    selected = selected_records(artifact_rows, int(protocol["sample"]["records_per_stratum"]))
    catalog = _json(root / protocol["supervision"]["source_catalog"])["probes"]
    probe_by_id = {str(row["probe_id"]): row for row in catalog}
    device = torch.device("cuda")
    observations = []
    for candidate_name, candidate_spec in protocol["candidates"].items():
        model, tokenizer, handles = _load_candidate(root, protocol, candidate_spec, device)
        for index, row in enumerate(selected):
            probe = probe_by_id[str(row["probe_id"])]
            for prompt_mode, prompt in (("host_projected", str(row["host_prompt"])), ("source_wrapped", str(probe["prompt"]))):
                value, tokens, task_route = _generate_enforced(model, tokenizer, prompt, int(probe["max_new_tokens"]), str(row["capability"]), device)
                observations.append({
                    "candidate": candidate_name,
                    "prompt_mode": prompt_mode,
                    "record_id": str(row["record_id"]),
                    "probe_id": str(row["probe_id"]),
                    "capability": str(row["capability"]),
                    "builder": int(row["builder"]),
                    "task_route": task_route,
                    "output": value,
                    "output_token_ids": tokens,
                    "target_exact": value == str(row["output"]),
                    "functional_v1": evaluate_functional(value, row["functional_evaluator"]),
                    "functional_v2": evaluate_functional_v2(value, row["functional_evaluator"], str(row["capability"])),
                    "required_literal_recall": literal_recall(value, row["functional_evaluator"]),
                    "repetition_collapse_v2": repetition_collapse_v2(value),
                })
            if (index + 1) % 80 == 0:
                print(json.dumps({"candidate": candidate_name, "records": index + 1}), flush=True)
        for handle in handles:
            handle.remove()
        del model
        torch.cuda.empty_cache()
    summary: dict[str, Any] = {}
    for candidate_name in protocol["candidates"]:
        summary[candidate_name] = {}
        for mode in ("host_projected", "source_wrapped"):
            summary[candidate_name][mode] = {}
            for capability in WEAK_CAPABILITIES:
                summary[candidate_name][mode][capability] = {}
                for builder in range(4):
                    values = [row for row in observations if row["candidate"] == candidate_name and row["prompt_mode"] == mode and row["capability"] == capability and row["builder"] == builder]
                    recalls = [float(row["required_literal_recall"]) for row in values if row["required_literal_recall"] is not None]
                    summary[candidate_name][mode][capability][str(builder)] = {
                        "observations": len(values),
                        "functional_v1_passes": sum(row["functional_v1"] for row in values),
                        "functional_v2_passes": sum(row["functional_v2"] for row in values),
                        "target_exact": sum(row["target_exact"] for row in values),
                        "repetition_collapses_v2": sum(row["repetition_collapse_v2"] for row in values),
                        "mean_required_literal_recall": sum(recalls) / len(recalls) if recalls else None,
                    }
    v484_host = [row for row in observations if row["candidate"] == "V484" and row["prompt_mode"] == "host_projected"]
    v484_wrapped = [row for row in observations if row["candidate"] == "V484" and row["prompt_mode"] == "source_wrapped"]
    host_rate = sum(row["functional_v1"] for row in v484_host) / len(v484_host)
    wrapped_rate = sum(row["functional_v1"] for row in v484_wrapped) / len(v484_wrapped)
    threshold = float(protocol["decision_rule"]["acquisition_fit_minimum"])
    if host_rate < threshold:
        attribution = "ACQUISITION_AUTONOMOUS_FIT_FAILURE"
    elif wrapped_rate + float(protocol["decision_rule"]["material_prompt_mode_delta"]) < host_rate:
        attribution = "PROMPT_PROJECTION_DISTRIBUTION_FAILURE"
    else:
        attribution = "HELDOUT_NONCE_AND_TASK_GENERALIZATION_FAILURE"
    raw = output.parent / "observations.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in observations))
    result = {
        "format": FORMAT,
        "status": f"PASS_ATTRIBUTED_{attribution}",
        "protocol_sha256": protocol_sha,
        "sample_records": len(selected),
        "generation_observations": len(observations),
        "summary": summary,
        "V484_host_functional_v1_rate": host_rate,
        "V484_source_wrapped_functional_v1_rate": wrapped_rate,
        "attribution": attribution,
        "raw_observations_sha256": sha256_file(raw),
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "artifact_mutated": False,
        "final_test_accessed": False,
        "phase3_certified": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ACQUISITION_REPLAY_AUDIT_PROTOCOL_V485.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_acquisition_replay/audit_v486/result.json"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output); print(json.dumps({"status": result["status"], "V484_host_functional_v1_rate": result["V484_host_functional_v1_rate"], "V484_source_wrapped_functional_v1_rate": result["V484_source_wrapped_functional_v1_rate"], "evidence_sha256": result["evidence_sha256"]}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
