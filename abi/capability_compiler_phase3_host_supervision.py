"""Build an immutable host-conformant weak-capability supervision artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import zipfile

from .capability_compiler_functional_v2 import NUMBER_WORDS, evaluate_functional_v2
from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_broad_ir import host_prompt_projection
from .capability_compiler_phase3_failure_to_supervision_audit import builder_index
from .capability_compiler_phase3_weak_residual import WEAK_CAPABILITIES
from .capability_compiler_phase1_ir import normalize_text


FORMAT = "abi-capability-compiler-phase3-host-supervision/1"
RECORD_FORMAT = "abi-host-conformant-weak-supervision-record/1"
ALLOWED_ENTRIES = ("accounting.json", "manifest.json", "records.jsonl")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_CACHED_EVIDENCE_HOST_SUPERVISION_BUILD"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("host supervision governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"host supervision binding changed: {relative}")
    return protocol, sha256_file(path)


def surface_repair(output: str, evaluator: Mapping[str, Any], capability: str) -> tuple[str, tuple[str, ...]]:
    """Apply only explicit semantic-preserving V2-to-V1 surface substitutions."""

    if evaluate_functional(output, evaluator):
        return output, ()
    if not evaluate_functional_v2(output, dict(evaluator), capability):
        raise Phase3Error("surface repair received a V2-invalid source output")
    value = output
    changes: list[str] = []
    for word, digit in NUMBER_WORDS.items():
        updated = re.sub(rf"\b{re.escape(word)}\b", digit, value, flags=re.IGNORECASE)
        if updated != value:
            changes.append(f"number_word_{word}_to_{digit}")
            value = updated
    if capability == "abstention":
        for pattern in (r"\bcannot be known\b", r"\bcan not be known\b", r"\bcan't be known\b"):
            updated = re.sub(pattern, "cannot determine", value, flags=re.IGNORECASE)
            if updated != value:
                changes.append("abstention_cannot_be_known_to_cannot_determine")
                value = updated
    if not evaluate_functional(value, evaluator):
        raise Phase3Error("allowed V2-to-V1 surface substitutions did not recover functional validity")
    return value, tuple(changes)


def _record_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(row)).hexdigest()


def _eligible(
    catalog: list[dict[str, Any]], attempts: list[dict[str, Any]]
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    probe_by_id = {str(row["probe_id"]): row for row in catalog}
    by_probe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        by_probe[str(attempt["probe_id"])].append(attempt)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for probe_id, values in by_probe.items():
        probe = probe_by_id.get(probe_id)
        if probe is None or probe.get("split") != "search":
            raise Phase3Error("teacher journal crossed the bound search catalog")
        capability = str(values[0]["canonical_capability"])
        if capability not in WEAK_CAPABILITIES:
            continue
        builder = builder_index(str(probe["phase3_targeted_template_family"]))
        if builder is None:
            raise Phase3Error("weak source record lost builder identity")
        chosen = None
        for attempt in sorted(values, key=lambda row: (int(row["attempt_index"]), str(row["attempt_sha256"]))):
            if attempt.get("finish_reason") != "eos_token":
                continue
            normalized_output, output_steps = normalize_text(str(attempt["output"]))
            if not evaluate_functional_v2(normalized_output, attempt["functional_evaluator"], capability):
                continue
            try:
                repaired_output, surface_steps = surface_repair(normalized_output, attempt["functional_evaluator"], capability)
            except Phase3Error:
                continue
            normalized_prompt, prompt_steps = normalize_text(str(probe["prompt"]))
            host_prompt, projection = host_prompt_projection(capability, normalized_prompt)
            chosen = {
                "format": RECORD_FORMAT,
                "probe_id": probe_id,
                "split": "acquisition",
                "capability": capability,
                "builder": builder,
                "template_family": str(probe["phase3_targeted_template_family"]),
                "host_prompt": host_prompt,
                "host_prompt_sha256": hashlib.sha256(host_prompt.encode()).hexdigest(),
                "output": repaired_output,
                "output_sha256": hashlib.sha256(repaired_output.encode()).hexdigest(),
                "functional_evaluator": attempt["functional_evaluator"],
                "functional_v1_pass": True,
                "source_functional_v2_pass": True,
                "source_model": "microsoft/Phi-3-mini-4k-instruct",
                "source_revision": "f39ac1d28e925b323eae81227eaba4464caced4e",
                "source_attempt_sha256": str(attempt["attempt_sha256"]),
                "source_attempt_index": int(attempt["attempt_index"]),
                "source_attempt_kind": str(attempt["kind"]),
                "source_generation_prompt_sha256": str(attempt["generation_prompt_sha256"]),
                "source_output_sha256": str(attempt["output_sha256"]),
                "source_teacher_input_tokens": int(attempt["teacher_input_tokens"]),
                "source_teacher_output_tokens": int(attempt["teacher_tokens"]),
                "source_authoritative_generated_token_ids": [int(value) for value in attempt["authoritative_generated_token_ids"]],
                "prompt_projection": projection,
                "prompt_normalization_steps": prompt_steps,
                "output_normalization_steps": output_steps,
                "surface_normalization_steps": list(surface_steps),
                "teacher_generation_prompt_excluded_from_host_input": True,
                "teacher_present_at_host_training_or_inference": False,
            }
            payload = dict(chosen)
            chosen["record_id"] = _record_hash(payload)
            break
        if chosen is not None:
            grouped[(capability, builder)].append(chosen)
    return grouped


def _selection_key(row: Mapping[str, Any], protocol_sha: str) -> str:
    return hashlib.sha256(
        f"{protocol_sha}:{row['probe_id']}:{row['source_attempt_sha256']}".encode("ascii")
    ).hexdigest()


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    catalog = _json(root / protocol["source"]["catalog"])["probes"]
    attempts = _jsonl(root / protocol["source"]["journal"])
    grouped = _eligible(catalog, attempts)
    minimum = int(protocol["selection"]["records_per_capability_builder"])
    counts = {f"{capability}:{builder}": len(grouped[(capability, builder)]) for capability in WEAK_CAPABILITIES for builder in range(4)}
    if min(counts.values()) < minimum:
        raise Phase3Error(f"host supervision depth unavailable: {counts}")
    return {
        "status": "PASS_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "eligible_by_capability_builder": counts,
        "minimum_required": minimum,
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
    }


def build(root: Path, protocol_path: Path, artifact: Path, result_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if artifact.exists() or result_path.exists():
        raise Phase3Error("immutable host supervision output exists")
    catalog = _json(root / protocol["source"]["catalog"])["probes"]
    attempts = _jsonl(root / protocol["source"]["journal"])
    grouped = _eligible(catalog, attempts)
    depth = int(protocol["selection"]["records_per_capability_builder"])
    selected: list[dict[str, Any]] = []
    eligible_counts = {}
    for capability in WEAK_CAPABILITIES:
        for builder in range(4):
            values = sorted(grouped[(capability, builder)], key=lambda row: _selection_key(row, protocol_sha))
            eligible_counts[f"{capability}:{builder}"] = len(values)
            if len(values) < depth:
                raise Phase3Error("preregistered host supervision depth unavailable")
            selected.extend(values[:depth])
    selected.sort(key=lambda row: str(row["record_id"]))
    if len({row["record_id"] for row in selected}) != len(selected):
        raise Phase3Error("host supervision record identity collision")
    records_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in selected)
    counts = Counter((row["capability"], row["builder"]) for row in selected)
    expected = {(capability, builder): depth for capability in WEAK_CAPABILITIES for builder in range(4)}
    if counts != expected:
        raise Phase3Error("host supervision balance changed")
    accounting = {
        "format": "abi-host-conformant-weak-supervision-accounting/1",
        "records": len(selected),
        "unique_prompt_bytes": sum(len(value.encode()) for value in {str(row["host_prompt"]) for row in selected}),
        "teacher_output_bytes": sum(len(str(row["output"]).encode()) for row in selected),
        "teacher_input_tokens": sum(int(row["source_teacher_input_tokens"]) for row in selected),
        "teacher_output_tokens": sum(int(row["source_teacher_output_tokens"]) for row in selected),
        "surface_normalized_records": sum(bool(row["surface_normalization_steps"]) for row in selected),
        "repair_attempt_records": sum(row["source_attempt_kind"] != "initial" for row in selected),
        "stored_logits": 0,
        "stored_hidden_activations": 0,
        "copied_source_parameters": 0,
        "teacher_model_loaded_for_this_build": False,
    }
    accounting_bytes = json.dumps(accounting, indent=2, sort_keys=True).encode() + b"\n"
    manifest = {
        "format": "abi-host-conformant-weak-supervision-manifest/1",
        "protocol_sha256": protocol_sha,
        "record_format": RECORD_FORMAT,
        "records": len(selected),
        "records_per_capability_builder": depth,
        "records_jsonl_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "accounting_sha256": hashlib.sha256(accounting_bytes).hexdigest(),
        "source_catalog_sha256": sha256_file(root / protocol["source"]["catalog"]),
        "source_journal_sha256": sha256_file(root / protocol["source"]["journal"]),
        "final_test_accessed": False,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in (("accounting.json", accounting_bytes), ("manifest.json", manifest_bytes), ("records.jsonl", records_bytes)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    result = {
        "format": "abi-capability-compiler-phase3-host-supervision-build-result/1",
        "status": "PASS_BUILD_HOSTILE_VERIFICATION_REQUIRED",
        "protocol_sha256": protocol_sha,
        "artifact": {"path": str(artifact.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(artifact), "bytes": artifact.stat().st_size},
        "records": len(selected),
        "records_per_capability_builder": depth,
        "eligible_by_capability_builder": eligible_counts,
        "accounting": accounting,
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "historical_artifact_mutated": False,
        "final_test_accessed": False,
        "phase3_certified": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(result_path, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_HOST_SUPERVISION_PROTOCOL_V479.json")
    parser.add_argument("--artifact", default="results/abi_capability_compiler_phase3_host_supervision/build_v480/host_weak_supervision_v480.abicir")
    parser.add_argument("--result", default="results/abi_capability_compiler_phase3_host_supervision/build_v480/result.json")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    if args.preflight:
        result = preflight(root, root / args.protocol)
    else:
        result = build(root, root / args.protocol, root / args.artifact, root / args.result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
