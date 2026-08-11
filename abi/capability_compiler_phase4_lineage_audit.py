"""Fail-closed imported-information lineage audit for the Phase 4 frontier."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-lineage-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_FULL_LINEAGE_AUDIT"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("Phase 4 lineage governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 4 lineage binding changed: {relative}")
    return protocol, sha256_file(path)


def _archive(root: Path, item: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = (root / str(item["path"])).resolve()
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("records.jsonl")
        physical = raw.splitlines()
        rows = [json.loads(line) for line in physical if line.strip()]
        accounting = json.loads(archive.read("accounting.json"))
    if len(rows) != int(item["records"]):
        raise Phase3Error(f"record count changed: {item['id']}")
    attempts = {str(row["source_attempt_sha256"]) for row in rows}
    probes = {str(row["probe_id"]) for row in rows}
    prompt_field = "normalized_acquisition_prompt" if "normalized_acquisition_prompt" in rows[0] else "host_prompt"
    output_field = "normalized_output" if "normalized_output" in rows[0] else "output"
    token_field = "authoritative_teacher_tokens" if "authoritative_teacher_tokens" in rows[0] else "source_teacher_output_tokens"
    result = {
        "id": str(item["id"]),
        "path": str(item["path"]),
        "sha256": sha256_file(path),
        "disk_bytes": path.stat().st_size,
        "records": len(rows),
        "physical_jsonl_lines": len(physical),
        "blank_jsonl_lines": len(physical) - len(rows),
        "unique_source_attempts": len(attempts),
        "unique_probe_ids": len(probes),
        "prompt_utf8_bytes": sum(len(str(row[prompt_field]).encode("utf-8")) for row in rows),
        "output_utf8_bytes": sum(len(str(row[output_field]).encode("utf-8")) for row in rows),
        "authoritative_teacher_output_tokens": sum(int(row[token_field]) for row in rows),
        "teacher_input_tokens": sum(int(row.get("teacher_input_tokens", row.get("source_teacher_input_tokens", 0))) for row in rows),
        "stored_logits": int(accounting.get("stored_logits", 0)),
        "stored_hidden_activations": int(accounting.get("stored_hidden_activations", accounting.get("stored_activations", 0))),
        "copied_source_parameters": int(accounting.get("copied_source_parameters", 0)),
        "capability_counts": dict(sorted(Counter(str(row["capability"]) for row in rows).items())),
    }
    return result, rows


def _overlap(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, int]:
    def values(rows: list[dict[str, Any]], key: str) -> set[str]:
        return {str(row[key]) for row in rows if key in row}
    return {
        "source_attempt_sha256": len(values(a, "source_attempt_sha256") & values(b, "source_attempt_sha256")),
        "probe_id": len(values(a, "probe_id") & values(b, "probe_id")),
        "normalized_output_sha256": len(values(a, "normalized_output_sha256") & values(b, "normalized_output_sha256")),
    }


def _stage(root: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    metadata_path = (root / str(item["metadata"])).resolve()
    metadata = _json(metadata_path)
    checkpoint_path = metadata_path.parent / str(metadata["checkpoint"]["path"])
    actual = sha256_file(checkpoint_path)
    failures = []
    if actual != str(item["checkpoint_sha256"]) or actual != str(metadata["checkpoint"]["sha256"]):
        failures.append("checkpoint_hash_mismatch")
    if metadata.get("final_test_accessed") is not False:
        failures.append("final_test_access_not_false")
    observed_parent = metadata.get("parent_checkpoint_sha256")
    if observed_parent is None:
        observed_parent = metadata.get("parent", {}).get("checkpoint_sha256")
    if item.get("parent_checkpoint_sha256") and observed_parent != item["parent_checkpoint_sha256"]:
        failures.append("parent_checkpoint_mismatch")
    observed_init = metadata.get("initialization", {}).get("checkpoint_sha256", metadata.get("initialization", {}).get("sha256"))
    if item.get("initialization_checkpoint_sha256") and observed_init != item["initialization_checkpoint_sha256"]:
        failures.append("initialization_checkpoint_mismatch")
    training = metadata.get("training", {})
    supervision = metadata.get("supervision", {})
    teacher_tokens = {
        key: int(value)
        for key, value in {
            "teacher_response_tokens_seen": training.get("teacher_response_tokens_seen"),
            "targeted_teacher_tokens_seen": training.get("targeted_teacher_tokens_seen"),
            "anchor_teacher_tokens_seen": training.get("anchor_teacher_tokens_seen"),
            "teacher_response_tokens_in_loss": training.get("teacher_response_tokens_in_loss"),
            "supervision_teacher_tokens_seen": supervision.get("teacher_tokens_seen"),
        }.items()
        if value is not None
    }
    return {
        "id": str(item["id"]),
        "metadata": str(item["metadata"]),
        "metadata_sha256": sha256_file(metadata_path),
        "checkpoint_sha256": actual,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parent_checkpoint_sha256": observed_parent,
        "initialization_checkpoint_sha256": observed_init,
        "data_artifacts": list(item.get("data_artifacts", [])),
        "seed": training.get("seed", metadata.get("seed")),
        "steps": training.get("steps"),
        "observations": training.get("observations", training.get("balanced_record_observations")),
        "teacher_token_exposures": teacher_tokens,
        "teacher_present_at_inference": metadata.get("teacher_present_at_inference", metadata.get("source", {}).get("teacher_present_at_inference")),
        "final_test_accessed": metadata.get("final_test_accessed"),
        "failures": failures,
    }


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    archives: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for item in protocol["teacher_artifacts"]:
        stats, values = _archive(root, item)
        archives[str(item["id"])] = stats
        rows[str(item["id"])] = values

    ids = list(rows)
    overlaps = {
        f"{ids[left]}__{ids[right]}": _overlap(rows[ids[left]], rows[ids[right]])
        for left in range(len(ids))
        for right in range(left + 1, len(ids))
    }
    all_attempts: dict[str, dict[str, Any]] = {}
    for artifact_id, values in rows.items():
        for row in values:
            attempt = str(row["source_attempt_sha256"])
            all_attempts.setdefault(attempt, {"artifacts": [], "teacher_output_tokens": int(row.get("authoritative_teacher_tokens", row.get("source_teacher_output_tokens", 0)))})
            all_attempts[attempt]["artifacts"].append(artifact_id)
    unique_tokens = sum(int(value["teacher_output_tokens"]) for value in all_attempts.values())

    stages = [_stage(root, item) for item in protocol["checkpoint_stages"]]
    failures = [f"{stage['id']}:{failure}" for stage in stages for failure in stage["failures"]]
    declared = {str(item["id"]) for item in protocol["teacher_artifacts"]}
    for stage in stages:
        unknown = set(stage["data_artifacts"]) - declared
        if unknown:
            failures.append(f"{stage['id']}:unknown_data_artifacts:{','.join(sorted(unknown))}")

    host_path = (root / protocol["fixed_host"]["metadata"]).resolve()
    host = _json(host_path)
    host_checkpoint = (root / protocol["fixed_host"]["checkpoint"]).resolve()
    if sha256_file(host_checkpoint) != protocol["fixed_host"]["checkpoint_sha256"]:
        failures.append("fixed_host:checkpoint_hash_mismatch")
    if host.get("test_accessed") is not False:
        failures.append("fixed_host:test_access_not_false")

    router_path = (root / protocol["router"]["metadata"]).resolve()
    router = _json(router_path)
    router_checkpoint = router_path.parent / router["checkpoint"]["path"]
    if sha256_file(router_checkpoint) != protocol["router"]["checkpoint_sha256"]:
        failures.append("router:checkpoint_hash_mismatch")

    naive_final_only_confounded = any(
        stage["id"] == "v484_host_recovery_bridge" and "v480_host_supervision" in stage["data_artifacts"]
        for stage in stages
    ) and any(stage["id"] == "v526_route_isolated_bridge" and stage["initialization_checkpoint_sha256"] for stage in stages)
    if not naive_final_only_confounded:
        failures.append("expected_hidden_warm_start_confound_not_detected")

    status = "PASS_COMPLETE_LINEAGE_AUDIT_FRONTIER_PROTOCOL_REQUIRED" if not failures else "FAIL_CLOSED_LINEAGE_INCOMPLETE"
    result = {
        "format": "abi-capability-compiler-phase4-lineage-audit-result/1",
        "status": status,
        "protocol_sha256": protocol_sha,
        "teacher_artifacts": archives,
        "artifact_overlap": overlaps,
        "unique_imported_information": {
            "source_attempts": len(all_attempts),
            "authoritative_teacher_output_tokens": unique_tokens,
            "stored_logits": sum(value["stored_logits"] for value in archives.values()),
            "stored_hidden_activations": sum(value["stored_hidden_activations"] for value in archives.values()),
            "copied_source_parameters": sum(value["copied_source_parameters"] for value in archives.values()),
        },
        "checkpoint_stages": stages,
        "fixed_host_prior": {
            "metadata": protocol["fixed_host"]["metadata"],
            "metadata_sha256": sha256_file(host_path),
            "checkpoint_sha256": sha256_file(host_checkpoint),
            "checkpoint_bytes": host_checkpoint.stat().st_size,
            "parameters_total": int(host["parameters"]["total"]),
            "teacher_initialized_source_blocks": list(host["initialization"]["retained_source_blocks"]),
            "raw_utf8_bytes_exposed": int(host["training"]["raw_utf8_bytes_exposed"]),
            "response_tokens_seen": int(host["training"]["response_tokens_seen"]),
            "role": "fixed pre-existing LayerCake host prior; separately accounted, never called zero-information",
        },
        "router": {
            "metadata": protocol["router"]["metadata"],
            "checkpoint_sha256": sha256_file(router_checkpoint),
            "checkpoint_bytes": router_checkpoint.stat().st_size,
            "parameters": int(router["config"]["trainable_parameters"]),
            "labeled_records": int(router["imported_information"]["records"]),
            "teacher_outputs_added": int(router["imported_information"]["teacher_outputs_added"]),
        },
        "deployed_composition": protocol["deployed_composition"],
        "confounds": {
            "varying_only_v526_v480_subset_is_invalid": naive_final_only_confounded,
            "reason": "V526 initializes from V484, which already consumed the complete V480 artifact; V463 also inherits the complete Phase 1 IR through V443, V459, and V463.",
            "minimum_valid_clean_start": "the immutable external LayerCake host before V443 plus a freshly initialized router and bridges",
            "required_frontier_scope": "retrain every ABI data-dependent stage from the common fixed host for each nested information budget",
        },
        "gates": {
            "all_bound_hashes_match": not failures,
            "teacher_attempt_overlap_recomputed": True,
            "fixed_host_prior_separately_accounted": True,
            "router_information_separately_accounted": True,
            "warm_start_hidden_information_detected": naive_final_only_confounded,
            "teacher_model_loaded": False,
            "neural_training_performed": False,
            "final_test_accessed": False,
        },
        "failures": failures,
        "phase2_status_unchanged": "BLOCKED_EXTERNAL_HUMAN_RATINGS_0_OF_21000_FILLED",
        "phase3_status_unchanged": "MACHINE_EVIDENCE_COMPLETE_NOT_UNCONDITIONALLY_CERTIFIED",
        "phase4_certified": False,
        "phase5_open": False,
        "decision": "Seal a nested-budget protocol that rebuilds the complete ABI data-dependent lineage from the same fixed host. Do not run the invalid final-bridge-only subset experiment.",
        "claim_boundary": "Read-only lineage closure only; no training, minimum-information, final-test, Phase 4 certificate, or ABI superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error(f"immutable audit output exists: {output}")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
