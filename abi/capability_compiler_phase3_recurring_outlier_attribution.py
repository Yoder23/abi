"""Read-only attribution of the recurring layer-zero validation outlier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-recurring-outlier-attribution/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_RECURRING_OUTLIER_ATTRIBUTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("recurring-outlier attribution governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"recurring-outlier attribution binding changed: {name}")
    if output.exists():
        raise Phase3Error("recurring-outlier attribution output exists")
    output.mkdir(parents=True)
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = root / protocol["artifact_directory"]
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str((root / protocol["layercake_host"]).resolve()))
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    examples = sequential.field._examples(root, base, tokenizer)
    by_id = {str(row["record_id"]): row for row in examples}
    calibration = base["calibration"]
    train, validation, _ = dual._calibration_examples(
        examples, seed=int(base["training"]["seed"]),
        train_per_capability=int(calibration["train_records_per_capability"]),
        validation_per_capability=int(calibration["validation_records_per_capability"]),
        maximum_tokens=int(calibration["maximum_sequence_tokens"]),
    )
    maximum = int(calibration["maximum_sequence_tokens"])

    def describe(selected: dict) -> dict:
        original = by_id[str(selected["record_id"])]
        source = list(original["source_ids"])
        target = list(original["target_actions"][:-1])
        full = source + target
        packed = list(selected["input_ids"])
        mapped = [trajectory.source_token_id(value, int(base["source"]["terminal_token_id"])) for value in packed]
        return {
            "record_id": selected["record_id"], "capability": selected["capability"],
            "source_actions": len(source), "target_context_actions": len(target),
            "full_context_actions": len(full), "packed_actions": len(packed),
            "truncated_actions": len(full) - len(packed),
            "source_truncated": len(source) > maximum,
            "target_context_retained": max(0, min(len(target), maximum - len(source))),
            "unique_packed_actions": len(set(packed)),
            "source_host_mapping_valid": len(mapped) == len(packed),
            "packed_action_set": set(packed),
        }

    train_rows = [describe(row) for row in train]
    validation_rows = [describe(row) for row in validation]
    train_by_capability = {}
    for row in train_rows:
        train_by_capability.setdefault(row["capability"], []).append(row)
    all_train_actions = set().union(*(row["packed_action_set"] for row in train_rows))
    metrics = []
    for row in validation_rows:
        peers = train_by_capability[row["capability"]]
        peer_union = set().union(*(peer["packed_action_set"] for peer in peers))
        unseen_capability = row["packed_action_set"] - peer_union
        unseen_global = row["packed_action_set"] - all_train_actions
        jaccards = []
        for peer in peers:
            union = row["packed_action_set"] | peer["packed_action_set"]
            jaccards.append(len(row["packed_action_set"] & peer["packed_action_set"]) / max(1, len(union)))
        metrics.append({
            key: value for key, value in row.items() if key != "packed_action_set"
        } | {
            "capability_train_max_full_context_actions": max(peer["full_context_actions"] for peer in peers),
            "capability_train_max_packed_actions": max(peer["packed_actions"] for peer in peers),
            "outside_capability_train_full_length_support": row["full_context_actions"] > max(peer["full_context_actions"] for peer in peers),
            "unseen_action_fraction_within_capability": len(unseen_capability) / max(1, len(row["packed_action_set"])),
            "unseen_action_fraction_global": len(unseen_global) / max(1, len(row["packed_action_set"])),
            "maximum_train_record_action_jaccard_within_capability": max(jaccards),
        })
    worst_id = str(protocol["recurring_outlier_record_id"])
    worst = next(row for row in metrics if row["record_id"] == worst_id)
    result = {
        "format": FORMAT,
        "status": "PASS_RECURRING_OUTLIER_ATTRIBUTED_TO_CONTEXT_COVERAGE" if worst["outside_capability_train_full_length_support"] or worst["source_truncated"] else "INCONCLUSIVE_RECURRING_OUTLIER_ATTRIBUTION",
        "protocol_sha256": sha256_file(protocol_path),
        "train_records": len(train_rows), "validation_records": len(validation_rows),
        "maximum_sequence_tokens": maximum,
        "train_truncated_records": sum(row["truncated_actions"] > 0 for row in train_rows),
        "validation_truncated_records": sum(row["truncated_actions"] > 0 for row in validation_rows),
        "all_source_host_mappings_valid": all(row["source_host_mapping_valid"] for row in metrics),
        "recurring_outlier": worst,
        "validation_record_metrics": metrics,
        "training_performed": False, "artifact_written": False,
        "final_test_accessed": False, "phase3_certified": False,
        "claim_boundary": "Read-only development-split context-coverage attribution only; no architecture, extracted model, autonomous quality, Phase 3, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_RECURRING_OUTLIER_ATTRIBUTION_PROTOCOL_V366.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/outlier_attribution_v367")
    args = parser.parse_args(); root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
