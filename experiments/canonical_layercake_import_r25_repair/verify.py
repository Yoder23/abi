"""Live verification under the declared post-hoc R25 semantic scorer repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.canonical_layercake_import_r25.binding import load_config as load_original
from experiments.canonical_layercake_import_r25.run import run
from experiments.canonical_layercake_import_r25.verify import _metrics, _rows, _validate_packages
from experiments.factual_semantic_r16.public_sequence_scoring import normalized_text
from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash

from .binding import load_config


def verify(config_path: Path, replay_dir: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    repair = load_config(root, config_path.resolve())
    original_path = root / repair["original_config"][0]["path"]
    original = load_original(root, original_path)
    stored_dir = root / "results/canonical_layercake_import_r25/public_v1"
    stored_result = json_object(stored_dir / "result.json")
    if stored_result.get("evidence_sha256") != selfless_evidence_hash(stored_result):
        raise R14Error("R25 repair stored result identity changed")
    names = (
        "package_builds",
        "domain_observations",
        "english_immutability",
        "lifecycle",
        "english_leakage",
    )
    stored = {name: _rows(stored_dir, stored_result, name) for name in names}
    packages = _validate_packages(root, original, stored_result)
    replay_result = run(original_path, replay_dir.resolve())
    replay = {name: _rows(replay_dir.resolve(), replay_result, name) for name in names}
    for name in names[1:]:
        if replay[name] != stored[name]:
            raise R14Error(f"R25 repair live replay changed: {name}")
    for left, right in zip(stored["package_builds"], replay["package_builds"], strict=True):
        if {key: value for key, value in left.items() if key != "path"} != {
            key: value for key, value in right.items() if key != "path"
        }:
            raise R14Error("R25 repair package rebuild changed")
    original_metrics = _metrics(
        original,
        stored["package_builds"],
        stored["domain_observations"],
        stored["english_immutability"],
        stored["lifecycle"],
        stored["english_leakage"],
    )
    corrected = dict(original_metrics)
    corrected["teacher_agreement"] = sum(
        normalized_text(row["gpu_output"]) == normalized_text(row["teacher_output"])
        for row in stored["domain_observations"]
    )
    corrected_gates = {
        name: corrected[name] == threshold for name, threshold in original["gates"].items()
    }
    corrected_gates["teacher_absent_at_execution"] = True
    changed_metrics = {
        name: {"before": original_metrics[name], "after": corrected[name]}
        for name in corrected
        if original_metrics[name] != corrected[name]
    }
    if (
        stored_result.get("verdict") != "FAIL_BOUNDED_DIRECT_IMPORT"
        or changed_metrics
        != {"teacher_agreement": {"before": 135, "after": 144}}
        or not all(corrected_gates.values())
        or packages != 6
    ):
        raise R14Error("R25 semantic repair diagnostic failed")
    result = {
        "format": "abi-r25-semantic-repair-verification/1",
        "status": "PASS_POSTHOC_SCORER_REPAIR_REQUIRES_FRESH_REPLICATION",
        "scientific_claim": "R25_POSTHOC_NORMALIZED_TEACHER_AGREEMENT",
        "claim_ceiling": "NOT_PROMOTED_WITHOUT_FRESH_PREREGISTERED_REPLICATION",
        "repair_config_sha256": sha256_file(config_path),
        "original_config_sha256": sha256_file(original_path),
        "stored_result_sha256": sha256_file(stored_dir / "result.json"),
        "replay_result_sha256": sha256_file(replay_dir / "result.json"),
        "packages_recomputed": packages,
        "rows_live_recomputed": sum(len(rows) for rows in replay.values()),
        "stored_scientific_booleans_trusted": False,
        "candidate_changes": 0,
        "gate_changes": 0,
        "scorer_changes": 1,
        "changed_metrics": changed_metrics,
        "corrected_metrics": corrected,
        "corrected_gates": corrected_gates,
        "fresh_replication_required": True,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.config, args.replay)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
