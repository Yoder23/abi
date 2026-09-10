"""R16 strict verifier repaired to open live files and rehash the source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.preexisting_representation_r15b.public_qualification import (
    canonical_json_bytes,
    sha256_bytes,
)
from experiments.preexisting_representation_r15b.verify_live import (
    _source_snapshot_inventory,
)

from .facts import PUBLIC_FACTS
from .verify_strict_v2 import verify_v2


def _verified(path: Path, name: str) -> dict[str, Any]:
    value = json_object(path)
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if value.get("evidence_sha256") != evidence_hash(scientific):
        raise R14Error(f"R16 {name} evidence hash changed")
    return value


def _validate_extraction_tree(path: Path) -> int:
    manifest = _verified(path / "manifest.json", "physical manifest")
    result = _verified(path / "result.json", "physical extraction result")
    launcher = _verified(path / "launcher.json", "physical launcher")
    mountinfo = path / "mountinfo.txt"
    if (
        not mountinfo.is_file()
        or launcher.get("worker_exit_code") != 0
        or launcher.get("sandbox_policy") != "linux-pivot-root-no-network/1"
        or launcher.get("manifest_evidence_sha256") != manifest["evidence_sha256"]
        or launcher.get("result_sha256") != sha256_file(path / "result.json")
        or launcher.get("mountinfo_sha256") != sha256_file(mountinfo)
        or result.get("capsule_manifest_evidence_sha256") != manifest["evidence_sha256"]
        or result.get("mountinfo_sha256") != sha256_file(mountinfo)
        or result.get("old_root_present") is not False
        or result.get("windows_mount_present") is not False
        or result.get("network_namespace_isolated") is not True
        or result.get("oracle_fields_consumed") != 0
        or result.get("fact_ids_consumed") != 0
        or result.get("success_ids_consumed") != 0
        or result.get("reveal_files_present") != 0
    ):
        raise R14Error("R16 physical extraction tree changed")
    packages = result.get("packages")
    if not isinstance(packages, list) or len(packages) != 2:
        raise R14Error("R16 physical package inventory changed")
    for item in packages:
        package = path / str(item["path"])
        if (
            not package.is_file()
            or package.stat().st_size != int(item["bytes"])
            or sha256_file(package) != item["sha256"]
        ):
            raise R14Error("R16 physical package changed")
    return len(packages)


def _validate_public_requalification(path: Path, root: Path) -> dict[str, Any]:
    receipt = _verified(path / "receipt.json", "public requalification receipt")
    protocol = root / "experiments/factual_semantic_r16/PUBLIC_PROTOCOL.md"
    if (
        receipt.get("verdict") != "PASS"
        or receipt.get("claim_ceiling") != "PUBLIC_SOURCE_PREREQUISITE_ONLY"
        or receipt.get("protocol_sha256") != sha256_file(protocol)
        or receipt.get("fact_registry_sha256")
        != sha256_bytes(canonical_json_bytes([fact.__dict__ for fact in PUBLIC_FACTS]))
        or receipt.get("metrics")
        != {
            "candidate_scored_exact": 48,
            "candidate_scored_total": 48,
            "facts_extracted_exact": 16,
            "facts_total": 16,
            "open_exact": 48,
            "open_total": 48,
            "semantic_exact": 48,
            "semantic_total": 48,
        }
    ):
        raise R14Error("R16 public requalification changed")
    source = receipt.get("source", {})
    if (
        source.get("model_id") != "Qwen/Qwen2-7B-Instruct"
        or source.get("revision") != "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
        or source.get("trainable_parameters") != 0
        or source.get("training_steps") != 0
    ):
        raise R14Error("R16 public requalification source changed")
    for item in receipt.get("artifacts", {}).values():
        artifact = path / str(item["path"])
        if not artifact.is_file() or sha256_file(artifact) != item["sha256"]:
            raise R14Error("R16 public requalification artifact changed")
    old = _verified(
        root / "results/factual_semantic_r16/public_v2_revision_002/receipt.json",
        "historical public receipt",
    )
    if (
        receipt["artifacts"] != old["artifacts"]
        or receipt["metrics"] != old["metrics"]
        or receipt["fact_registry_sha256"] != old["fact_registry_sha256"]
    ):
        raise R14Error("R16 public requalification does not reproduce historical evidence")
    return receipt


def verify_v3(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    live_path: Path,
    public_requalification: Path,
    source_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = verify_v2(config_path, reveal_path, run_dir, live_path)
    root = config_path.resolve().parents[3]
    config = json_object(config_path)
    run = _verified(run_dir / "receipt.json", "run receipt")
    live = _verified(live_path, "live receipt")
    live_dir = live_path.parent
    live_files = live.get("files_replayed_byte_exact")
    if not isinstance(live_files, list) or len(live_files) != 7:
        raise R14Error("R16 live file inventory changed")
    for item in live_files:
        path = live_dir / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise R14Error("R16 actual live replay file changed")
    physical_packages = sum(
        _validate_extraction_tree(path)
        for path in (
            run_dir / "extraction",
            run_dir / "control_extraction",
            live_dir / "extraction",
            live_dir / "control_extraction",
        )
    )
    stored_snapshot = _verified(
        root / "results/preexisting_representation_r15b/heldout_v1_live_v2/receipt.json",
        "source snapshot receipt",
    )["source_snapshot"]
    actual_snapshot = source_inventory or _source_snapshot_inventory(
        str(config["source"]["model_id"]), str(config["source"]["revision"])
    )
    if actual_snapshot != stored_snapshot:
        raise R14Error("R16 direct source snapshot rehash changed")
    public = _validate_public_requalification(public_requalification, root)
    historical_public = _verified(
        root / "results/factual_semantic_r16/public_v2_revision_002/receipt.json",
        "historical public receipt",
    )
    historical_protocol_matches_current = (
        historical_public["protocol_sha256"] == public["protocol_sha256"]
    )
    if historical_protocol_matches_current:
        raise R14Error("R16 public protocol repair no longer describes a mismatch")
    source_rows = [
        json.loads(line)
        for line in (run_dir / run["artifacts"]["source_rows"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    extraction_rows = [row for row in source_rows if row["split"] == "extraction"]
    candidate_answer_rows = sum(
        list(map(str, row["candidate_values"])).count(str(row["answer"])) == 1
        for row in extraction_rows
    )
    bundle = _verified(run_dir / "source_bundle.json", "compiler source bundle")
    forbidden = {"answer", "oracle", "namespace", "fact_id", "secret", "reveal", "success_id"}
    explicit_labeled_answer_fields = sum(
        bool(forbidden.intersection(record)) for record in bundle["records"]
    )
    if candidate_answer_rows != 48 or explicit_labeled_answer_fields != 0:
        raise R14Error("R16 closed-candidate disclosure changed")
    result = {
        **{key: value for key, value in base.items() if key != "evidence_sha256"},
        "format": "abi-r16-strict-verification/3",
        "actual_live_files_rehashed": len(live_files),
        "physical_extraction_trees_verified": 4,
        "physical_package_files_verified": physical_packages,
        "source_snapshot_files_rehashed": actual_snapshot["file_count"],
        "source_snapshot_bytes_rehashed": actual_snapshot["total_bytes"],
        "public_requalification_artifacts_verified": len(public["artifacts"]),
        "public_requalification_protocol_sha256": public["protocol_sha256"],
        "historical_public_protocol_declared_sha256": historical_public[
            "protocol_sha256"
        ],
        "historical_public_protocol_matches_current_file": (
            historical_protocol_matches_current
        ),
        "correct_answer_present_once_in_candidate_rows": candidate_answer_rows,
        "explicit_labeled_answer_fields_in_compiler_bundle": explicit_labeled_answer_fields,
        "candidate_boundary": (
            "The correct answer string is present once in each registered 12-candidate set; "
            "no field labels which candidate is the oracle answer."
        ),
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--public-requalification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_v3(
        args.config,
        args.reveal,
        args.run_dir,
        args.live,
        args.public_requalification,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
