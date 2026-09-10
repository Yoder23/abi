"""Additive fail-closed verification of every declared R16 evidence artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.preexisting_representation_r15b.public_qualification import sha256_bytes

from .verify import _verified_object, verify


def _launcher(path: Path, expected_sha: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise R14Error("R16 isolation launcher file changed")
    return _verified_object(path, "isolation launcher")


def _physical_tree(run_dir: Path, name: str, receipt: dict[str, Any]) -> None:
    extraction = run_dir / name
    result = _verified_object(extraction / "result.json", f"{name} result")
    launcher_key = "launcher_sha256" if name == "extraction" else None
    if launcher_key is not None:
        launcher = _launcher(extraction / "launcher.json", receipt["isolation"][launcher_key])
    else:
        launcher = _verified_object(extraction / "launcher.json", f"{name} launcher")
    manifest = _verified_object(extraction / "manifest.json", f"{name} manifest")
    if (
        result.get("old_root_present") is not False
        or result.get("windows_mount_present") is not False
        or result.get("network_namespace_isolated") is not True
        or launcher.get("worker_exit_code") != 0
        or launcher.get("sandbox_policy") != "linux-pivot-root-no-network/1"
        or manifest.get("reveal_files_included") != 0
        or manifest.get("oracle_fields_included") != 0
        or manifest.get("fact_ids_included") != 0
        or manifest.get("success_ids_included") != 0
        or manifest.get("network") is not False
    ):
        raise R14Error("R16 physical isolation claim changed")
    inventory = {item["path"]: item for item in manifest.get("files", [])}
    if set(inventory) != {"isolated_worker.py", "source_bundle.json", "spec.json"}:
        raise R14Error("R16 physical capsule inventory changed")
    for relative, item in inventory.items():
        if relative == "source_bundle.json":
            source = run_dir / ("source_bundle.json" if name == "extraction" else "rotated_score_bundle.json")
        else:
            source = Path(__file__).resolve().parent / relative
            if relative == "spec.json":
                continue
        if sha256_file(source) != item["sha256"] or source.stat().st_size != item["bytes"]:
            raise R14Error("R16 capsule source identity changed")
    if result.get("capsule_manifest_evidence_sha256") != manifest["evidence_sha256"]:
        raise R14Error("R16 capsule/result binding changed")
    if launcher.get("manifest_evidence_sha256") != manifest["evidence_sha256"]:
        raise R14Error("R16 capsule/launcher binding changed")
    if launcher.get("result_sha256") != sha256_file(extraction / "result.json"):
        raise R14Error("R16 result/launcher binding changed")
    for item in result.get("packages", []):
        package = extraction / item["path"]
        if (
            not package.is_file()
            or sha256_file(package) != item["sha256"]
            or package.stat().st_size != item["bytes"]
        ):
            raise R14Error("R16 isolated package/result binding changed")


def verify_v2(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    live_path: Path,
) -> dict[str, Any]:
    base = verify(config_path, reveal_path, run_dir)
    config = json_object(config_path)
    receipt = _verified_object(run_dir / "receipt.json", "run receipt")
    source_rows = [
        json.loads(line)
        for line in (run_dir / receipt["artifacts"]["source_rows"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    extraction_rows = [row for row in source_rows if row["split"] == "extraction"]
    residual_ref = receipt["artifacts"]["source_residuals"]
    residual_path = run_dir / residual_ref["path"]
    if sha256_file(residual_path) != residual_ref["sha256"]:
        raise R14Error("R16 source residual artifact changed")
    tensors = load_file(str(residual_path), device="cpu")
    if set(tensors) != {"residuals"} or list(tensors["residuals"].shape) != [48, 3584]:
        raise R14Error("R16 source residual tensor contract changed")
    if not all(math.isfinite(float(value)) for value in tensors["residuals"].reshape(-1)):
        raise R14Error("R16 source residual tensor is non-finite")
    for row, residual in zip(extraction_rows, tensors["residuals"]):
        if row["residual_sha256"] != sha256_bytes(residual.numpy().tobytes()):
            raise R14Error("R16 source row/residual binding changed")

    bundle = _verified_object(run_dir / "source_bundle.json", "source bundle")
    raw_by_key = {
        (row["entity"], row["question"], int(row["view"])): row for row in extraction_rows
    }
    for record in bundle["records"]:
        row = raw_by_key.get((record["subject"], record["question"], int(record["view"])))
        if row is None or record["candidates"] != row["candidate_values"] or record["scores"] != row["candidate_scores"]:
            raise R14Error("R16 anonymous bundle/source-row binding changed")
    rotated_ref = run_dir / "rotated_score_bundle.json"
    rotated = _verified_object(rotated_ref, "rotated-score bundle")
    if len(rotated["records"]) != len(bundle["records"]):
        raise R14Error("R16 rotated-score row count changed")
    for original, control in zip(bundle["records"], rotated["records"]):
        if (
            {key: control[key] for key in control if key != "scores"}
            != {key: original[key] for key in original if key != "scores"}
            or control["scores"] != original["scores"][1:] + original["scores"][:1]
        ):
            raise R14Error("R16 rotated-score intervention changed")

    _physical_tree(run_dir, "extraction", receipt)
    _physical_tree(run_dir, "control_extraction", receipt)
    if sha256_file(run_dir / "control_extraction/result.json") != receipt["isolation"]["control_result_sha256"]:
        raise R14Error("R16 control extraction identity changed")

    info = receipt["information_accounting"]
    packages = list((run_dir / "extraction/packages").glob("*.abipkg"))
    if (
        info.get("raw_source_prompts") != len(source_rows)
        or info.get("candidate_scores") != sum(
            len(row["candidate_scores"]) for row in extraction_rows
        )
        or info.get("generated_tokens", 0) <= 0
        or info.get("output_bytes")
        != sum(len(row["completion"].encode()) for row in source_rows)
        or info.get("source_bundle_bytes") != (run_dir / "source_bundle.json").stat().st_size
        or info.get("source_residual_bytes") != residual_path.stat().st_size
        or info.get("final_package_bytes") != sum(path.stat().st_size for path in packages)
        or info.get("final_package_facts") != 16
        or info.get("source_parameters_in_final_packages") != 0
        or info.get("bridge_parameters_trained") != 0
        or info.get("source_training_steps") != 0
    ):
        raise R14Error("R16 information accounting changed")
    source = receipt["source"]
    if (
        source.get("model_id") != config["source"]["model_id"]
        or source.get("revision") != config["source"]["revision"]
        or source.get("training_steps") != 0
        or source.get("present_at_package_execution") is not False
    ):
        raise R14Error("R16 source execution boundary changed")
    r15b_live = _verified_object(
        config_path.resolve().parents[3]
        / "results/preexisting_representation_r15b/heldout_v1_live_v2/receipt.json",
        "pinned source snapshot receipt",
    )
    if r15b_live["source_snapshot"]["evidence_sha256"] != config["source"]["snapshot_evidence_sha256"]:
        raise R14Error("R16 pinned source snapshot identity changed")

    live = _verified_object(live_path, "live verification")
    expected_live = {
        "source_observations.jsonl": receipt["artifacts"]["source_rows"]["sha256"],
        "source_prompt_end_residuals.safetensors": residual_ref["sha256"],
        "source_bundle.json": receipt["artifacts"]["source_bundle"]["sha256"],
        "rotated_score_bundle.json": sha256_file(rotated_ref),
        "evaluation.jsonl": receipt["artifacts"]["evaluation_rows"]["sha256"],
    }
    for item in receipt["packages"]:
        expected_live[(Path("extraction") / item["path"]).as_posix()] = item["sha256"]
    observed_live = {item["path"].replace("\\", "/"): item["sha256"] for item in live["files_replayed_byte_exact"]}
    if (
        live.get("verdict") != "PASS"
        or any(observed_live.get(path) != digest for path, digest in expected_live.items())
        or live.get("source_rows_replayed") != len(source_rows)
        or live.get("evaluation_rows_replayed") != base["evaluation_rows_exact"]
        or live.get("packages_replayed") != 2
        or live.get("physical_extractions_replayed") != 2
    ):
        raise R14Error("R16 live evidence binding changed")
    result = {
        **{key: value for key, value in base.items() if key != "evidence_sha256"},
        "format": "abi-r16-strict-verification/2",
        "declared_artifacts_verified": 13,
        "source_residual_rows_verified": len(extraction_rows),
        "live_files_verified": len(observed_live),
        "source_snapshot_evidence_sha256": config["source"]["snapshot_evidence_sha256"],
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_v2(args.config, args.reveal, args.run_dir, args.live)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
