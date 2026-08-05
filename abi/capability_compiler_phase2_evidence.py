"""Fail-closed audit of the complete Phase 2 baseline evidence tree."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import Phase2Error, canonical_json_bytes, sha256_file
from .capability_compiler_phase2_verify import verify_protocol


EVIDENCE_ROOT = Path("results/abi_capability_compiler_phase2")
PACK_SHA256 = "c253a1f3e5fcf60f1efdc65259f0064bb18ec6b37e9f327ae4f24b2ecad53ed5"
TOPK_SHA256 = "0e574db92c6eba351d4571348c4e422e788c5f0ebcd7f82dbd291290c51e87f1"
ANALYSIS_SHA256 = "f9525cfc0adbb9284a390b7ddc542f0b3ecfdc5302c08206373493825a284b0f"
SYSTEMS = ("T0", "L0", "L1", "D0", "D1", "D2")
TRAINABLE = ("L0", "L1", "D0", "D1", "D2")
SEEDS = (104729, 130363, 155921)
GRID_COUNTS = {"L0": 8, "L1": 8, "D0": 6, "D1": 6, "D2": 1}
FULL_COUNTS = {"L0": 2, "L1": 2, "D0": 2, "D1": 2, "D2": 1}
EXPECTED_CONFIG = {
    "L0": {"rank": 16, "learning_rate": 0.0001, "target_token_exposures": 1},
    "L1": {"rank": 32, "learning_rate": 0.0001, "target_token_exposures": 1},
    "D0": {"learning_rate": 0.00003, "target_token_exposures": 4},
    "D1": {"learning_rate": 0.00003, "target_token_exposures": 4},
    "D2": {"learning_rate": 0.00003, "target_token_exposures": 4},
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise Phase2Error(f"evidence path escaped repository: {value}")
    if not resolved.is_file():
        raise Phase2Error(f"missing evidence file: {resolved}")
    return resolved


def _identity(receipt: Mapping[str, Any]) -> str:
    value = receipt.get("system") or receipt.get("method")
    if value not in SYSTEMS:
        raise Phase2Error("invalid evidence system identity")
    return str(value)


def _verify_outputs(root: Path, receipt: Mapping[str, Any], expected_observations: int) -> None:
    development = receipt.get("development")
    if not isinstance(development, Mapping) or development.get("observations") != expected_observations:
        raise Phase2Error("development evidence depth changed")
    path = _inside(root, str(development["outputs_path"]))
    if sha256_file(path) != development.get("outputs_sha256"):
        raise Phase2Error("development output hash changed")
    lines = [line for line in path.read_bytes().splitlines() if line]
    if len(lines) != expected_observations:
        raise Phase2Error("development output row count changed")


def _receipts(path: Path) -> list[Path]:
    return sorted(path.glob("**/receipt.json"))


def _verify_run_stage(
    root: Path,
    evidence: Path,
    stage: str,
    counts: Mapping[str, int],
    expected_observations: int,
) -> dict[str, int]:
    summary: dict[str, int] = {}
    for system, expected_count in counts.items():
        receipts = _receipts(evidence / stage / system)
        if len(receipts) != expected_count:
            raise Phase2Error(f"{stage} {system} receipt count changed")
        passes = 0
        for path in receipts:
            receipt = _read_json(path)
            if (
                receipt.get("status") != "PASS"
                or receipt.get("final_prompts_accessed") is not False
                or _identity(receipt) != system
            ):
                raise Phase2Error(f"invalid {stage} receipt: {path}")
            _verify_outputs(root, receipt, expected_observations)
            passes += int(receipt["development"]["functional_passes"])
        summary[system] = passes
    return summary


def validate_runtime_result(value: Mapping[str, Any], *, system: str, mode: str) -> None:
    expected_count = 1 if mode == "cold" else 20
    if (
        value.get("format") != "abi-capability-compiler-phase2-runtime/1"
        or value.get("status") != "PASS"
        or value.get("system") != system
        or value.get("mode") != mode
        or value.get("observation_count") != expected_count
        or len(value.get("observations", [])) != expected_count
        or value.get("p95_supported") is not False
        or value.get("p99_supported") is not False
        or value.get("final_prompts_accessed") is not False
    ):
        raise Phase2Error("runtime contract changed")
    for observation in value["observations"]:
        if (
            int(observation.get("output_tokens", -1)) < 0
            or float(observation.get("time_to_first_output_seconds", 0.0)) <= 0.0
            or float(observation.get("total_seconds", 0.0)) <= 0.0
        ):
            raise Phase2Error("runtime observation changed")
    if mode == "cold":
        observation = value["observations"][0]
        if (
            float(observation.get("first_output_from_cold_start_seconds", 0.0))
            <= float(value.get("model_load_seconds", 0.0))
            or float(observation.get("total_from_cold_start_seconds", 0.0))
            < float(observation.get("first_output_from_cold_start_seconds", 0.0))
        ):
            raise Phase2Error("cold-start timing changed")


def _verify_topk(root: Path, path: Path) -> dict[str, Any]:
    if sha256_file(path) != TOPK_SHA256:
        raise Phase2Error("uninterrupted top-k summary changed")
    summary = _read_json(path)
    if (
        summary.get("status") != "PASS"
        or summary.get("pack_count") != 1181
        or summary.get("response_positions") != 222647
        or summary.get("stored_logit_values") != 14249408
        or summary.get("topk") != 64
        or len(summary.get("files", [])) != 1181
    ):
        raise Phase2Error("top-k accounting changed")
    for binding in summary["files"]:
        shard = _inside(root, str(binding["path"]))
        if sha256_file(shard) != binding["sha256"]:
            raise Phase2Error("top-k shard hash changed")
    return summary


def _verify_packet(root: Path, packet_dir: Path) -> dict[str, Any]:
    manifest_path = packet_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("format") != "abi-capability-compiler-phase2-human-rating-packet/1"
        or manifest.get("status") != "AWAITING_THREE_INDEPENDENT_HUMAN_RATERS"
        or manifest.get("distinct_prompts") != 1400
        or manifest.get("pairs_per_form") != 7000
        or manifest.get("rater_forms") != 3
        or manifest.get("ratings_required") != 21000
        or manifest.get("final_prompts_accessed") is not False
    ):
        raise Phase2Error("human-rating packet contract changed")
    for name, binding in manifest["input_bindings"].items():
        if name not in SYSTEMS or sha256_file(_inside(root, binding["path"])) != binding["sha256"]:
            raise Phase2Error("human-rating input binding changed")
    rating_ids: set[str] = set()
    for form_index in range(1, 4):
        name = f"rater_form_{form_index}.jsonl"
        binding = manifest["file_bindings"][name]
        path = packet_dir / name
        if binding["rows"] != 7000 or sha256_file(path) != binding["sha256"]:
            raise Phase2Error("human-rating form binding changed")
        rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
        if len(rows) != 7000:
            raise Phase2Error("human-rating form depth changed")
        for row in rows:
            if (
                row["rating_id"] in rating_ids
                or "system_A" in row
                or "system_B" in row
                or row.get("preference") is not None
            ):
                raise Phase2Error("human-rating form blindness changed")
            rating_ids.add(row["rating_id"])
    key_binding = manifest["file_bindings"]["blinding_key.jsonl"]
    key_path = packet_dir / "blinding_key.jsonl"
    if key_binding["rows"] != 21000 or sha256_file(key_path) != key_binding["sha256"]:
        raise Phase2Error("human-rating key binding changed")
    if len(rating_ids) != 21000:
        raise Phase2Error("human-rating identity depth changed")
    return manifest


def _human_ratings_status(root: Path, packet_sha256: str) -> dict[str, Any]:
    path = root / EVIDENCE_ROOT / "human_ratings_v1" / "manifest.json"
    if not path.is_file():
        return {
            "complete": False,
            "status": "AWAITING_THREE_INDEPENDENT_HUMAN_RATERS",
            "required_raters": 3,
            "required_ratings": 21000,
        }
    value = _read_json(path)
    if (
        value.get("status") != "PASS"
        or value.get("independent_raters") != 3
        or value.get("ratings") != 21000
        or value.get("packet_manifest_sha256") != packet_sha256
    ):
        raise Phase2Error("completed human-rating manifest changed")
    return {"complete": True, **value}


def verify_evidence(root: Path) -> dict[str, Any]:
    root = root.resolve()
    evidence = root / EVIDENCE_ROOT
    protocol = verify_protocol(root)

    pack_path = evidence / "packs" / "manifest_v1.json"
    pack = _read_json(pack_path)
    if (
        sha256_file(pack_path) != PACK_SHA256
        or pack.get("status") != "PASS"
        or pack.get("pack_count") != 1181
        or pack.get("input_tokens") != 799572
        or pack.get("response_tokens") != 222647
    ):
        raise Phase2Error("packed training evidence changed")
    topk = _verify_topk(root, evidence / "topk_uninterrupted" / "summary_v1.json")

    t0_path = evidence / "teacher" / "T0" / "receipt.json"
    t0 = _read_json(t0_path)
    if (
        t0.get("status") != "PASS"
        or t0.get("system") != "T0"
        or t0.get("observations") != 1400
        or t0.get("final_prompts_accessed") is not False
        or sha256_file(_inside(root, t0["outputs_path"])) != t0["outputs_sha256"]
    ):
        raise Phase2Error("T0 evidence changed")

    grid = _verify_run_stage(root, evidence, "development", GRID_COUNTS, 140)
    full = _verify_run_stage(root, evidence, "full_development", FULL_COUNTS, 1400)

    headline: dict[str, Any] = {}
    checkpoint_hashes: set[str] = set()
    for system in TRAINABLE:
        receipts = _receipts(evidence / "headline" / system)
        if len(receipts) != 3:
            raise Phase2Error(f"headline {system} seed count changed")
        rows = []
        for path in receipts:
            receipt = _read_json(path)
            if (
                receipt.get("status") != "PASS"
                or receipt.get("final_prompts_accessed") is not False
                or _identity(receipt) != system
                or int(receipt.get("seed", -1)) not in SEEDS
            ):
                raise Phase2Error(f"invalid headline receipt: {path}")
            for field, expected in EXPECTED_CONFIG[system].items():
                if receipt.get(field) != expected:
                    raise Phase2Error(f"headline {system} configuration changed")
            _verify_outputs(root, receipt, 1400)
            checkpoint = _inside(root, receipt["checkpoint_path"])
            if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
                raise Phase2Error("headline checkpoint hash changed")
            checkpoint_hashes.add(str(receipt["checkpoint_sha256"]))
            rows.append(receipt)
        if sorted(int(row["seed"]) for row in rows) != list(SEEDS):
            raise Phase2Error(f"headline {system} seeds changed")
        headline[system] = {
            "seeds": list(SEEDS),
            "functional_passes": [int(row["development"]["functional_passes"]) for row in rows],
            "repetition_collapses": [int(row["development"]["repetition_collapses"]) for row in rows],
            "checkpoint_sha256": [str(row["checkpoint_sha256"]) for row in rows],
        }
    if len(checkpoint_hashes) != 15:
        raise Phase2Error("headline checkpoints are not seed-distinct")

    candidates: dict[str, str] = {}
    for system in TRAINABLE:
        path = evidence / "runtime_candidates" / f"{system}_seed104729.json"
        value = _read_json(path)
        if (
            value.get("status") != "FROZEN_BEFORE_RUNTIME"
            or value.get("system") != system
            or value.get("seed") != 104729
            or sha256_file(_inside(root, value["checkpoint_path"])) != value["checkpoint_sha256"]
            or sha256_file(_inside(root, value["quality_receipt_path"])) != value["quality_receipt_sha256"]
        ):
            raise Phase2Error("runtime candidate binding changed")
        candidates[system] = sha256_file(path)

    runtime: dict[str, Any] = {}
    for system in SYSTEMS:
        for mode in ("cold", "warm"):
            path = evidence / "runtime" / f"{system}_{mode}_v1.json"
            value = _read_json(path)
            validate_runtime_result(value, system=system, mode=mode)
            if system == "T0":
                if value.get("candidate_manifest_sha256") is not None:
                    raise Phase2Error("T0 gained a runtime candidate")
            elif value.get("candidate_manifest_sha256") != candidates[system]:
                raise Phase2Error("runtime candidate identity changed")
            runtime[f"{system}_{mode}"] = {
                "sha256": sha256_file(path),
                "observation_count": value["observation_count"],
                "median_bytes_per_second": value["median_bytes_per_second"],
                "median_characters_per_second": value["median_characters_per_second"],
                "median_time_to_first_output_seconds": value["median_time_to_first_output_seconds"],
                "peak_process_rss_bytes": value["peak_process_rss_bytes"],
                "peak_cuda_allocated_bytes": value["peak_cuda_allocated_bytes"],
            }

    analysis_path = evidence / "analysis" / "paired_all_headline_v2.json"
    if sha256_file(analysis_path) != ANALYSIS_SHA256:
        raise Phase2Error("paired analysis changed")
    analysis = _read_json(analysis_path)
    expected_names = {"T0"} | {f"{system}_seed{seed}" for system in TRAINABLE for seed in SEEDS}
    if set(analysis.get("systems", {})) != expected_names or set(analysis.get("paired_vs_T0", {})) != expected_names - {"T0"}:
        raise Phase2Error("paired analysis system depth changed")
    for name, binding in analysis["input_bindings"].items():
        if name not in expected_names or sha256_file(_inside(root, binding["path"])) != binding["sha256"]:
            raise Phase2Error("paired analysis input binding changed")
    for value in analysis["systems"].values():
        if value["observations"] != 1400:
            raise Phase2Error("paired analysis prompt depth changed")
    for value in analysis["paired_vs_T0"].values():
        if value["resamples"] != 10000 or value["seed"] != 1729:
            raise Phase2Error("paired bootstrap changed")

    packet_dir = evidence / "human_rating_packet_v1"
    packet = _verify_packet(root, packet_dir)
    packet_sha = sha256_file(packet_dir / "manifest.json")
    human = _human_ratings_status(root, packet_sha)
    phase2_complete = bool(human["complete"])

    return {
        "format": "abi-capability-compiler-phase2-machine-evidence/1",
        "status": "PASS" if phase2_complete else "BLOCKED_EXTERNAL_HUMAN_RATINGS",
        "machine_evidence_complete": True,
        "phase2_complete": phase2_complete,
        "phase3_status": "OPEN" if phase2_complete else "LOCKED",
        "protocol": protocol,
        "pack": {
            "sha256": PACK_SHA256,
            "pack_count": pack["pack_count"],
            "input_tokens": pack["input_tokens"],
            "response_tokens": pack["response_tokens"],
        },
        "topk": {
            "sha256": TOPK_SHA256,
            "response_positions": topk["response_positions"],
            "stored_logit_values": topk["stored_logit_values"],
            "source_inference_seconds": topk["source_inference_seconds"],
            "wall_seconds": topk["wall_seconds"],
        },
        "selection_receipt_counts": {"sentinel": GRID_COUNTS, "full": FULL_COUNTS},
        "selection_functional_pass_totals": {"sentinel": grid, "full": full},
        "teacher": {
            "functional_passes": t0["functional_passes"],
            "observations": t0["observations"],
            "repetition_collapses": t0["repetition_collapses"],
            "receipt_sha256": sha256_file(t0_path),
        },
        "headline": headline,
        "analysis": {"sha256": ANALYSIS_SHA256, "systems": len(expected_names), "bootstrap_resamples_each": 10000},
        "runtime": runtime,
        "human_rating_packet": {
            "manifest_sha256": packet_sha,
            "ratings_required": packet["ratings_required"],
            "status": packet["status"],
        },
        "human_ratings": human,
        "claim_boundary": "Phase 2 baseline machine evidence is complete. Phase 2 is not complete and Phase 3 cannot open until three independent blinded human forms are completed and verified. No ABI candidate or LayerCake candidate was trained in Phase 2.",
        "final_prompts_accessed": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify_evidence(root)
    if args.output:
        output = Path(args.output).resolve()
        if output.exists():
            raise Phase2Error("immutable Phase 2 machine-evidence report already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(result))
    print(json.dumps({key: result[key] for key in ("status", "machine_evidence_complete", "phase2_complete", "phase3_status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
