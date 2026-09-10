"""Run the frozen R22 normalized-target matched bakeoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)

from .binding import load_config
from .normalize import validate_source


def run(config_path: Path, source_run: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    output = output.resolve()
    source_run = source_run.resolve()
    config_path = config_path.resolve()
    if output.exists():
        raise R14Error(f"immutable R22 result exists: {output}")
    if not output.is_relative_to(root):
        raise R14Error("R22 result must remain inside the ABI repository")
    config = load_config(root, config_path)
    source_rows = validate_source(root, config_path, source_run)
    receipt = json_object(source_run / "receipt.json")
    if receipt.get("verdict") != "PASS_NORMALIZED_SOURCE":
        raise R14Error("R22 forbids training after a failed source prerequisite")
    base_config = (root / str(config["base_training_config"]["path"])).resolve()
    engine_dir = output / "engine"
    original_validator = base_run._validate_source
    original_layercake = base_run._layercake

    def bound_validator(
        _root: Path, candidate_config: Path, candidate_source: Path
    ) -> list[dict[str, Any]]:
        if sha256_file(candidate_config) != config["base_training_config"]["sha256"]:
            raise R14Error("R22 bakeoff received a different base config")
        if candidate_source.resolve() != source_run:
            raise R14Error("R22 bakeoff received a different source directory")
        return source_rows

    def repaired_layercake(candidate_root: Path) -> dict[str, Any]:
        api = original_layercake(candidate_root)
        manifest_type = api["CakeManifest"]

        def repaired_manifest(*args: Any, **kwargs: Any) -> Any:
            contract = dict(kwargs.get("input_contract", {}))
            if "mode" in contract:
                raise R14Error("R22 base manifest unexpectedly declares mode")
            contract["mode"] = "direct_selected_portable_decoder"
            kwargs["input_contract"] = contract
            return manifest_type(*args, **kwargs)

        api["CakeManifest"] = repaired_manifest
        return api

    base_run._validate_source = bound_validator
    base_run._layercake = repaired_layercake
    try:
        engine = base_run.run(base_config, source_run, engine_dir)
    finally:
        base_run._validate_source = original_validator
        base_run._layercake = original_layercake

    result = {
        "format": "abi-r22-semantic-plan-bakeoff-result/1",
        "verdict": engine["verdict"],
        "scientific_claim": (
            "R22_BOUNDED_PUBLIC_SEMANTIC_PLAN_TRANSFER_CANDIDATE"
            if engine["verdict"] == "PASS_PUBLIC_PREREQUISITE"
            else "R22_BOUNDED_PUBLIC_SEMANTIC_PLAN_TRANSFER_FAILED"
        ),
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "config_sha256": sha256_file(config_path),
        "normalized_source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "normalized_source_rows_sha256": sha256_file(
            source_run / "source_observations.jsonl"
        ),
        "engine_result": {
            "path": "engine/result.json",
            "sha256": sha256_file(engine_dir / "result.json"),
            "evidence_sha256": engine["evidence_sha256"],
        },
        "normalized_source_rows": len(source_rows),
        "raw_teacher_responses_preserved": True,
        "normalized_targets_used_for_training": True,
        "teacher_present_at_student_training": False,
        "teacher_present_at_package_execution": False,
        "layercake_code_changes": 0,
        "input_contract_mode": "direct_selected_portable_decoder",
        "full_abi_moonshot": "OPEN",
        "next_action": (
            "fresh live replay, then preregistered hidden replication"
            if engine["verdict"] == "PASS_PUBLIC_PREREQUISITE"
            else "preserve the negative and close the R22 branch"
        ),
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.config, args.source_run, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
