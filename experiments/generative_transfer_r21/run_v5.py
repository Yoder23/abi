"""Run R21 with exactly one additive LayerCake manifest-contract field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.foreign_capability_r14.core import (
    R14Error,
    sha256_file,
    write_json_once,
)

from . import run as base_run
from . import run_v4
from .hash_assurance_binding import selfless_evidence_hash
from .manifest_repair_binding import load_manifest_repair_config


def run(config_path: Path, source_run: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 manifest-repaired result exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_manifest_repair_config(root, config_path)
    assurance_config = root / str(config["hash_assurance_config"]["path"])
    original_layercake = base_run._layercake

    def repaired_layercake(candidate_root: Path):
        api = original_layercake(candidate_root)
        manifest_type = api["CakeManifest"]

        def repaired_manifest(*args, **kwargs):
            contract = dict(kwargs.get("input_contract", {}))
            if "mode" in contract:
                raise R14Error("R21 base manifest unexpectedly already declares mode")
            contract.update(config["input_contract_addition"])
            kwargs["input_contract"] = contract
            return manifest_type(*args, **kwargs)

        api["CakeManifest"] = repaired_manifest
        return api

    base_run._layercake = repaired_layercake
    try:
        result = run_v4.run(assurance_config, source_run, output)
    finally:
        base_run._layercake = original_layercake

    binding = {
        "format": "abi-r21-manifest-repair-run/1",
        "manifest_repair_config_sha256": sha256_file(config_path),
        "result_sha256": sha256_file(output / "result.json"),
        "result_evidence_sha256": result["evidence_sha256"],
        "input_contract_addition": config["input_contract_addition"],
        "retraining_performed": True,
        "data_changes_performed": False,
        "model_changes_performed": False,
        "gate_changes_performed": False,
        "layercake_changes_performed": False,
        "full_abi_moonshot": "OPEN",
    }
    binding["evidence_sha256"] = selfless_evidence_hash(binding)
    write_json_once(output / "manifest_repair_binding.json", binding)
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
