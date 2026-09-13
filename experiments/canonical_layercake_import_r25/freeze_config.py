"""Freeze R25 before direct package construction and execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import json_object, sha256_file

from .binding import CODE_PATHS, GATES, LAYERCAKE_PATHS
from .protocol import BUILD_INITIALIZATIONS, HOST_INITIALIZATIONS, NAMESPACES


def _binding(root: Path, relative: str, **extra: object) -> dict[str, object]:
    target = (root / relative).resolve()
    return {
        "path": relative,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        **extra,
    }


def _commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    layercake_root = (root / "../layercake_release").resolve()
    r16_root = Path("results/factual_semantic_r16/heldout_v2")
    r16_receipt = json_object(root / r16_root / "receipt.json")
    r16_packages = [
        _binding(
            root,
            (r16_root / "extraction" / item["path"]).as_posix(),
            namespace=item["namespace"],
            facts=item["facts"],
        )
        for item in r16_receipt["packages"]
    ]
    r23_relative = "experiments/semantic_replication_r23/configs/hidden_v1.json"
    r23_config = json_object(root / r23_relative)
    base_config = json_object(root / r23_config["base_config"]["path"])
    engine = json_object(
        root / "results/generative_transfer_r21/public_v5_bakeoff/engine/result.json"
    )
    english = sorted(
        (
            _binding(
                root,
                package["path"],
                cake_id=package["cake_id"],
            )
            for package in engine["systems"]["21022"]["abi_factorized"]["packages"]
        ),
        key=lambda row: str(row["cake_id"]),
    )
    files = {
        "r16_source_rows": (r16_root / "source_observations.jsonl").as_posix(),
        "r16_receipt": (r16_root / "receipt.json").as_posix(),
        "r16_strict_verification": (
            "results/factual_semantic_r16/heldout_v2_strict_v4.json"
        ),
        "r23_config": r23_relative,
        "r23_reveal": "experiments/semantic_replication_r23/configs/hidden_v1_reveal.json",
        "r23_live_verification": (
            "results/semantic_replication_r23/hidden_v2_live/strict_verification.json"
        ),
        "r23_hidden_rows": (
            "results/semantic_replication_r23/hidden_v1_result/engine/observations.jsonl"
        ),
    }
    config = {
        "format": "abi-r25-config/1",
        "implementation_freeze_commit": _commit(root),
        "layercake_commit": _commit(layercake_root),
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in CODE_PATHS
        },
        "layercake_code_sha256": {
            relative: sha256_file(layercake_root / relative)
            for relative in LAYERCAKE_PATHS
        },
        **{name: _binding(root, relative) for name, relative in files.items()},
        "r16_packages": r16_packages,
        "english_packages": english,
        "layercake_abi": {
            "abi_version": base_config["layercake"]["abi_version"],
            "abi_sha256": base_config["layercake"]["abi_sha256"],
            "research_signing_seed_hex": base_config["layercake"][
                "research_signing_seed_hex"
            ],
        },
        "namespaces": list(NAMESPACES),
        "host_initializations": list(HOST_INITIALIZATIONS),
        "build_initializations": list(BUILD_INITIALIZATIONS),
        "gates": GATES,
        "new_source_calls_authorized": False,
        "training_authorized": False,
        "package_mutation_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R25 config")
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
