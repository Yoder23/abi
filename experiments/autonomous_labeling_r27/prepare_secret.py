"""Prepare the committed R27 held-out config and separately revealable secret."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path

from experiments.foreign_capability_r14.core import json_object, sha256_file, write_json_once

from .binding import CODE_PATHS, LAYERCAKE_PATHS


def _binding(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run(freeze_commit: str, config_path: Path, reveal_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    public = "results/autonomous_labeling_r27/public_v2/receipt.json"
    public_value = json_object(root / public)
    if public_value.get("verdict") != "PASS": raise ValueError("R27 public prerequisite did not pass")
    base = json_object(root / "experiments/canonical_layercake_replication_r26/configs/import_v1.json")
    secret = secrets.token_bytes(32)
    reveal = {"format": "abi-r27-heldout-reveal/1", "secret_hex": secret.hex()}
    reveal_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_once(reveal_path, reveal)
    layercake_root = (root / "../layercake_release").resolve()
    config = {
        "format": "abi-r27-heldout-config/1", "implementation_freeze_commit": freeze_commit,
        "heldout_seed_commitment": hashlib.sha256(secret).hexdigest(), "reveal_sha256": sha256_file(reveal_path),
        "facts_per_domain": 3, "extraction_views": [0, 1, 2], "evaluation_views": [0, 1, 2],
        "label_choices_supplied": 0, "training_authorized": False, "max_control_accuracy": 0.1,
        "physical_extraction": {"distribution": "Ubuntu", "policy": "linux-pivot-root-no-network/1"},
        "source": {"model_id": "Qwen/Qwen2-7B-Instruct", "revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c", "max_new_tokens": 32, "training_steps": 0},
        "public_prerequisite": _binding(root, public),
        "code_sha256": {relative: sha256_file(root / relative) for relative in CODE_PATHS},
        "layercake_commit": base["layercake_commit"],
        "layercake_code_sha256": {relative: sha256_file(layercake_root / relative) for relative in LAYERCAKE_PATHS},
        "layercake_abi": base["layercake_abi"], "english_packages": base["english_packages"],
        "r23_config": base["r23_config"], "r23_reveal": base["r23_reveal"], "r23_hidden_rows": base["r23_hidden_rows"],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True); write_json_once(config_path, config)
    print(json.dumps({"config": str(config_path), "commitment": config["heldout_seed_commitment"], "reveal_sha256": config["reveal_sha256"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze-commit", required=True); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--reveal", type=Path, required=True)
    args = parser.parse_args(); run(args.freeze_commit, args.config, args.reveal)


if __name__ == "__main__": main()

