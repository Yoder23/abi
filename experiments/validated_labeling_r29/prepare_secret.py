"""Create the R29 hidden config and separately revealable secret."""

from __future__ import annotations
import argparse, hashlib, json, secrets
from pathlib import Path
from experiments.foreign_capability_r14.core import json_object, sha256_file, write_json_once
from .binding import CODE_PATHS


def item(root, relative):
    path = root / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
def run(freeze_commit, config_path, reveal_path):
    root = Path(__file__).resolve().parents[2]; public = "results/validated_labeling_r29/public_v1/result.json"; host = "experiments/autonomous_labeling_r27/configs/heldout_v1.json"
    if json_object(root / public).get("verdict") != "PASS": raise ValueError("R29 public prerequisite failed")
    secret = secrets.token_bytes(32); reveal = {"format": "abi-r29-heldout-reveal/1", "secret_hex": secret.hex()}; reveal_path.parent.mkdir(parents=True, exist_ok=True); write_json_once(reveal_path, reveal)
    config = {"format": "abi-r29-heldout-config/1", "implementation_freeze_commit": freeze_commit, "heldout_seed_commitment": hashlib.sha256(secret).hexdigest(), "reveal_sha256": sha256_file(reveal_path), "facts_per_domain": 3, "label_choices_supplied": 0, "validator": "safe-stdlib-ast/1", "minimum_parse": 68, "minimum_raw_answer_exact": 60, "training_authorized": False, "physical_extraction": {"distribution": "Ubuntu", "policy": "linux-pivot-root-no-network/1"}, "source": {"model_id": "Qwen/Qwen2-7B-Instruct", "revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c", "max_new_tokens": 32, "training_steps": 0}, "public_prerequisite": item(root, public), "r27_host_config": item(root, host), "code_sha256": {relative: sha256_file(root / relative) for relative in CODE_PATHS}}
    config_path.parent.mkdir(parents=True, exist_ok=True); write_json_once(config_path, config); print(json.dumps({"commitment": config["heldout_seed_commitment"], "reveal_sha256": config["reveal_sha256"]}, indent=2))
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze-commit", required=True); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--reveal", type=Path, required=True); args = parser.parse_args(); run(args.freeze_commit, args.config, args.reveal)
if __name__ == "__main__": main()
