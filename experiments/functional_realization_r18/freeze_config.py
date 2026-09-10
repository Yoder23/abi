"""Create the R18 preregistration without printing or copying its secret."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

CODE_PATHS = (
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/linguistic_realization_r17/frames.py",
    "experiments/linguistic_realization_r17/frames_v2.py",
    "experiments/linguistic_realization_r17/public_qualification.py",
    "experiments/linguistic_realization_r17/verify_source.py",
    "experiments/functional_realization_r18/PUBLIC_PROTOCOL.md",
    "experiments/functional_realization_r18/HELDOUT_PROTOCOL.md",
    "experiments/functional_realization_r18/package.py",
    "experiments/functional_realization_r18/isolated_worker.py",
    "experiments/functional_realization_r18/extraction_pivot_runner.sh",
    "experiments/functional_realization_r18/isolation.py",
    "experiments/functional_realization_r18/run_public.py",
    "experiments/functional_realization_r18/verify.py",
    "experiments/functional_realization_r18/hidden_frames.py",
    "experiments/functional_realization_r18/heldout_protocol.py",
    "experiments/functional_realization_r18/acquire_heldout.py",
    "experiments/functional_realization_r18/verify_heldout_source.py",
    "experiments/functional_realization_r18/run_heldout.py",
    "experiments/functional_realization_r18/verify_heldout.py",
)

PUBLIC_EVIDENCE = {
    "receipt": "results/functional_realization_r18/public_v1/receipt.json",
    "strict": "results/functional_realization_r18/public_v1/strict_verification.json",
    "hostile": "results/functional_realization_r18/public_v1/hostile_audit.json",
    "live": "results/functional_realization_r18/public_v1_live_replay/live_verification.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.exists():
        raise RuntimeError(f"immutable R18 config exists: {args.output}")
    reveal = json.loads(args.reveal.read_text(encoding="utf-8"))
    if reveal.get("format") != "abi-r18-heldout-reveal/1":
        raise RuntimeError("invalid R18 reveal format")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = {
        "format": "abi-r18-heldout-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {relative: sha256_file(root / relative) for relative in CODE_PATHS},
        "heldout_seed_commitment": reveal["commitment"],
        "reveal_sha256": sha256_file(args.reveal),
        "public_prerequisite": {
            name: {"path": path, "sha256": sha256_file(root / path)}
            for name, path in PUBLIC_EVIDENCE.items()
        },
        "source": {
            "model_id": "Qwen/Qwen2-7B-Instruct",
            "revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
            "max_new_tokens": 48,
            "device": "cuda",
            "snapshot_evidence_sha256": "bdb164548ca29e580e465f520df01fce77524eeec01e807c413443751b4f004b",
        },
        "data": {
            "signatures": 24,
            "extraction_rows_per_signature": 3,
            "evaluation_rows_per_signature": 2,
            "extraction_rows": 72,
            "evaluation_rows": 48,
            "lexeme_bank": 20,
        },
        "gates": {
            "minimum_parseable_per_signature": 2,
            "package_functional_exact": 48,
            "removed_abstain": 48,
            "max_control_accuracy": 0.10,
            "max_source_exact_regressions": 0,
            "require_package_at_least_modal": True,
        },
        "physical_extraction": {
            "distribution": "Ubuntu",
            "policy": "linux-pivot-root-no-network/1",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(config, indent=2, sort_keys=True).encode() + b"\n")
    print(reveal["commitment"])


if __name__ == "__main__":
    main()
