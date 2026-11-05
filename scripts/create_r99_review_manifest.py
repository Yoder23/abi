"""Inventory the completed negative pilot for public review; not certification."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.grounded_english_r99_fp32 import core as c  # noqa: E402


def main():
    manifest = c.verify_binding()
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    for relative in manifest["files"]:
        if not relative.startswith("../layercake_release/") and relative not in tracked:
            raise RuntimeError(f"Unpublished required input: {relative}")
    paths = {c.PARENT / "model.safetensors"}
    for path in c.OUT.rglob("*"):
        if path.is_file() and not path.name.startswith("trainable-step"):
            paths.add(path)
    paths |= {
        ROOT / "results/review_2026_11_05" / f"abi-seal-{name}.xml"
        for name in ("default", "phase7")
    }
    inventory = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix()
        if relative not in tracked:
            raise RuntimeError(f"Required review asset not staged/tracked: {relative}")
        inventory[relative] = {"sha256": c.sha(path), "bytes": path.stat().st_size}
    decision = c.read(c.OUT / "decision.json")
    if decision["verdict"] != "ABI_ACQUISITION_PILOT_FAILED":
        raise RuntimeError("This manifest is for the completed negative pilot only")
    value = {
        "format": "abi-r99-public-review-manifest/1",
        "intended_tag": "abi-research-2026-11-05",
        "source_commit_at_inventory": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "safetensors",
                "numpy",
                "psutil",
                "huggingface_hub",
                "pytest",
                "ruff",
            )
        },
        "files": inventory,
        "required_input_manifest": {
            "path": "results/grounded_english_r99_fp32/freeze.json",
            "sha256": c.sha(c.OUT / "freeze.json"),
        },
        "source_teacher": {
            "model": c.SOURCE,
            "revision": c.REVISION,
            "weights_in_repository": False,
            "note": "Fetch the pinned public snapshot separately for acquisition; verification uses its tokenizer files.",
        },
        "sibling_host_source_commit": c.HOST_COMMIT,
        "intermediate_trainable_snapshots": "Retained locally; final full states suffice for evaluation and are inventoried here",
        "verdict": decision["verdict"],
        "full_moonshot": "NOT_PROVEN",
        "human_fluency": "UNASSESSED",
        "independent_hardware": "NOT_EXECUTED",
    }
    output = ROOT / "results/review_2026_11_05/r99_manifest.json"
    c.write_once(output, value)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "sha256": c.sha(output),
                "assets": len(inventory),
                "total_asset_bytes": sum(r["bytes"] for r in inventory.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
