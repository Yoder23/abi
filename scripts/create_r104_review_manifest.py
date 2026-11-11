"""Inventory the complete R104 review, without recursing into unrelated payloads."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.evaluator_control_r104 import core as c  # noqa: E402


def main():
    frozen = c.binding()
    decision = c.read(c.OUT / "decision.json")
    if decision["verdict"] not in ("CONTROL_QUALIFICATION_FAILED", "BOUNDED_RECIPIENT_INTERFACE_DIAGNOSTIC_COMPLETED") or (c.OUT / "stop.json").exists():
        raise RuntimeError("Expected complete frozen diagnostic")
    names = {p for p in frozen["files"] if not p.startswith("../")}
    names.update(p.relative_to(ROOT).as_posix() for p in c.OUT.rglob("*") if p.is_file())
    names.update(("scripts/create_r104_review_manifest.py", "scripts/review_r104_public.py",
                  "scripts/review_r102_public.py", "tests/test_r104_completed_verifier.py"))
    names.add("experiments/evaluator_control_r104/error_audit.py")
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "results/review_2026_11_10").glob("r104-*.xml"))
    names.add("results/review_2026_11_10/r104_preflight_operations.md")
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    if names - tracked:
        raise RuntimeError(f"Required assets not staged/tracked: {sorted(names-tracked)}")
    ordered = sorted(names)
    attrs = subprocess.check_output(["git", "check-attr", "-z", "--stdin", "filter"], cwd=ROOT,
                                    input=b"\0".join(p.encode() for p in ordered)+b"\0").split(b"\0")[:-1]
    if len(attrs) != len(ordered)*3:
        raise RuntimeError("Incomplete Git attributes")
    lfs = {attrs[i].decode() for i in range(0, len(attrs), 3) if attrs[i+2] == b"lfs"}
    if len(lfs) != 4:
        raise RuntimeError("Unexpected R104 checkpoint dependency set")
    files = {p: {"sha256": c.sha(ROOT/p), "bytes": (ROOT/p).stat().st_size,
                 "storage": "git_lfs" if p in lfs else "git"} for p in ordered}
    manifest = {"format": "r104-public-review-inventory/1", "files": files, "lfs_paths": sorted(lfs),
        "sibling_host_commit": c.HOST_CHECKOUT, "source_tag": "abi-research-r104-2026-11-10",
        "freeze_sha256": c.sha(c.OUT / "freeze.json"), "decision_sha256": c.sha(c.OUT / "decision.json"),
        "frozen_inputs": len(frozen["files"]), "external_host_inputs": {p: h for p, h in frozen["files"].items() if p.startswith("../")},
        "control_source": {"model": c.SOURCE, "revision": c.REVISION, "files": frozen["control_files"]},
        "tokenizer_files_required_for_recomputation": ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
        "packages": frozen["packages"], "python": frozen["python"],
        "scope": "Bounded evaluator/control-gated recipient diagnostic; no training or product promotion",
        "new_model_checkpoints": 0, "full_moonshot": "NOT_PROVEN"}
    path = ROOT / "results/review_2026_11_10/r104_manifest.json"
    c.write_once(path, manifest)
    print(json.dumps({"manifest_sha256": c.sha(path), "files": len(files), "lfs_paths": len(lfs)}))


if __name__ == "__main__":
    main()
