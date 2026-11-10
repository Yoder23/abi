"""Complete local custody inventory for R102, including transitive LFS inputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.coverage_exposure_r102 import core as c  # noqa: E402


def main():
    frozen = c.binding()
    decision = c.read(c.OUT / "decision.json")
    if (c.OUT / "stop.json").exists() or decision["verdict"] != "BOUNDED_FACTORIAL_COMPLETED_NO_PRODUCT_PROMOTION":
        raise RuntimeError("R102 did not complete")
    for index in range(16):
        if c.read(c.OUT / "operations" / f"stage-{index:02d}.json")["returncode"]:
            raise RuntimeError("Failed stage in completed study")
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    names = {name for name in frozen["files"] if not name.startswith("../")}
    for directory in (c.OUT, ROOT / "results/coverage_exposure_r102_error_audit_v1"):
        names.update(p.relative_to(ROOT).as_posix() for p in directory.rglob("*") if p.is_file())
    names.update(("scripts/audit_r102_errors.py", "tests/test_r102_error_audit.py",
                  "scripts/create_r102_review_manifest.py", "scripts/check_r102_git_custody.py"))
    names.update(f"results/review_2026_11_10/{name}" for name in
                 ("r102-final-default.xml", "r102-final-phase7.xml", "r102-final-ci-surface.xml"))
    if names - tracked:
        raise RuntimeError(f"Untracked required inputs: {sorted(names-tracked)}")
    ordered = sorted(names)
    attrs = subprocess.check_output(["git", "check-attr", "-z", "--stdin", "filter"], cwd=ROOT,
                                    input=b"\0".join(p.encode() for p in ordered) + b"\0")
    fields = attrs.split(b"\0")[:-1]
    if len(fields) != 3*len(ordered):
        raise RuntimeError("Incomplete Git attribute response")
    lfs = {fields[i].decode() for i in range(0, len(fields), 3) if fields[i+2] == b"lfs"}
    if len(lfs) != 14:
        raise RuntimeError("Unexpected transitive checkpoint dependency set")
    inventory = {name: {"sha256": c.sha(ROOT / name), "bytes": (ROOT / name).stat().st_size,
                         "storage": "git_lfs" if name in lfs else "git"} for name in ordered}
    source = c.read(c.OUT / "source/receipt.json")
    old_source = c.read(c.previous.OUT / "source/result.json")["accounting"]
    old_r99 = c.read(c.previous.r99.OUT / "source/result.json")["accounting"]
    early = {}
    for method in c.METHODS:
        current = c.OUT / "checkpoints" / f"{method}_repeat_s384/model.safetensors"
        original = c.previous.baseline.OUT / method / "model.safetensors"
        if c.sha(current) != c.sha(original):
            raise RuntimeError("Original R99 checkpoint identity no longer matches")
        early[method] = {"sha256": c.sha(current), "original_path": original.relative_to(ROOT).as_posix()}
    value = {
        "format": "r102-complete-local-review/1", "client_date": "2026-11-10",
        "publication_state_at_creation": "LOCAL_ONLY_PENDING_USER_CHOICE",
        "intended_local_tag": "abi-research-r102-2026-11-10",
        "source_commit_at_inventory": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": inventory, "lfs_paths": sorted(lfs),
        "sibling_host_commit": "865a2b9e59e6221caa1185f9b64479d2077953f3",
        "host_source_binding_commit": c.previous.HOST_COMMIT,
        "external_host_inputs": {name.removeprefix("../layercake_release/"): digest for name, digest in frozen["files"].items()
                                 if name.startswith("../layercake_release/")},
        "freeze_sha256": c.sha(c.OUT / "freeze.json"), "frozen_inputs": len(frozen["files"]),
        "decision_sha256": c.sha(c.OUT / "decision.json"), "repeated_384_checkpoint_identity": early,
        "new_teacher_queries": source["new_queries"], "new_teacher_generated_tokens": source["generated_tokens"],
        "qwen_control_series_total_queries": old_source["total_source_requests_including_prior"] + old_r99["new_raw_prompts"] + source["new_queries"],
        "qwen_control_series_total_generated_tokens": old_source["total_source_generated_tokens_including_prior"] + old_r99["new_source_generated_tokens"] + source["generated_tokens"],
        "import_note": "Series totals exclude older parent acquisition; all 87,710,428 inherited parameters and their provenance remain imported information, not from-scratch learning",
        "source_teacher": {"model": c.previous.SOURCE, "revision": c.previous.REVISION,
                           "tokenizer_files": {name: record for name, record in source["source_inventory"].items()
                                               if name in {"tokenizer.json", "tokenizer_config.json", "config.json", "vocab.json", "merges.txt"}}},
        "packages": frozen["packages"], "python": frozen["python"],
        "scope": "One-seed matched factorial and explicitly post-run descriptive error inventory; no fluent-core promotion",
        "full_moonshot": "NOT_PROVEN", "independent_hardware": "NOT_EXECUTED",
    }
    path = ROOT / "results/review_2026_11_10/r102_manifest.json"
    c.write_once(path, value)
    print(json.dumps({"sha256": c.sha(path), "files": len(inventory), "lfs_paths": len(lfs),
                      "unique_lfs_objects": len({inventory[p]["sha256"] for p in lfs}),
                      "bytes": sum(r["bytes"] for r in inventory.values())}))


if __name__ == "__main__":
    main()
