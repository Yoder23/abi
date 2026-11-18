"""Index completed R110 evidence after strict recomputation; no model execution.

This additive publication helper does not change the frozen experiment. Its
manifest includes historical dependencies, which must be obtained separately
through Git/LFS and the pinned source cache. It is not a self-contained model
archive or evidence of independent reproduction.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/review_2026_11_18"
MANIFEST = OUT / "r110_manifest.json"
RECEIPT = OUT / "r110_strict_recomputation.json"
DIAGNOSTIC = OUT / "r110_failure_inventory.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def package():
    if any(path.exists() for path in (MANIFEST, RECEIPT, DIAGNOSTIC)):
        raise RuntimeError("Preserve the existing publication packet; do not overwrite")
    sys.path.insert(0, str(ROOT))
    from experiments.scope_adherence_r110 import design as d
    from experiments.scope_adherence_r110 import evaluator as e
    from experiments.scope_adherence_r110 import execution as x
    from experiments.scope_adherence_r110.run import verify

    started = time.monotonic()
    transcript = io.StringIO()
    print("Recomputing the complete frozen R110 result; no model calls", flush=True)
    with contextlib.redirect_stdout(transcript):
        result = verify()
    elapsed = time.monotonic() - started
    frozen = x.read(x.OUT / "implementation_freeze.json")
    result_path = "results/scope_adherence_r110/execution/live/decision.json"
    if x.read(ROOT / result_path) != result:
        raise RuntimeError("Recomputed result differs from the immutable decision")

    names = {name for name in frozen["files"] if not name.startswith("../")}
    names.update(p.relative_to(ROOT).as_posix()
                 for p in (x.OUT.parent).rglob("*") if p.is_file())
    names.update(("scripts/package_r110_evidence.py", "tests/test_r110_publication.py",
                  "docs/R110_SCOPE_ADHERENCE.md",
                  "docs/R110_CLEANUP.md", "docs/releases/R110_REVIEW.md",
                  "results/review_2026_11_18/cleanup_plan.json",
                  "results/review_2026_11_18/cleanup_completion.json"))
    names.update(p.relative_to(ROOT).as_posix() for p in OUT.glob("r110-*.xml"))
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    if names - tracked:
        raise RuntimeError(f"Stage required publication files first: {sorted(names - tracked)}")
    for name in names:
        if not (ROOT / name).resolve().is_relative_to(ROOT):
            raise RuntimeError("Invalid publication path")
    lfs = set(subprocess.check_output(["git", "lfs", "ls-files", "-n"],
                                      cwd=ROOT, text=True).splitlines()) & names
    write_new(RECEIPT, {
        "format": "r110-strict-recomputation/1",
        "verification_entrypoint": "experiments.scope_adherence_r110.run.verify",
        "elapsed_seconds": elapsed,
        "result_path": result_path,
        "result_sha256": sha(ROOT / result_path),
        "stdout": transcript.getvalue(),
        "model_calls": 0,
        "scope": "Same-machine full raw recomputation, not another live or independent-hardware run",
    })
    names.add(RECEIPT.relative_to(ROOT).as_posix())
    descriptive = {}
    for interface in d.INTERFACES:
        raw_path = x.OUT / "live" / interface / "initial/raw.jsonl"
        raw = x.rows(raw_path)
        scored = [(ref, e.screen(ref, row["output"])) for ref, row in zip(d.population(), raw)]
        descriptive[interface] = {
            "raw_sha256": sha(raw_path),
            "sentence_limited_rows": 24,
            "sentence_count_failures": [ref["id"] for ref, screen in scored
                                        if not screen["requested_sentence_count"]],
            "approval_class_counts": dict(Counter(screen["approval_scope"]["classification"]
                                                   for _, screen in scored)),
            "unresolved_rows": [ref["id"] for ref, screen in scored
                                if screen["approval_scope"]["classification"] == e.UNKNOWN],
        }
    write_new(DIAGNOSTIC, {
        "format": "r110-descriptive-failure-inventory/1",
        "result_sha256": sha(ROOT / result_path),
        "scope": "Post-result counts from unchanged scoring; initial observations only, no rescoring or new gate",
        "interfaces": descriptive,
    })
    names.add(DIAGNOSTIC.relative_to(ROOT).as_posix())
    value = {
        "format": "r110-public-manifest/1",
        "tag": "abi-research-r110-2026-11-18",
        "files": {name: {"sha256": sha(ROOT / name), "bytes": (ROOT / name).stat().st_size}
                  for name in sorted(names)},
        "host_inputs": {name: digest for name, digest in frozen["files"].items()
                        if name.startswith("../")},
        "host_revision": x.h.HOST_REVISION,
        "source_model": x.h.SOURCE,
        "source_revision": x.h.REVISION,
        "source_files": frozen["source_files"],
        "packages": frozen["packages"],
        "lfs_paths": sorted(lfs),
        "result_path": result_path,
        "result_sha256": sha(ROOT / result_path),
        "scope_decision_sha256": sha(x.OUT / "scope/decision.json"),
        "scope": "Publication custody; external source cache and historical Git/LFS dependencies remain required",
    }
    write_new(MANIFEST, value)
    print(json.dumps({"manifest": str(MANIFEST), "manifest_sha256": sha(MANIFEST),
                      "assets": len(names), "host_bindings": len(value["host_inputs"]),
                      "strict_recomputation_seconds": elapsed, "verdict": result["verdict"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    package()
