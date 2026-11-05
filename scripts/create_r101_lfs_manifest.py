"""Inventory every direct and transitive LFS dependency of the frozen R101 audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.execution_continuation_r101 import core as c  # noqa: E402


def main():
    frozen = c.binding()
    direct = c.read(ROOT / "results/review_2026_11_05/r101_manifest.json")
    paths = sorted({p for p in frozen["files"] if not p.startswith("../")} | set(direct["files"]))
    attrs = subprocess.check_output(["git", "check-attr", "-z", "--stdin", "filter"],
                                    input=b"\0".join(p.encode() for p in paths) + b"\0", cwd=ROOT)
    fields = attrs.split(b"\0")[:-1]
    if len(fields) != 3 * len(paths):
        raise RuntimeError("Incomplete Git attribute response")
    names = [fields[i].decode() for i in range(0, len(fields), 3) if fields[i+2] == b"lfs"]
    if len(names) != 10:
        raise RuntimeError("Unexpected R101 direct/transitive LFS dependency set")
    files = {name: {"sha256": c.sha(ROOT / name), "bytes": (ROOT / name).stat().st_size} for name in names}
    path = ROOT / "results/review_2026_11_05/r101_lfs_manifest.json"
    c.write_once(path, {"format": "r101-complete-lfs-dependencies/1", "files": files,
                 "source_tag": "abi-research-r101-2026-11-05",
                 "source_commit": "e311451c1d3e634c64c215c8ac7389a7b9dc827c",
                 "freeze_sha256": c.sha(c.OUT / "freeze.json"),
                 "direct_inventory_sha256": c.sha(ROOT / "results/review_2026_11_05/r101_manifest.json"),
                 "scope": "Includes legacy ABI-plan checkpoint and numerical-failure payload required by inherited bindings; neither is a new R101 arm"})
    print(json.dumps({"sha256": c.sha(path), "payloads": len(files), "bytes": sum(v["bytes"] for v in files.values())}))


if __name__ == "__main__":
    main()
