"""Report archive signatures found below explicitly named diagnostic roots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from abi_v2.isolated_certification import _capability_archive_signatures


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args(argv)
    findings: list[dict[str, object]] = []
    for raw_root in args.roots:
        root = Path(raw_root)
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
        for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
            names[:] = sorted(names)
            candidates.extend(Path(directory) / name for name in sorted(filenames))
        for path in candidates:
            if path.is_symlink() or not path.is_file():
                continue
            result = _capability_archive_signatures(path)
            if result["signatures"] or result["unsupported_signatures"]:
                findings.append(
                    {
                        "path": path.as_posix(),
                        "capability_archive_signatures": result["signatures"],
                        "unsupported_archive_signatures": result[
                            "unsupported_signatures"
                        ],
                    }
                )
                print(json.dumps(findings[-1], sort_keys=True), flush=True)
    print(json.dumps({"findings": len(findings)}, sort_keys=True))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
