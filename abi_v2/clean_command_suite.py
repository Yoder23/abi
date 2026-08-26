"""Run the non-matrix clean-room validation, test, and build commands once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .final_validation import sha256_file, write_json


def _run(root: Path, command: list[str]) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stdout_tail": completed.stdout.decode("utf-8", errors="replace")[-1000:],
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stderr_tail": completed.stderr.decode("utf-8", errors="replace")[-1000:],
        "passed": completed.returncode == 0,
    }


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    output = root / "clean_run"
    hostile = output / "hostile_release_verification.json"
    distribution = output / "dist"
    commands = [
        [sys.executable, "-m", "abi", "self-check"],
        [
            sys.executable,
            "-m",
            "abi_v2.strict_validation",
            "--root",
            ".",
        ],
        [
            sys.executable,
            "-m",
            "abi_v2.hostile_final_validation",
            "--root",
            ".",
            "--output",
            hostile.relative_to(root).as_posix(),
        ],
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "abi/reproduce.py",
            "abi_v2/build_final_validation_bundle.py",
            "abi_v2/clean_command_suite.py",
            "abi_v2/clean_reproduction_receipt.py",
            "abi_v2/final_validation.py",
            "abi_v2/hostile_final_validation.py",
            "abi_v2/isolated_certification.py",
            "abi_v2/live_causality.py",
            "abi_v2/public_reconstruction.py",
            "abi_v2/repaired_candidate.py",
            "abi_v2/strict_hostile.py",
            "abi_v2/strict_validation.py",
            "tests/test_abi_v2_final_validation.py",
            "tests/test_public_release.py",
        ],
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            distribution.relative_to(root).as_posix(),
        ],
    ]
    rows = [_run(root, command) for command in commands]
    artifacts = []
    if distribution.is_dir():
        artifacts = [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(distribution.iterdir())
            if path.is_file()
        ]
    wheel_count = sum(row["path"].endswith(".whl") for row in artifacts)
    sdist_count = sum(row["path"].endswith(".tar.gz") for row in artifacts)
    passed = all(row["passed"] for row in rows) and wheel_count == 1 and sdist_count == 1
    return {
        "format": "abi-final-clean-command-suite/1",
        "status": "PASS_CLEAN_COMMANDS_TESTS_AND_BUILD" if passed else "FAIL_CLEAN_COMMANDS_TESTS_AND_BUILD",
        "commands": rows,
        "release_artifacts": artifacts,
        "wheel_count": wheel_count,
        "sdist_count": sdist_count,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "development_cache_reuse_allowed": False,
        "pytest_cache_disabled": True,
        "bytecode_writes_disabled": True,
        "lint_scope": "final-validation and external-reproduction changes only",
        "historical_research_tree_lint_certified": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="clean_run/command_receipt.json")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    value = run(root)
    write_json((root / args.output).resolve(), value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
