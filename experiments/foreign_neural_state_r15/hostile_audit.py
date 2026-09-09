"""Hostile mutation audit for the fail-closed R15A verifier."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.foreign_capability_r14.core import (
    evidence_hash,
    json_object,
    write_json_once,
)

from .verify import verify


def _copy_tree(source: Path, destination: Path) -> None:
    """Make a cheap copy whose files are detached before byte mutation."""

    shutil.copytree(source, destination, copy_function=os.link)


def _flip_one_byte(path: Path) -> None:
    content = bytearray(path.read_bytes())
    if not content:
        raise RuntimeError(f"cannot mutate empty evidence: {path}")
    content[len(content) // 2] ^= 1
    path.unlink()
    path.write_bytes(content)


def audit(config_path: Path, reveal_path: Path, run_dir: Path) -> dict[str, Any]:
    baseline = verify(config_path, reveal_path, run_dir)
    if baseline.get("verdict") != "PASS":
        raise RuntimeError("R15A hostile-audit baseline is not a strict PASS")
    receipt = json_object(run_dir / "receipt.json")
    manifest = json_object(run_dir / "package_manifest.json")
    results: list[dict[str, Any]] = []

    def rejected(
        name: str,
        mutation: Callable[[Path], Path | None],
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="abi-r15a-hostile-") as temporary:
            copied_run = Path(temporary) / "run"
            _copy_tree(run_dir, copied_run)
            changed_reveal = mutation(copied_run) or reveal_path
            try:
                verify(config_path, changed_reveal, copied_run)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                results.append(
                    {"case": name, "outcome": "REJECTED", "error": str(exc)}
                )
                return
            results.append({"case": name, "outcome": "UNSAFE_ACCEPT"})

    rejected(
        "missing_source_observations",
        lambda copied: (copied / receipt["source_observations"]["path"]).unlink()
        or None,
    )
    rejected(
        "corrupted_source_observations",
        lambda copied: _flip_one_byte(
            copied / receipt["source_observations"]["path"]
        ),
    )
    first_source = receipt["source_acquisitions"][0]
    rejected(
        "missing_source_adapter",
        lambda copied: (copied / first_source["adapter_artifact"]["path"]).unlink()
        or None,
    )
    rejected(
        "corrupted_effective_delta",
        lambda copied: _flip_one_byte(copied / first_source["delta_artifact"]["path"]),
    )
    rejected(
        "missing_capability_package",
        lambda copied: (
            copied / "packages" / manifest["after"][0]["path"]
        ).unlink()
        or None,
    )
    first_extraction = receipt["neural_state_extractions"][0]
    rejected(
        "corrupted_isolated_extraction",
        lambda copied: _flip_one_byte(
            copied / first_extraction["isolated_result"]["path"]
        ),
    )
    first_worker = receipt["recipient_workers"][0]
    rejected(
        "missing_recipient_rows",
        lambda copied: (
            copied
            / "recipients"
            / first_worker["host"]
            / first_worker["observations"]["path"]
        ).unlink()
        or None,
    )

    def wrong_reveal(copied: Path) -> Path:
        value = json_object(reveal_path)
        value["secret_hex"] = "00" * 32
        path = copied / "wrong_reveal.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    rejected("wrong_heldout_reveal", wrong_reveal)
    passed = all(item["outcome"] == "REJECTED" for item in results)
    result = {
        "format": "abi-r15a-hostile-audit/1",
        "verdict": "PASS" if passed else "FAIL",
        "cases_passed": sum(item["outcome"] == "REJECTED" for item in results),
        "cases_total": len(results),
        "cases": results,
        "baseline_evidence_sha256": baseline["evidence_sha256"],
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(
        Path(args.config).resolve(),
        Path(args.reveal).resolve(),
        Path(args.run_dir).resolve(),
    )
    write_json_once(Path(args.output).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
