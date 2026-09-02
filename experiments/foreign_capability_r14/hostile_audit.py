"""Hostile controls for the recomputed R14 negative result."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .capability import heldout_capabilities
from .core import (
    capability_rows,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)
from .diagnose import diagnose
from .verify_live import verify as verify_live


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)


def _replace_bytes(path: Path, content: bytes) -> None:
    path.unlink()
    path.write_bytes(content)


def _bundle(
    config: Path, reveal: Path, run_dir: Path, replay_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    return diagnose(config, reveal, run_dir), verify_live(config, reveal, run_dir, replay_dir)


def _science(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"evidence_sha256", "run_evidence_sha256"}
    }


def audit(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    replay_dir: Path,
) -> dict[str, Any]:
    baseline_diagnosis, baseline_live = _bundle(config_path, reveal_path, run_dir, replay_dir)
    if baseline_diagnosis["verdict"] != "FAIL" or baseline_live["verdict"] != "PASS":
        raise RuntimeError("hostile audit baseline is not a verified negative")
    results = []

    def rejected(name: str, mutation: Callable[[Path, Path], Path | None]) -> None:
        with tempfile.TemporaryDirectory(prefix="abi-r14-hostile-") as temporary:
            root = Path(temporary)
            copied_run = root / "run"
            copied_replay = root / "replay"
            _copy_tree(run_dir, copied_run)
            _copy_tree(replay_dir, copied_replay)
            changed_reveal = mutation(copied_run, copied_replay) or reveal_path
            try:
                _bundle(config_path, changed_reveal, copied_run, copied_replay)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                results.append({"case": name, "outcome": "REJECTED", "error": str(exc)})
                return
            results.append({"case": name, "outcome": "UNSAFE_ACCEPT"})

    rejected(
        "missing_source_observations",
        lambda copied_run, _copied_replay: (
            (copied_run / "source_observations.jsonl").unlink() or None
        ),
    )
    first_source = json_object(run_dir / "receipt.json")["source_acquisitions"][0]
    adapter_relative = Path(first_source["adapter_artifact"]["path"])
    rejected(
        "missing_source_adapter",
        lambda copied_run, _copied_replay: ((copied_run / adapter_relative).unlink() or None),
    )
    manifest = json_object(run_dir / "package_manifest.json")
    package_relative = Path("packages") / str(manifest["after"][0]["path"])

    def corrupt_package(copied_run: Path, _copied_replay: Path) -> None:
        path = copied_run / package_relative
        content = bytearray(path.read_bytes())
        content[len(content) // 2] ^= 1
        _replace_bytes(path, bytes(content))

    rejected("corrupted_package", corrupt_package)

    def wrong_reveal(_copied_run: Path, copied_replay: Path) -> Path:
        value = json_object(reveal_path)
        value["secret_hex"] = "00" * 32
        path = copied_replay / "wrong_reveal.json"
        path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        return path

    rejected("wrong_heldout_reveal", wrong_reveal)
    first_worker = json_object(replay_dir / "receipt.json")["workers"][0]
    replay_relative = Path("sources") / "0" / str(first_worker["observations"]["path"])
    rejected(
        "missing_live_replay_rows",
        lambda _copied_run, copied_replay: ((copied_replay / replay_relative).unlink() or None),
    )

    def forged_source(copied_run: Path, _copied_replay: Path) -> None:
        config = json_object(config_path)
        reveal = json_object(reveal_path)
        capabilities = heldout_capabilities(
            str(reveal["secret_hex"]),
            expected_commitment=str(config["heldout_seed_commitment"]),
            count=int(config["data"]["heldout_capabilities"]),
        )
        generated = capability_rows(config, capabilities)
        answers = {
            str(row["row_id"]): int(row["answer"])
            for item in generated
            for split in ("queries", "evaluation", "counterfactual")
            for row in item[split]
        }
        path = copied_run / "source_observations.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        target = next(
            row for row in rows if row["split"] == "EVALUATION" and row["row_id"] in answers
        )
        probabilities = target["canonical_probabilities"]
        answer = answers[str(target["row_id"])]
        predicted = max(range(8), key=probabilities.__getitem__)
        mutable = [index for index in range(8) if index not in {answer, predicted}]
        donor, recipient = sorted(mutable, key=probabilities.__getitem__, reverse=True)[:2]
        delta = min(float(probabilities[donor]) / 2.0, 1e-7)
        probabilities[donor] -= delta
        probabilities[recipient] += delta
        _replace_bytes(path, b"".join(canonical_json_bytes(row) for row in rows))
        receipt_path = copied_run / "receipt.json"
        receipt = json_object(receipt_path)
        receipt["source_observations"]["sha256"] = sha256_file(path)
        payload = dict(receipt)
        payload.pop("evidence_sha256", None)
        receipt["evidence_sha256"] = evidence_hash(payload)
        _replace_bytes(receipt_path, json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")

    rejected("hash_consistent_forged_source_probability", forged_source)

    with tempfile.TemporaryDirectory(prefix="abi-r14-status-") as temporary:
        root = Path(temporary)
        copied_run = root / "run"
        copied_replay = root / "replay"
        _copy_tree(run_dir, copied_run)
        _copy_tree(replay_dir, copied_replay)
        receipt_path = copied_run / "receipt.json"
        receipt = json_object(receipt_path)
        receipt["status"] = "PASS"
        payload = dict(receipt)
        payload.pop("evidence_sha256", None)
        receipt["evidence_sha256"] = evidence_hash(payload)
        _replace_bytes(receipt_path, json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
        diagnosis, live = _bundle(config_path, reveal_path, copied_run, copied_replay)
        unchanged = (
            _science(diagnosis) == _science(baseline_diagnosis)
            and live == baseline_live
            and diagnosis["verdict"] == "FAIL"
        )
        results.append(
            {
                "case": "forged_stored_pass_boolean",
                "outcome": "IGNORED_WITH_UNCHANGED_SCIENCE" if unchanged else "UNSAFE_ACCEPT",
            }
        )
    passed = all(item["outcome"] != "UNSAFE_ACCEPT" for item in results)
    result = {
        "format": "abi-r14-hostile-audit/1",
        "verdict": "PASS" if passed else "FAIL",
        "cases_passed": sum(item["outcome"] != "UNSAFE_ACCEPT" for item in results),
        "cases_total": len(results),
        "cases": results,
        "baseline_diagnosis_evidence_sha256": baseline_diagnosis["evidence_sha256"],
        "baseline_live_evidence_sha256": baseline_live["evidence_sha256"],
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(
        Path(args.config).resolve(),
        Path(args.reveal).resolve(),
        Path(args.run_dir).resolve(),
        Path(args.replay_dir).resolve(),
    )
    write_json_once(Path(args.output).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
