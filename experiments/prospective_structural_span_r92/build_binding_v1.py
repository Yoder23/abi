"""Bind the R91 candidate and R92 source evidence before candidate execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-binding", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--source-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    unsigned_source = dict(source)
    claimed_source = unsigned_source.pop("evidence_sha256", None)
    if (
        source.get("verdict") != "PASS_R92_SOURCE"
        or source.get("candidate_accessed") is not False
        or source.get("layercake_loaded") is not False
        or not all(source.get("gates", {}).values())
        or claimed_source != evidence_hash(unsigned_source)
    ):
        parser.error("R92 source is absent, failed, or unrecomputable")
    files = {
        "candidate_checkpoint": args.candidate / "bridge.safetensors",
        "candidate_metadata": args.candidate / "metadata.json",
        "candidate_development_binding": args.candidate_binding,
        "catalog": args.catalog,
        "source_result": args.source_result,
        "source_raw": args.source_raw,
        "candidate_core": root / "experiments/structural_span_r91/core_v1.py",
        "candidate_trainer": root / "experiments/structural_span_r91/train_v1.py",
        "candidate_protocol": root / "experiments/structural_span_r91/PROTOCOL.md",
        "holdout_screen": root / "experiments/prospective_structural_span_r92/screen_v1.py",
        "holdout_protocol": root / "experiments/prospective_structural_span_r92/PROTOCOL.md",
        "holdout_builder": root / "experiments/prospective_structural_span_r92/build_catalog_v1.py",
        "source_scorer": root / "experiments/prospective_structural_span_r92/score_source_v1.py",
    }
    value = {
        "format": "abi-r92-prospective-candidate-source-binding/1",
        "candidate_observations_before_binding": 0,
        "candidate_retrained_after_r91_development": False,
        "files": {
            name: {"path": str(path.resolve().relative_to(root)).replace("\\", "/"),
                   "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for name, path in files.items()
        },
        "source_evidence_sha256": claimed_source,
        "promotion_eligible_before_screen": False,
        "full_abi_moonshot": "OPEN",
    }
    value["binding_sha256"] = _canonical_sha(value)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
