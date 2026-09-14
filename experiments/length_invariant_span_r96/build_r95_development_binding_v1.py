"""Bind frozen R96 to disclosed R95 evidence before development execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "candidate-binding", "catalog", "source-result", "source-raw", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    candidate_binding = json.loads(args.candidate_binding.read_text(encoding="utf-8"))
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    unsigned = dict(source); source_claim = unsigned.pop("evidence_sha256", None)
    if candidate_binding.get("candidate_generation_observations_before_binding") != 0 or source.get("verdict") != "PASS_R95_SOURCE_CAPTURE" or source_claim != evidence_hash(unsigned):
        parser.error("R96 candidate or R95 source evidence is invalid")
    paths = {
        "candidate_checkpoint": args.candidate / "bridge.safetensors", "candidate_metadata": args.candidate / "metadata.json",
        "candidate_freeze_binding": args.candidate_binding, "catalog": args.catalog,
        "source_result": args.source_result, "source_raw": args.source_raw,
        "candidate_core": root / "experiments/length_invariant_span_r96/core_v1.py",
        "candidate_trainer": root / "experiments/length_invariant_span_r96/train_v1.py",
        "candidate_protocol": root / "experiments/length_invariant_span_r96/PROTOCOL.md",
        "shared_screen": root / "experiments/source_qualified_span_r93/screen_v1.py",
        "screen_wrapper": root / "experiments/length_invariant_span_r96/screen_r95_development_v1.py",
    }
    value = {
        "format": "abi-r96-r95-development-binding/1", "candidate_observations_before_binding": 0,
        "candidate_retrained_after_r91_development": True, "source_pass_field": "raw_passed",
        "files": {name: {"path": str(path.resolve().relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": _sha256_file(path)} for name, path in paths.items()},
        "source_evidence_sha256": source_claim, "promotion_eligible": False, "full_abi_moonshot": "OPEN",
    }
    value["binding_sha256"] = _canonical_sha(value); write_json_once(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
