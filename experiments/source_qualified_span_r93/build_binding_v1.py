"""Bind the frozen R91 candidate to qualified R93 source evidence."""

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
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    unsigned = dict(source)
    source_claim = unsigned.pop("evidence_sha256", None)
    if source.get("verdict") != "PASS_R93_SOURCE" or source.get("candidate_accessed") is not False or source.get("layercake_loaded") is not False or not all(source.get("gates", {}).values()) or source_claim != evidence_hash(unsigned):
        parser.error("R93 source is absent, failed, or unrecomputable")
    paths = {
        "candidate_checkpoint": args.candidate / "bridge.safetensors",
        "candidate_metadata": args.candidate / "metadata.json",
        "candidate_development_binding": args.candidate_binding,
        "catalog": args.catalog, "source_result": args.source_result, "source_raw": args.source_raw,
        "candidate_core": root / "experiments/structural_span_r91/core_v1.py",
        "candidate_trainer": root / "experiments/structural_span_r91/train_v1.py",
        "candidate_protocol": root / "experiments/structural_span_r91/PROTOCOL.md",
        "screen": root / "experiments/source_qualified_span_r93/screen_v1.py",
        "protocol": root / "experiments/source_qualified_span_r93/PROTOCOL.md",
        "builder": root / "experiments/source_qualified_span_r93/build_catalog_v1.py",
        "source_scorer": root / "experiments/source_qualified_span_r93/score_source_v1.py",
    }
    value = {
        "format": "abi-r93-source-qualified-candidate-binding/1",
        "candidate_observations_before_binding": 0,
        "candidate_retrained_after_r91_development": False,
        "files": {name: {
            "path": str(path.resolve().relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size, "sha256": _sha256_file(path),
        } for name, path in paths.items()},
        "source_evidence_sha256": source_claim,
        "full_abi_moonshot": "OPEN",
    }
    value["binding_sha256"] = _canonical_sha(value)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
