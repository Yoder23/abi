"""Recompute the V41 labeled-BPE decision from immutable raw evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_bpe_core_analysis import build_decision as build_bpe_decision
from .capability_compiler_phase3_direct_core import _json


FORMAT = "abi-capability-compiler-phase3-labeled-bpe-core-decision/1"


def build_decision(
    *,
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    fit_dir: Path,
    evaluation_dir: Path,
) -> dict[str, Any]:
    result = build_bpe_decision(
        root=root,
        protocol_path=protocol_path,
        candidate_dir=candidate_dir,
        fit_dir=fit_dir,
        evaluation_dir=evaluation_dir,
    )
    metadata = _json(candidate_dir / "metadata.json")
    auxiliary = metadata.get("training_auxiliary", {})
    auxiliary_path = candidate_dir / str(auxiliary.get("path", ""))
    if (
        metadata.get("representation") != {
            "pointer_supervision": False,
            "label_aware_training": True,
            "header_dropout": True,
            "causal_history_corruption": True,
        }
        or auxiliary.get("present_at_inference") is not False
        or not auxiliary_path.is_file()
        or sha256_file(auxiliary_path) != auxiliary.get("sha256")
    ):
        raise Phase3Error("V41 labeled-acquisition identity changed")
    initial_pass = bool(result["decision"]["initial_screen_pass"])
    result["format"] = FORMAT
    result["status"] = (
        "PASS_INITIAL_SCREEN_CONTROLS_REQUIRED"
        if initial_pass
        else "FAIL_INITIAL_SCREEN_LABELED_BPE_CANDIDATE_CLOSED"
    )
    result["candidate"]["system"] = "L0"
    result["candidate"]["training_auxiliary"] = auxiliary
    result["candidate"]["representation"] = metadata["representation"]
    result["evidence"]["training_auxiliary_sha256"] = sha256_file(auxiliary_path)
    result["decision"]["next_step"] = (
        "Preregister matched label-ablation, header-ablation, causal-corruption-ablation controls and two paired seeds."
        if initial_pass
        else "Preserve this negative candidate; do not run ablations or seeds, and attribute the remaining measured failure before any successor."
    )
    result["claim_boundary"] = "Development-only label-aware ABI acquisition screen. It cannot certify Phase 3, inherit LayerCake quality or performance, or establish ABI superiority over LoRA or distillation."
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze", "verify"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_LABELED_BPE_CORE_PROTOCOL_V41.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_labeled_bpe_core/development_v41/L0-seed240017")
    parser.add_argument("--fit-dir", default="results/abi_capability_compiler_phase3_labeled_bpe_core/fit_v41/L0-seed240017")
    parser.add_argument("--evaluation-dir", default="results/abi_capability_compiler_phase3_labeled_bpe_core/evaluation_v41/L0-seed240017")
    parser.add_argument("--decision", default="results/abi_capability_compiler_phase3_labeled_bpe_core/labeled_bpe_decision_v41.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = build_decision(
        root=root,
        protocol_path=(root / args.protocol).resolve(),
        candidate_dir=(root / args.candidate_dir).resolve(),
        fit_dir=(root / args.fit_dir).resolve(),
        evaluation_dir=(root / args.evaluation_dir).resolve(),
    )
    decision_path = (root / args.decision).resolve()
    if args.command == "analyze":
        _write_immutable(decision_path, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    elif _json(decision_path) != result:
        raise Phase3Error("stored V41 decision differs from raw-evidence recomputation")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
