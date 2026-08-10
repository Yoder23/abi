"""FP64 factorization replay of the fixed within-head V/O oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import capability_compiler_phase3_within_head_vo_factorization as original
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


RESULT_FORMAT = "abi-capability-compiler-phase3-within-head-vo-stable-replay/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    numeric = protocol.get("numeric_factorization", {})
    if (
        protocol.get("format") != original.FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_WITHIN_HEAD_VO_FACTORIZATION"
        or numeric.get("decomposition_precision") != "fp64"
        or numeric.get("runtime_factor_precision") != "fp32"
        or protocol.get("rank_schedule_sweep_authorized") is not False
    ):
        raise Phase3Error("stable within-head V/O governance changed")
    if output.exists():
        raise Phase3Error("stable within-head V/O output exists")

    prior_qr = original.torch.linalg.qr
    prior_svd = original.torch.linalg.svd
    prior_linear = original.F.linear

    def qr64(values, *args, **kwargs):
        return prior_qr(values.double(), *args, **kwargs)

    def svd64(values, *args, **kwargs):
        return prior_svd(values.double(), *args, **kwargs)

    def runtime_linear(values, weight, bias=None):
        if weight.dtype == torch.float64:
            weight = weight.float()
            if bias is not None:
                bias = bias.float()
        return prior_linear(values, weight, bias)

    original.torch.linalg.qr = qr64
    original.torch.linalg.svd = svd64
    original.F.linear = runtime_linear
    try:
        source_result = original.execute(root, protocol_path, output / "numeric_replay")
    finally:
        original.torch.linalg.qr = prior_qr
        original.torch.linalg.svd = prior_svd
        original.F.linear = prior_linear

    source_path = output / "numeric_replay" / "result.json"
    result = dict(source_result)
    result.update(
        {
            "format": RESULT_FORMAT,
            "status": "PASS_STABLE_WITHIN_HEAD_VO_FACTORIZATION"
            if source_result["passed"]
            else "FAIL_STABLE_WITHIN_HEAD_VO_FACTORIZATION",
            "numeric_factorization": numeric,
            "source_numeric_replay": "numeric_replay/result.json",
            "source_numeric_replay_sha256": sha256_file(source_path),
            "source_numeric_evidence_sha256": source_result["evidence_sha256"],
            "phase3_certified": False,
            "claim_boundary": "FP64 numerical replay of the fixed within-head V/O factorization with fp32 runtime factors only; no factors installed or promoted and no physical runtime, model, Phase 3, or superiority claim is made.",
        }
    )
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_WITHIN_HEAD_VO_STABLE_REPLAY_PROTOCOL_V416.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/within_head_vo_stable_v417",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
