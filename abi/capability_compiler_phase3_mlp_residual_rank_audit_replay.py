"""Exact MLP-rank audit replay with the omitted sequence-bound field restored."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import capability_compiler_phase3_mlp_residual_rank_audit as base
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-mlp-residual-rank-audit-replay/1"


def repaired_protocol(root: Path, path: Path):
    repair = json.loads(path.read_text(encoding="utf-8"))
    if (
        repair.get("format") != FORMAT
        or repair.get("status") != "PREREGISTERED_EXACT_PREFLIGHT_REPLAY"
        or repair.get("scientific_fields_changed") is not False
    ):
        raise Phase3Error("MLP residual rank-audit replay governance changed")
    for name, expected in repair["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"MLP residual rank-audit replay binding changed: {name}")
    protocol = json.loads((root / repair["base_protocol"]).read_text(encoding="utf-8"))
    if "architecture" in protocol:
        raise Phase3Error("base rank-audit protocol unexpectedly changed")
    protocol["architecture"] = dict(repair["architecture_repair"])
    return protocol, sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_MLP_RESIDUAL_RANK_AUDIT_REPLAY_V239.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_dual_path/mlp_residual_rank_audit_replay_v240.json",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error("MLP residual rank-audit replay output exists")
    base.load_protocol = repaired_protocol
    result = base.execute(root, root / args.protocol)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
