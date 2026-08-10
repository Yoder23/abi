"""Unchanged routed-v15 autonomous screen through the certified v16 fp16 host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

from . import capability_compiler_phase3_routed_v15_autonomous_screen_isolated as screen


def execute(root: Path, protocol_path: Path, output: Path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    host = protocol.get("layercake_host", {})
    if (
        host.get("interface") != "lc-direct-neural-core/16"
        or host.get("accepted_artifact_interface") != "lc-direct-neural-core/15"
        or host.get("artifact_tensor_mutation") is not False
        or protocol.get("repair_scope") != "HOST_ONLY_FP16_ROUTER_CONFORMANCE"
        or protocol.get("scientific_fields_changed") is not False
        or "transformers" in sys.modules
    ):
        raise screen.Phase3Error("v16 replay host boundary changed")
    layercake_root = (root / host["repository"]).resolve()
    sys.path.insert(0, str(layercake_root))
    import layercake.routed_sparse_rank768_progressive_core as v15_module
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore

    original = v15_module.RoutedSparseRank768ProgressiveCore
    if not issubclass(PrecisionConformantRoutedSparseRank768ProgressiveCore, original):
        raise screen.Phase3Error("v16 host is not a v15 state-compatible successor")
    v15_module.RoutedSparseRank768ProgressiveCore = PrecisionConformantRoutedSparseRank768ProgressiveCore
    try:
        return screen.execute(root, protocol_path, output)
    finally:
        v15_module.RoutedSparseRank768ProgressiveCore = original


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_AUTONOMOUS_SCREEN_PROTOCOL_V331.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_routed_v15/autonomous_screen_v332")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
