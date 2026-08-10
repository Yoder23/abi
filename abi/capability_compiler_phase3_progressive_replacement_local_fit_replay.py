"""Exact replay of local fit with corrected copied-norm key classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error
from . import capability_compiler_phase3_progressive_replacement_local_fit as base


FORMAT = "abi-capability-compiler-phase3-progressive-replacement-local-fit-replay/1"


def corrected_replacement_trainable_keys(model: Any) -> set[str]:
    copied_norms = {
        f"layers.{layer_index}.{name}.weight"
        for layer_index in range(len(model.layers))
        for name in ("input_norm", "post_attention_norm")
    }
    return {
        name for name, _ in model.named_parameters()
        if name.startswith("layers.") and name not in copied_norms
    }


def _repair_loader(root: Path, path: Path):
    repair = json.loads(path.read_text(encoding="utf-8"))
    if repair.get("format") != FORMAT or repair.get("status") != "PREREGISTERED_EXACT_PREFLIGHT_REPLAY":
        raise Phase3Error("progressive local-fit replay governance changed")
    for name, expected in repair["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"progressive local-fit replay binding changed: {name}")
    protocol = json.loads((root / repair["base_protocol"]).read_text(encoding="utf-8"))
    return protocol, sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_PROGRESSIVE_REPLACEMENT_LOCAL_FIT_REPLAY_V227.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_progressive_replacement/local_fit_replay_v228")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    base.replacement_trainable_keys = corrected_replacement_trainable_keys
    base.load_protocol = _repair_loader
    result = base.inventory(root, root / args.protocol) if args.command == "inventory" else base.train(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not str(result["status"]).startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
