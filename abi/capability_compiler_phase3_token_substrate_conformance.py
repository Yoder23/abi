"""Single token-substrate conformance successor from V459."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from . import capability_compiler_phase3_copy_balanced_transition as base
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error


FORMAT = "abi-capability-compiler-phase3-token-substrate-conformance/1"
FROZEN_PREFIXES = ("transformer.wpe.",)


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_SINGLE_BOUNDED_SUCCESSOR" or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("nearby_sweeps_authorized") is not False:
        raise Phase3Error("token substrate governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"token substrate binding changed: {name}")
    return protocol, sha256_file(path)


def configure_trainable(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    trainable = []
    for name, parameter in model.named_parameters():
        selected = not name.startswith(FROZEN_PREFIXES)
        parameter.requires_grad_(selected)
        if selected:
            trainable.append(parameter)
    return trainable


class _PatchedBase:
    def __enter__(self):
        self.values = (base.load_protocol, base._configure_trainable, base.FROZEN_PREFIXES, base.FORMAT)
        base.load_protocol = load_protocol
        base._configure_trainable = configure_trainable
        base.FROZEN_PREFIXES = FROZEN_PREFIXES
        base.FORMAT = FORMAT

    def __exit__(self, *_):
        base.load_protocol, base._configure_trainable, base.FROZEN_PREFIXES, base.FORMAT = self.values


def preflight(root: Path, protocol: Path):
    with _PatchedBase():
        return base.preflight(root, protocol)


def train(root: Path, protocol: Path, output: Path):
    with _PatchedBase():
        return base.train(root, protocol, output)


def evaluate(root: Path, protocol: Path, candidate: Path, output: Path):
    with _PatchedBase():
        return base.evaluate(root, protocol, candidate, output)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_TOKEN_SUBSTRATE_CONFORMANCE_PROTOCOL_V462.json"); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); tp = sub.add_parser("train"); tp.add_argument("--output-dir", required=True); ep = sub.add_parser("evaluate"); ep.add_argument("--candidate-dir", required=True); ep.add_argument("--output-dir", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = root / args.protocol; result = preflight(root, protocol) if args.command == "preflight" else train(root, protocol, root / args.output_dir) if args.command == "train" else evaluate(root, protocol, root / args.candidate_dir, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
