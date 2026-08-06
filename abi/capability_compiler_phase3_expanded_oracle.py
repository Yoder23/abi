"""Expanded-bridge V22 oracle wrapper with exact C0 subspace transplant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch
from torch import nn

from . import capability_compiler_phase3_oracle_fit as oracle
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_sequence_bridge import PromptConditionedSequenceBridge, _make_adapter_hook, _sequence_forward
from .capability_compiler_phase3_shared_output import SharedOutputCake, _shared_dispatch
from .layercake_core_loader import load_layercake_core


SEQUENCE_RANK = 256
OUTPUT_RANK = 128
EXPANDED_TRAINABLE_PARAMETERS = 2_238_982


def install_expanded(model: nn.Module) -> None:
    device = model.transformer.wte.weight.device; dtype = model.transformer.wte.weight.dtype
    model.abi_sequence_bridge = PromptConditionedSequenceBridge(rank=SEQUENCE_RANK).to(device=device, dtype=dtype)
    model.abi_sequence_route_classifier = nn.Linear(SEQUENCE_RANK, 6).to(device=device, dtype=dtype)
    nn.init.zeros_(model.abi_sequence_route_classifier.weight); nn.init.zeros_(model.abi_sequence_route_classifier.bias)
    for index, block in enumerate(model.transformer.h): block.register_forward_pre_hook(_make_adapter_hook(model, index), with_kwargs=True)
    model.abi_shared_output_cake = SharedOutputCake(rank=OUTPUT_RANK).to(device=device, dtype=dtype)
    model._dispatch = MethodType(_shared_dispatch, model); model._abi_sequence_condition = None; model.forward = MethodType(_sequence_forward, model)


def transplant_c0(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    target = model.state_dict()
    # The frozen host must already be identical; copy only registered bridge tensors.
    for name, value in state.items():
        if not name.startswith(("abi_sequence_bridge.", "abi_sequence_route_classifier.", "abi_shared_output_cake.")):
            if name not in target or not torch.equal(target[name].cpu(), value.cpu()): raise Phase3Error(f"expanded oracle frozen host mismatch: {name}")
    with torch.no_grad():
        b = model.abi_sequence_bridge; old = state
        b.prompt_norm.weight.copy_(old["abi_sequence_bridge.prompt_norm.weight"]); b.prompt_norm.bias.copy_(old["abi_sequence_bridge.prompt_norm.bias"])
        b.prompt_projection.weight[:128].copy_(old["abi_sequence_bridge.prompt_projection.weight"])
        b.prompt_output.weight.zero_(); b.prompt_output.weight[:128, :128].copy_(old["abi_sequence_bridge.prompt_output.weight"]); nn.init.normal_(b.prompt_output.weight[128:, 128:], mean=0.0, std=0.02)
        b.route_embedding.weight.zero_(); b.route_embedding.weight[:, :128].copy_(old["abi_sequence_bridge.route_embedding.weight"])
        for index, adapter in enumerate(b.adapters):
            prefix = f"abi_sequence_bridge.adapters.{index}."
            adapter.norm.weight.copy_(old[prefix + "norm.weight"]); adapter.norm.bias.copy_(old[prefix + "norm.bias"])
            adapter.down.weight[:128].copy_(old[prefix + "down.weight"])
            adapter.condition.weight.zero_(); adapter.condition.weight[:128, :128].copy_(old[prefix + "condition.weight"]); nn.init.normal_(adapter.condition.weight[128:, 128:], mean=0.0, std=0.02)
            adapter.up.weight.zero_(); adapter.up.weight[:, :128].copy_(old[prefix + "up.weight"])
        model.abi_sequence_route_classifier.weight.zero_(); model.abi_sequence_route_classifier.weight[:, :128].copy_(old["abi_sequence_route_classifier.weight"]); model.abi_sequence_route_classifier.bias.copy_(old["abi_sequence_route_classifier.bias"])
        cake = model.abi_shared_output_cake; cake.norm.weight.copy_(old["abi_shared_output_cake.norm.weight"]); cake.norm.bias.copy_(old["abi_shared_output_cake.norm.bias"]); cake.down.weight[:64].copy_(old["abi_shared_output_cake.down.weight"]); cake.up.weight.zero_(); cake.up.weight[:, :64].copy_(old["abi_shared_output_cake.up.weight"])


def _load_expanded(root: Path, protocol: Mapping[str, Any], device: torch.device):
    v11 = json.loads((root / protocol["v11_protocol"]).read_text(encoding="utf-8")); parent = (root / v11["host"]["parent_path"]).resolve(); model, tokenizer, _ = load_layercake_core(parent, layercake_root=(root / v11["host"]["layercake_root"]).resolve(), device=device); install_expanded(model); state = load_file(str((root / protocol["starting_candidate"] / "model.safetensors").resolve()), device="cpu"); transplant_c0(model, state); return model, tokenizer, v11


def _load_expanded_candidate(*, root: Path, protocol: Mapping[str, Any], candidate_dir: Path, device: torch.device):
    parent = (root / protocol["host"]["parent_path"]).resolve(); model, tokenizer, _ = load_layercake_core(parent, layercake_root=(root / protocol["host"]["layercake_root"]).resolve(), device=device); install_expanded(model); state = load_file(str(candidate_dir / "model.safetensors"), device=str(device)); model.load_state_dict(state, strict=True); return model.eval(), tokenizer


def _install_delegate() -> None:
    oracle.EXPECTED_TRAINABLE_PARAMETERS = EXPANDED_TRAINABLE_PARAMETERS
    oracle._load = _load_expanded
    oracle.load_candidate = _load_expanded_candidate


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_EXPANDED_ORACLE_PROTOCOL_V22.json"); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("preflight"); tr = sub.add_parser("train"); tr.add_argument("--output-dir", required=True); ev = sub.add_parser("evaluate"); ev.add_argument("--candidate-dir", required=True); ev.add_argument("--output-dir", required=True); args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = (root / args.protocol).resolve(); _install_delegate()
    if args.command == "preflight": result = oracle.preflight(root, protocol)
    elif args.command == "train": result = oracle.train(root, protocol, (root / args.output_dir).resolve())
    else: result = oracle.evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
