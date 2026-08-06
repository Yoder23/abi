"""V18 recent-repeat-only recovery wrapper around the sealed V17 machinery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from . import capability_compiler_phase3_self_prefix as base


def construct_recent_repeat_batch(
    ids: torch.Tensor,
    labels: torch.Tensor,
    policy_logits: torch.Tensor,
    *,
    horizon: int,
    repeat_window: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Corrupt only a wrong prediction that repeats a recent response token."""

    corrupted = ids.clone()
    recovery = torch.full_like(labels, -100)
    predictions = policy_logits[:, :-1].detach().argmax(dim=-1)
    events = 0
    for row in range(ids.shape[0]):
        for source_position in range(labels.shape[1] - 1):
            target_position = source_position + 1
            target = int(labels[row, target_position].item())
            predicted = int(predictions[row, source_position].item())
            if target == -100 or predicted == target:
                continue
            prior = labels[row, max(0, target_position - int(repeat_window)) : target_position]
            if not bool(((prior != -100) & (prior == predicted)).any()):
                continue
            if target_position + 1 >= labels.shape[1] or int(labels[row, target_position + 1].item()) == -100:
                continue
            corrupted[row, target_position] = predicted
            stop = min(labels.shape[1], target_position + 1 + int(horizon))
            recovery[row, target_position + 1 : stop] = labels[row, target_position + 1 : stop]
            events += 1
            break
    return corrupted, recovery, events


def _install() -> None:
    base.construct_self_prefix_batch = construct_recent_repeat_batch


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_REPEAT_PREFIX_PROTOCOL_V18.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    training = sub.add_parser("train"); training.add_argument("--system", choices=base.SYSTEMS, required=True); training.add_argument("--output-dir", required=True)
    evaluation = sub.add_parser("evaluate"); evaluation.add_argument("--candidate-dir", required=True); evaluation.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = (root / args.protocol).resolve(); _install()
    if args.command == "preflight": result = base.preflight(root, protocol)
    elif args.command == "train": result = base.train(root, protocol, args.system, (root / args.output_dir).resolve())
    else: result = base.evaluate(root, protocol, (root / args.candidate_dir).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
