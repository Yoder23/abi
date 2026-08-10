"""Read-only oracle for exact prompt-token vocabulary selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_v443_prompt_pointer import _load_v443


FORMAT = "abi-capability-compiler-phase3-prompt-vocabulary-oracle/1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_within_prompt(logits: torch.Tensor, prompt_tokens: torch.Tensor, target: int) -> int:
    unique = torch.unique(prompt_tokens)
    order = unique[logits.index_select(0, unique).argsort(descending=True)]
    match = torch.nonzero(order == int(target), as_tuple=False).flatten()
    if not match.numel():
        raise ValueError("target is absent from prompt vocabulary")
    return int(match[0].item()) + 1


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_CAPACITY_ORACLE"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("prompt vocabulary oracle governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"prompt vocabulary oracle binding changed: {name}")
    return protocol, sha256_file(path)


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable prompt vocabulary oracle exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("prompt vocabulary oracle CUDA device unavailable")
    device = torch.device("cuda")
    parent_protocol = _json(root / protocol["parent_protocol"])
    model, tokenizer, _ = _load_v443(root, parent_protocol, device)
    candidate = {
        str(row["probe_id"]): row
        for row in map(json.loads, (root / protocol["v443_outputs"]).read_text(encoding="utf-8").splitlines())
    }
    teacher = {
        str(row["probe_id"]): row
        for row in map(json.loads, (root / protocol["teacher_outputs"]).read_text(encoding="utf-8").splitlines())
    }
    probes = {str(row["probe_id"]): row for row in development_probes(root / protocol["catalog"])}
    selected = sorted(probe_id for probe_id in probes if not candidate[probe_id]["functional_pass"] and teacher[probe_id]["functional_pass"])
    if len(selected) != 81:
        raise Phase3Error("prompt vocabulary population changed")

    ranks: list[int] = []
    per_capability: Counter[str] = Counter()
    per_capability_top1: Counter[str] = Counter()
    for probe_id in selected:
        probe = probes[probe_id]
        prompt_ids = [int(value) for value in tokenizer.encode(str(probe["prompt"]).rstrip() + "\n", add_special_tokens=False)]
        response_ids = [int(value) for value in tokenizer.encode(str(teacher[probe_id]["output"]), add_special_tokens=False)] + [int(tokenizer.eos_token_id)]
        response_ids = response_ids[: int(model.config.max_tokens) - len(prompt_ids)]
        sequence = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
        prompt_length = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)
        prefill = model(sequence[:, : len(prompt_ids)], prompt_lengths=prompt_length, use_cache=False)
        result = model(sequence, prompt_lengths=prompt_length, task_routes=prefill["task_routes"], use_cache=False)
        prompt_tensor = sequence[0, : len(prompt_ids)]
        capability = str(probe["canonical_capability"])
        for offset, target in enumerate(response_ids):
            prediction_index = len(prompt_ids) - 1 + offset
            logits = result["logits"][0, prediction_index].float()
            if int(logits.argmax()) == int(target) or not bool((prompt_tensor == int(target)).any()):
                continue
            rank = _rank_within_prompt(logits, prompt_tensor, int(target))
            ranks.append(rank)
            per_capability[capability] += 1
            per_capability_top1[capability] += int(rank == 1)
    total = len(ranks)
    top1 = sum(rank <= 1 for rank in ranks) / total
    top3 = sum(rank <= 3 for rank in ranks) / total
    top5 = sum(rank <= 5 for rank in ranks) / total
    gates = {
        "population_exact": len(selected) == 81,
        "eligible_positions_exact": total == int(protocol["population"]["expected_eligible_positions"]),
        "prompt_vocabulary_top1_material": top1 >= float(protocol["gates"]["top1_minimum"]),
        "no_training_or_mutation": True,
    }
    result = {
        "format": FORMAT,
        "status": "PASS_PROMPT_VOCABULARY_MATERIAL_CAPACITY" if all(gates.values()) else "FAIL_PROMPT_VOCABULARY_INSUFFICIENT",
        "protocol_sha256": protocol_sha,
        "records": len(selected),
        "eligible_positions": total,
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "top5_accuracy": top5,
        "median_rank": sorted(ranks)[len(ranks) // 2],
        "per_capability": {
            capability: {
                "eligible_positions": count,
                "top1_correct": per_capability_top1[capability],
                "top1_accuracy": per_capability_top1[capability] / count,
            }
            for capability, count in sorted(per_capability.items())
        },
        "gates": gates,
        "training_performed": False,
        "artifact_mutated": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Teacher-forced development capacity oracle with an oracle copy-position decision; not an autonomous generation or promotion result.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=Path("ABI_CAPABILITY_COMPILER_PHASE3_PROMPT_VOCABULARY_ORACLE_PROTOCOL_V454.json"))
    parser.add_argument("--output", type=Path, default=Path("results/abi_capability_compiler_phase3_prompt_vocabulary_oracle/oracle_v455/result.json"))
    args = parser.parse_args()
    result = run(args.root, args.root / args.protocol, args.root / args.output)
    print(json.dumps({key: result[key] for key in ("status", "eligible_positions", "top1_accuracy", "top3_accuracy", "top5_accuracy", "median_rank", "gates")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
