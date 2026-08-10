"""Teacher-forced mechanism audit for the failed V451 prompt pointer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_v443_prompt_pointer import _bridge, _load_v443
from .capability_compiler_phase2_common import sha256_file


FORMAT = "abi-capability-compiler-phase3-pointer-mechanism-audit/1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_MECHANISM_AUDIT"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("pointer mechanism audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"pointer mechanism audit binding changed: {name}")
    return protocol, sha256_file(path)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable mechanism audit exists: {output}")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise Phase3Error("pointer mechanism audit CUDA device unavailable")
    parent_protocol = _json(root / protocol["pointer_protocol"])
    model, tokenizer, _ = _load_v443(root, parent_protocol, device)
    bridge = _bridge(device)
    bridge.load_state_dict(load_file(str(root / protocol["pointer_checkpoint"]), device="cuda"), strict=True)
    bridge.eval()

    candidate_rows = {
        str(row["probe_id"]): row
        for row in map(
            json.loads,
            (root / protocol["v443_outputs"]).read_text(encoding="utf-8").splitlines(),
        )
    }
    teacher_rows = {
        str(row["probe_id"]): row
        for row in map(
            json.loads,
            (root / protocol["teacher_outputs"]).read_text(encoding="utf-8").splitlines(),
        )
    }
    probes = {
        str(row["probe_id"]): row
        for row in development_probes(root / protocol["catalog"])
    }
    selected = sorted(
        probe_id
        for probe_id in probes
        if not bool(candidate_rows[probe_id]["functional_pass"])
        and bool(teacher_rows[probe_id]["functional_pass"])
    )
    if len(selected) != int(protocol["population"]["candidate_specific_failures"]):
        raise Phase3Error("pointer mechanism population changed")

    eligible_positions = 0
    pointer_correct = 0
    mixture_correct = 0
    gates: list[float] = []
    rows: list[dict[str, Any]] = []
    for probe_id in selected:
        probe = probes[probe_id]
        prompt_ids = [
            int(value)
            for value in tokenizer.encode(str(probe["prompt"]).rstrip() + "\n", add_special_tokens=False)
        ]
        response_ids = [
            int(value)
            for value in tokenizer.encode(str(teacher_rows[probe_id]["output"]), add_special_tokens=False)
        ] + [int(tokenizer.eos_token_id)]
        maximum = int(model.config.max_tokens)
        response_ids = response_ids[: maximum - len(prompt_ids)]
        sequence = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
        prompt_length = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)
        prefill = model(sequence[:, : len(prompt_ids)], prompt_lengths=prompt_length, use_cache=False)
        route = prefill["task_routes"]
        result = model(sequence, prompt_lengths=prompt_length, task_routes=route, use_cache=False)
        prompt_tensor = sequence[0, : len(prompt_ids)]
        prompt_hidden = result["hidden"][0, : len(prompt_ids)]
        record_eligible = 0
        record_pointer = 0
        record_mixture = 0
        record_gates: list[float] = []
        for offset, target in enumerate(response_ids):
            prediction_index = len(prompt_ids) - 1 + offset
            language_logits = result["logits"][0, prediction_index].float()
            if int(language_logits.argmax()) == int(target) or not bool((prompt_tensor == int(target)).any()):
                continue
            query_hidden = result["hidden"][0, prediction_index]
            attention = F.softmax(
                bridge.pointer_scores(query_hidden[None], prompt_hidden)[0], dim=-1
            )
            pointer_token = int(prompt_tensor[int(attention.argmax())])
            gate = float(bridge.copy_gate(query_hidden[None], route)[0])
            pointer_probability = torch.zeros_like(language_logits)
            pointer_probability.scatter_add_(0, prompt_tensor, attention.float())
            mixture = (1.0 - gate) * F.softmax(language_logits, dim=-1) + gate * pointer_probability
            record_eligible += 1
            record_pointer += int(pointer_token == int(target))
            record_mixture += int(int(mixture.argmax()) == int(target))
            record_gates.append(gate)
        eligible_positions += record_eligible
        pointer_correct += record_pointer
        mixture_correct += record_mixture
        gates.extend(record_gates)
        rows.append(
            {
                "probe_id": probe_id,
                "capability": str(probe["canonical_capability"]),
                "eligible_positions": record_eligible,
                "pointer_correct_tokens": record_pointer,
                "mixture_repaired_tokens": record_mixture,
                "copy_gate_mean": sum(record_gates) / len(record_gates) if record_gates else 0.0,
            }
        )
    pointer_rate = pointer_correct / eligible_positions
    mixture_rate = mixture_correct / eligible_positions
    audit_gates = {
        "population_exact": len(selected) == 81,
        "eligible_positions_nonzero": eligible_positions > 0,
        "pointer_selection_material": pointer_rate >= float(protocol["gates"]["pointer_token_accuracy_minimum"]),
        "mixture_causally_inactive": mixture_rate <= float(protocol["gates"]["mixture_repair_rate_maximum"]),
        "no_training_or_mutation": True,
    }
    result = {
        "format": FORMAT,
        "status": "PASS_SELECTION_WORKS_MIXTURE_INACTIVE" if all(audit_gates.values()) else "FAIL_POINTER_SELECTION_INADEQUATE_OR_MIXTURE_ACTIVE",
        "protocol_sha256": protocol_sha,
        "records": len(selected),
        "eligible_teacher_forced_positions": eligible_positions,
        "pointer_correct_tokens": pointer_correct,
        "pointer_token_accuracy": pointer_rate,
        "mixture_repaired_tokens": mixture_correct,
        "mixture_repair_rate": mixture_rate,
        "copy_gate": {
            "minimum": min(gates) if gates else 0.0,
            "median": median(gates) if gates else 0.0,
            "mean": sum(gates) / len(gates) if gates else 0.0,
            "p95": _quantile(gates, 0.95),
            "maximum": max(gates) if gates else 0.0,
        },
        "per_record": rows,
        "gates": audit_gates,
        "training_performed": False,
        "artifact_mutated": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Teacher-forced read-only development mechanism audit; not an autonomous quality or architecture-promotion result.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=Path("ABI_CAPABILITY_COMPILER_PHASE3_POINTER_MECHANISM_AUDIT_PROTOCOL_V452.json"))
    parser.add_argument("--output", type=Path, default=Path("results/abi_capability_compiler_phase3_pointer_mechanism_audit/audit_v453/result.json"))
    args = parser.parse_args()
    result = run(args.root, args.root / args.protocol, args.root / args.output)
    print(json.dumps({key: result[key] for key in ("status", "eligible_teacher_forced_positions", "pointer_token_accuracy", "mixture_repair_rate", "copy_gate", "gates")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
