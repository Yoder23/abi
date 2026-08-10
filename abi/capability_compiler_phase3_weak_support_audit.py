"""Attribute V463 weak-capability failures without training or final access."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    set_determinism,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import (
    CAPABILITY_TO_ROUTE,
    Phase3Error,
    _write_immutable,
    load_phase1_ir,
)
from .capability_compiler_phase3_sequence_bridge import _batch, _examples, _generate
from .capability_compiler_phase3_v443_prompt_pointer import _load_v443
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase3-weak-support-audit/1"
WEAK_CAPABILITIES = (
    "abstention",
    "coherence",
    "fluent_realization",
    "tone_control",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_ATTRIBUTION"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or tuple(protocol.get("scope", {}).get("capabilities", ()))
        != WEAK_CAPABILITIES
    ):
        raise Phase3Error("weak-support audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"weak-support binding changed: {relative}")
    return protocol, sha256_file(path)


def _example(
    tokenizer: Any,
    capability: str,
    record_id: str,
    prompt: str,
    output: str,
    max_tokens: int,
) -> dict[str, Any]:
    eos = int(tokenizer.eos_token_id)
    prompt_ids = [
        int(value)
        for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)
    ]
    response_ids = [
        int(value) for value in tokenizer.encode(output, add_special_tokens=False)
    ] + [eos]
    available = max_tokens - len(prompt_ids)
    if available < 2:
        raise Phase3Error("audit prompt exceeds context")
    response_ids = response_ids[:available]
    if response_ids[-1] != eos:
        response_ids[-1] = eos
    return {
        "record_id": record_id,
        "capability": capability,
        "route": CAPABILITY_TO_ROUTE[capability],
        "input_ids": prompt_ids + response_ids,
        "labels": [-100] * len(prompt_ids) + response_ids,
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
    }


@torch.inference_mode()
def _teacher_forced(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    totals: dict[str, Counter] = defaultdict(Counter)
    for start in range(0, len(examples), batch_size):
        selected = examples[start : start + batch_size]
        ids, labels, attention, prompt_lengths, routes = _batch(
            selected, int(tokenizer.eos_token_id), device
        )
        result = model(
            ids,
            attention_mask=attention,
            prompt_lengths=prompt_lengths,
            task_routes=routes,
            use_cache=False,
        )
        logits = result["logits"][:, :-1].float()
        targets = labels[:, 1:]
        mask = targets.ge(0)
        losses = F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten(),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(targets)
        predicted = logits.argmax(dim=-1)
        for index, row in enumerate(selected):
            active = mask[index]
            name = str(row["capability"])
            totals[name]["tokens"] += int(active.sum())
            totals[name]["correct"] += int(predicted[index][active].eq(targets[index][active]).sum())
            totals[name]["nll_micros"] += int(float(losses[index][active].sum()) * 1_000_000)
        print(json.dumps({"teacher_forced": min(start + batch_size, len(examples))}), flush=True)
    result = {}
    for name in WEAK_CAPABILITIES:
        value = totals[name]
        result[name] = {
            "tokens": value["tokens"],
            "top1_correct": value["correct"],
            "top1_accuracy": value["correct"] / value["tokens"],
            "mean_nll": value["nll_micros"] / 1_000_000 / value["tokens"],
        }
    return result


def _ngrams(values: Sequence[int], width: int):
    for index in range(max(0, len(values) - width + 1)):
        yield tuple(values[index : index + width])


def _coverage(
    tokenizer: Any,
    acquisition: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("prompt", "output"):
        result[field] = {}
        for width in (1, 2, 3, 4):
            known = {name: set() for name in WEAK_CAPABILITIES}
            for row in acquisition:
                values = tokenizer.encode(str(row[field]), add_special_tokens=False)
                known[str(row["capability"])].update(_ngrams(values, width))
            per = {}
            for name in WEAK_CAPABILITIES:
                total = 0
                seen = 0
                for row in heldout:
                    if row["capability"] != name:
                        continue
                    values = tokenizer.encode(str(row[field]), add_special_tokens=False)
                    for gram in _ngrams(values, width):
                        total += 1
                        seen += int(gram in known[name])
                per[name] = {
                    "total": total,
                    "seen": seen,
                    "coverage": seen / total if total else 1.0,
                }
            result[field][str(width)] = per
    return result


def _family(record_id: str) -> str:
    match = re.search(r"-(\d+)-v\d+$", record_id)
    if match is None:
        raise Phase3Error(f"development family cannot be derived: {record_id}")
    return f"index_mod4_{int(match.group(1)) % 4}"


def _family_report(
    probes: Sequence[Mapping[str, Any]],
    teacher: Mapping[str, Mapping[str, Any]],
    parent: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for probe in probes:
        capability = str(probe["canonical_capability"])
        if capability not in WEAK_CAPABILITIES:
            continue
        probe_id = str(probe["probe_id"])
        family = _family(probe_id)
        evaluator = probe["evaluator"]
        teacher_output = str(teacher[probe_id]["output"])
        parent_output = str(parent[probe_id]["output"])
        item = values[capability][family]
        item["observations"] += 1
        item["teacher_v1"] += int(evaluate_functional(teacher_output, evaluator))
        item["teacher_v2"] += int(evaluate_functional_v2(teacher_output, evaluator, capability))
        item["parent_v1"] += int(evaluate_functional(parent_output, evaluator))
        item["parent_v2"] += int(evaluate_functional_v2(parent_output, evaluator, capability))
    return {
        capability: {family: dict(counts) for family, counts in sorted(families.items())}
        for capability, families in sorted(values.items())
    }


@torch.inference_mode()
def _autonomous_replay(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    per_capability: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = []
    for name in WEAK_CAPABILITIES:
        ordered = sorted(
            (row for row in rows if row["capability"] == name),
            key=lambda row: str(row["ir_record_id"]),
        )
        selected.extend(ordered[:per_capability])
    details = []
    for index, row in enumerate(selected):
        expected = str(row["normalized_output"])
        maximum = min(256, max(32, len(tokenizer.encode(expected, add_special_tokens=False)) + 16))
        output, tokens, route = _generate(
            model,
            tokenizer,
            str(row["normalized_generation_prompt"]),
            maximum,
            device,
        )
        details.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "expected_sha256": hashlib.sha256(expected.encode()).hexdigest(),
                "exact_response_bytes": output == expected,
                "sequence_similarity": SequenceMatcher(None, output, expected).ratio(),
                "generated_tokens": len(tokens),
                "automatic_task_route": route,
                "task_route_correct": route == CAPABILITY_TO_ROUTE[str(row["capability"])],
                "repetition_collapse_v2": repetition_collapse_v2(output),
            }
        )
        print(json.dumps({"autonomous_replay": index + 1}), flush=True)
    summary = {}
    for name in WEAK_CAPABILITIES:
        values = [row for row in details if row["capability"] == name]
        summary[name] = {
            "observations": len(values),
            "exact_response_bytes": sum(row["exact_response_bytes"] for row in values),
            "mean_sequence_similarity": sum(row["sequence_similarity"] for row in values) / len(values),
            "route_correct": sum(row["task_route_correct"] for row in values),
            "repetition_collapses_v2": sum(row["repetition_collapse_v2"] for row in values),
        }
    return summary, details


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable weak-support output exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("weak-support audit CUDA unavailable")
    set_determinism(int(protocol["execution"]["seed"]))
    device = torch.device("cuda")
    model, tokenizer, _ = _load_v443(root, protocol, device)
    model.eval()
    acquisition_rows = load_phase1_ir(root / protocol["phase1_ir"]["path"])
    acquisition_examples = _examples(
        acquisition_rows,
        tokenizer,
        system="A0",
        seed=int(protocol["execution"]["seed"]),
        max_tokens=int(protocol["execution"]["max_tokens"]),
    )
    acquisition_examples = [
        row for row in acquisition_examples if row["capability"] in WEAK_CAPABILITIES
    ]
    probes = development_probes(root / protocol["development"]["catalog_path"])
    teacher = {
        str(row["probe_id"]): row
        for row in map(
            json.loads,
            (root / protocol["development"]["teacher_reference"]).open(encoding="utf-8"),
        )
    }
    parent = {
        str(row["probe_id"]): row
        for row in map(
            json.loads,
            (root / protocol["parent"]["development_outputs"]).open(encoding="utf-8"),
        )
    }
    development_examples = [
        _example(
            tokenizer,
            str(probe["canonical_capability"]),
            str(probe["probe_id"]),
            str(probe["prompt"]),
            str(teacher[str(probe["probe_id"])]["output"]),
            int(protocol["execution"]["max_tokens"]),
        )
        for probe in probes
        if str(probe["canonical_capability"]) in WEAK_CAPABILITIES
    ]
    training_tf = _teacher_forced(
        model,
        tokenizer,
        acquisition_examples,
        device,
        int(protocol["execution"]["teacher_forced_batch_size"]),
    )
    development_tf = _teacher_forced(
        model,
        tokenizer,
        development_examples,
        device,
        int(protocol["execution"]["teacher_forced_batch_size"]),
    )
    replay, replay_rows = _autonomous_replay(
        model,
        tokenizer,
        acquisition_rows,
        int(protocol["execution"]["autonomous_replay_per_capability"]),
        device,
    )
    acquisition_support = [
        {
            "capability": str(row["capability"]),
            "prompt": str(row["normalized_generation_prompt"]),
            "output": str(row["normalized_output"]),
        }
        for row in acquisition_rows
        if row["capability"] in WEAK_CAPABILITIES
    ]
    heldout_support = [
        {
            "capability": str(probe["canonical_capability"]),
            "prompt": str(probe["prompt"]),
            "output": str(teacher[str(probe["probe_id"])]["output"]),
        }
        for probe in probes
        if str(probe["canonical_capability"]) in WEAK_CAPABILITIES
    ]
    coverage = _coverage(tokenizer, acquisition_support, heldout_support)
    minimum_training_top1 = min(value["top1_accuracy"] for value in training_tf.values())
    minimum_development_top1 = min(value["top1_accuracy"] for value in development_tf.values())
    mean_replay_exact = sum(value["exact_response_bytes"] for value in replay.values()) / sum(value["observations"] for value in replay.values())
    thresholds = protocol["attribution_thresholds"]
    attribution = {
        "acquisition_teacher_forced_fit_high": minimum_training_top1 >= float(thresholds["training_top1_minimum"]),
        "development_teacher_forced_gap_material": minimum_training_top1 - minimum_development_top1 >= float(thresholds["training_development_gap_minimum"]),
        "autonomous_replay_gap_material": minimum_training_top1 >= float(thresholds["training_top1_minimum"]) and mean_replay_exact < float(thresholds["autonomous_exact_minimum"]),
    }
    output.mkdir(parents=True)
    raw = output / "autonomous_replay.jsonl"
    raw.write_bytes(b"".join(canonical_json_bytes(row) for row in replay_rows))
    result = {
        "format": FORMAT,
        "status": "PASS_READ_ONLY_ATTRIBUTION",
        "protocol_sha256": protocol_sha,
        "records": {
            "acquisition_weak": len(acquisition_examples),
            "development_weak": len(development_examples),
            "autonomous_replay": len(replay_rows),
        },
        "training_teacher_forced": training_tf,
        "development_teacher_forced": development_tf,
        "autonomous_training_replay": replay,
        "token_ngram_coverage": coverage,
        "development_family_outcomes": _family_report(probes, teacher, parent),
        "headline": {
            "minimum_training_top1": minimum_training_top1,
            "minimum_development_top1": minimum_development_top1,
            "mean_autonomous_exact_replay": mean_replay_exact,
        },
        "attribution": attribution,
        "raw_replay_sha256": sha256_file(raw),
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_WEAK_SUPPORT_AUDIT_PROTOCOL_V467.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_weak_support_audit/audit_v468",
    )
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "headline": result["headline"],
        "attribution": result["attribution"],
        "evidence_sha256": result["evidence_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
