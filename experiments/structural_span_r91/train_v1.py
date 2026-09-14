"""Train R91 from immutable teacher labels with structural normalization."""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from safetensors.torch import save_file

from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core
from abi.layercake_full_core_acquisition import _manifest_sha, load_english_training_rows
from abi.layercake_host import _sha256_file
from experiments.joint_span_r88.core_v1 import (
    FEEDFORWARD_WIDTH, HEADS, LAYERS, MAX_SPAN_TOKENS, MAX_TOKENS, WIDTH,
    JointSpanBridge, bridge_parameter_count, joint_span_scores,
)
from experiments.stateful_span_r85.train_candidate_v1 import _batch, _span_example
from .core_v1 import ARCHITECTURE, PACKAGE_FORMAT, validate_metadata


PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
ARTIFACT_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
ROUTE = CAPABILITY_CAKE_ORDER.index("domain_independent_reasoning")
CODE = re.compile(r"\b[A-Z]{3}\d{6}\b")
STRUCTURAL_FORMS = 16
INTERFACES = 6
LIST_PLACEMENTS = 4


class R91TrainingError(RuntimeError):
    pass


def _stem(value: int) -> str:
    return "".join(chr(65 + (value // divisor) % 26) for divisor in (26**2, 26, 1))


def _transition_text(form: int, a: str, b: str, c: str, subject: str) -> str:
    forms = (
        f"Every {a} is a {b}. Every {b} is a {c}. {subject} is a {a}.",
        f"All {a} belong to {b}; all {b} belong to {c}; {subject} belongs to {a}.",
        f"If something is {a}, it is {b}. If it is {b}, it is {c}. {subject} is {a}.",
        f"The {a} group is inside {b}, and {b} is inside {c}. {subject} is in {a}.",
        f"Rule one maps {a} to {b}; rule two maps {b} to {c}; start {subject} at {a}.",
        f"Begin with {subject} in {a}. Membership in {a} implies {b}, while {b} implies {c}.",
        f"{subject} has class {a}. Class {a} flows into {b}, and class {b} flows into {c}.",
        f"From {a} proceed to {b}; from {b} proceed to {c}. The initial class of {subject} is {a}.",
        f"Two links are supplied: {a} -> {b} and {b} -> {c}. {subject} starts at {a}.",
        f"Place {subject} in {a}; include each {a} within {b}; include each {b} within {c}.",
        f"The first relation sends {a} into {b}. The next sends {b} into {c}. {subject} begins in {a}.",
        f"Given {subject}:{a}, and the chain {a}:{b}, {b}:{c}, follow both relations.",
        f"Initial membership for {subject} is {a}. One hop reaches {b}; a second hop reaches {c}.",
        f"{a} objects qualify as {b} objects, and {b} objects qualify as {c} objects. {subject} qualifies as {a}.",
        f"Read {a} <= {b} and {b} <= {c} as containment. The item {subject} is contained by {a}.",
        f"For {subject}, start label={a}. Apply label({a})={b}, followed by label({b})={c}.",
    )
    return forms[form]


def _render(tokenizer: Any, row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    source_codes = list(dict.fromkeys(CODE.findall(str(row["prompt"]))))
    if len(source_codes) != 4 or str(row["response"]) != source_codes[2]:
        raise R91TrainingError("source record does not have the audited four-code chain")
    stems: list[str] = []
    while len(stems) < 4:
        value = _stem(rng.randrange(26**3))
        if value not in stems:
            stems.append(value)
    number = rng.randrange(100_000, 999_999)
    a, b, c, subject = [f"{stem}{number:06d}" for stem in stems]
    form = rng.randrange(STRUCTURAL_FORMS)
    interface = rng.randrange(INTERFACES)
    placement = rng.randrange(LIST_PLACEMENTS)
    relation = _transition_text(form, a, b, c, subject)
    candidates = [a, b, c]
    rng.shuffle(candidates)
    candidate_text = "Candidate labels: " + " | ".join(candidates) + "."
    question = (
        f"Determine the class reached by {subject} after both supplied relations. "
        "Return only that class label."
    )
    if placement == 0:
        body = f"{relation} {question}"
    elif placement == 1:
        body = f"{candidate_text} {relation} {question}"
    elif placement == 2:
        body = f"{relation} {candidate_text} {question}"
    else:
        body = f"{relation} {question} {candidate_text}"
    wrappers = (
        body,
        f"Solve this self-contained classification task: {body}",
        f"Use only the following invented relations. {body}",
        f"Please reason from the supplied statements and answer directly. {body}",
        f"A fictional system is described below. {body}",
        f"Input record {rng.randrange(1_000_000, 9_999_999)}. {body}",
    )
    prompt = wrappers[interface]
    transformed = {
        **row,
        "record_id": f"{row['record_id']}:{number}:{'-'.join(stems)}:{form}:{interface}:{placement}",
        "prompt": prompt,
        "response": c,
    }
    example = _span_example(tokenizer, transformed)
    example.update({
        "base_record_id": str(row["record_id"]),
        "structural_form": form,
        "interface": interface,
        "list_placement": placement,
        "synthetic_prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "synthetic_response_utf8_bytes": len(c.encode("utf-8")),
    })
    return example


def _gold(examples: list[dict[str, Any]], tokens: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(len(examples), tokens, tokens, dtype=torch.bool, device=device)
    for index, row in enumerate(examples):
        length = int(row["span_length"])
        for start in row["valid_starts"]:
            mask[index, int(start), int(start) + length - 1] = True
    if not mask.flatten(1).any(dim=1).all():
        raise R91TrainingError("batch contains no gold span")
    return mask


def train(*, artifact: Path, parent: Path, layercake_root: Path, output: Path) -> dict[str, Any]:
    seed, steps, batch_size, learning_rate = 91_001, 4_000, 32, 1.0e-3
    if output.exists() or not torch.cuda.is_available():
        raise R91TrainingError("R91 output exists or CUDA is unavailable")
    for path, digest in (
        (artifact, ARTIFACT_SHA256),
        (parent / "model.safetensors", PARENT_SHA256),
        (parent / "metadata.json", PARENT_METADATA_SHA256),
    ):
        if not path.is_file() or _sha256_file(path) != digest:
            raise R91TrainingError("R91 immutable input changed")
    rows, budget, bundle = load_english_training_rows(artifact, budget_index=-1)
    if len(rows) != 8_129 or bundle["verification"]["training_eligible"] is not True:
        raise R91TrainingError("R91 source coverage changed")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    rng = random.Random(seed)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats(device)
    parent_before = _sha256_file(parent / "model.safetensors")
    model, tokenizer, parent_metadata = load_layercake_core(
        parent, layercake_root=layercake_root, device=device
    )
    model.eval().requires_grad_(False)
    # Audit the assumed information factorization over every teacher record.
    for row in rows:
        codes = list(dict.fromkeys(CODE.findall(str(row["prompt"]))))
        if len(codes) != 4 or str(row["response"]) != codes[2]:
            raise R91TrainingError("R91 source factorization audit failed")
    bridge = JointSpanBridge().to(device)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=learning_rate, weight_decay=0.01)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cursor = 0
    unique_seen: set[str] = set()
    forms_seen, interfaces_seen, placements_seen = set(), set(), set()
    synthetic_examples = prompt_bytes = response_bytes = 0
    curves = []
    started = time.perf_counter()
    cpu_before = process.cpu_times()
    for step in range(1, steps + 1):
        indexes = []
        while len(indexes) < batch_size:
            take = min(batch_size - len(indexes), len(order) - cursor)
            indexes.extend(order[cursor:cursor + take])
            cursor += take
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
        examples = [_render(tokenizer, rows[index], rng) for index in indexes]
        unique_seen.update(row["base_record_id"] for row in examples)
        forms_seen.update(row["structural_form"] for row in examples)
        interfaces_seen.update(row["interface"] for row in examples)
        placements_seen.update(row["list_placement"] for row in examples)
        synthetic_examples += len(examples)
        prompt_bytes += sum(row["synthetic_prompt_utf8_bytes"] for row in examples)
        response_bytes += sum(row["synthetic_response_utf8_bytes"] for row in examples)
        ids, attention, _, _ = _batch(
            examples, pad_token_id=tokenizer.pad_token_id, device=device
        )
        routes = torch.full((len(examples),), ROUTE, dtype=torch.long, device=device)
        with torch.no_grad():
            result = model(
                ids, attention_mask=attention,
                prompt_lengths=attention.long().sum(dim=1),
                task_routes=routes, use_cache=False,
            )
        traces = [
            tuple(getattr(block, "_abi_last_deep_adapter_routes", ()))
            for block in model.transformer.h
        ]
        if tuple(model.last_cake_calls) != (ROUTE,) or any(
            trace != (ROUTE,) for trace in traces
        ):
            raise R91TrainingError("R91 parent sparse execution changed")
        starts, ends = bridge(result["hidden"].detach(), attention, routes)
        scores = joint_span_scores(starts, ends, attention).float()
        gold = _gold(examples, scores.shape[1], device)
        loss = (
            torch.logsumexp(scores.flatten(1), dim=-1)
            - torch.logsumexp(scores.masked_fill(~gold, -torch.inf).flatten(1), dim=-1)
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            predicted = scores.flatten(1).argmax(dim=-1)
            accuracy = gold.flatten(1).gather(1, predicted[:, None]).float().mean()
            point = {
                "step": step, "loss": float(loss.detach().item()),
                "batch_joint_span_accuracy": float(accuracy.item()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(point)
            print(json.dumps(point), flush=True)
    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    if (
        len(unique_seen) != len(rows)
        or len(forms_seen) != STRUCTURAL_FORMS
        or len(interfaces_seen) != INTERFACES
        or len(placements_seen) != LIST_PLACEMENTS
        or _sha256_file(parent / "model.safetensors") != parent_before
        or _sha256_file(artifact) != ARTIFACT_SHA256
    ):
        raise R91TrainingError("R91 coverage or frozen-artifact invariant failed")
    output.mkdir(parents=True)
    checkpoint = output / "bridge.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in bridge.state_dict().items()},
        str(checkpoint),
    )
    metadata = {
        "format": PACKAGE_FORMAT,
        "status": "TRAINED_FROZEN_DEVELOPMENT_UNSCREENED",
        "bridge": {
            "architecture": ARCHITECTURE,
            "width": WIDTH, "layers": LAYERS, "heads": HEADS,
            "feedforward_width": FEEDFORWARD_WIDTH,
            "maximum_tokens": MAX_TOKENS,
            "maximum_span_tokens": MAX_SPAN_TOKENS,
            "parameter_count": bridge_parameter_count(),
            "active_parameter_count": bridge_parameter_count(),
            "contains_prompt_templates": False,
            "contains_rules": False,
            "contains_output_lookup_table": False,
        },
        "checkpoint": {
            "path": checkpoint.name, "sha256": _sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "parent_layercake": {
            "checkpoint_sha256": PARENT_SHA256,
            "metadata_sha256": PARENT_METADATA_SHA256,
            "architecture": parent_metadata["architecture"]["architecture_version"],
            "all_parameters_frozen": True, "changed_on_disk": False,
        },
        "imported_artifact": {
            "sha256": ARTIFACT_SHA256, "records": len(rows),
            "budget_id": budget["budget_id"],
            "teacher_tokens": sum(int(row["teacher_tokens"]) for row in rows),
            "all_records_seen": True, "artifact_unchanged": True,
        },
        "normalization": {
            "method": "bijective_lexical_and_meaning_preserving_structural_variation",
            "structural_forms": STRUCTURAL_FORMS,
            "neutral_interfaces": INTERFACES,
            "candidate_list_placements": LIST_PLACEMENTS,
            "synthetic_examples_consumed": synthetic_examples,
            "synthetic_prompt_utf8_bytes_consumed": prompt_bytes,
            "synthetic_response_utf8_bytes_consumed": response_bytes,
            "adds_teacher_claims": False, "synthetic_teacher_tokens": 0,
            "normalized_text_stored_in_package": False,
        },
        "source_boundary": {
            "teacher_present_during_training": False,
            "teacher_present_at_inference": False,
            "source_parameters_copied": 0,
            "source_transformer_blocks_retained": 0,
            "source_logits_stored": 0,
            "source_hidden_activations_stored": 0,
            "source_generated_text_in_package": False,
        },
        "training": {
            "seed": seed, "device": "cuda", "successful_optimizer_steps": steps,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "weight_decay": 0.01, "unique_records_seen": len(unique_seen),
            "wall_seconds": elapsed, "gpu_hours": elapsed / 3600,
            "cpu_seconds": cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
            "active_parameter_seconds": bridge_parameter_count() * elapsed,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "rss_before_bytes": rss_before, "rss_after_bytes": process.memory_info().rss,
            "curves": curves,
        },
        "development": {
            "r60_outputs_used_for_training": 0,
            "candidate_outputs_observed": 0,
            "screen_status": "UNSCREENED",
        },
        "implementation": {
            "core_sha256": _sha256_file(Path(__file__).with_name("core_v1.py")),
            "trainer_sha256": _sha256_file(Path(__file__).resolve()),
            "protocol_sha256": _sha256_file(Path(__file__).with_name("PROTOCOL.md")),
        },
        "hardware": {
            "machine": platform.node(), "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": "Unscreened structure-invariant reasoning bridge; not broad English.",
    }
    metadata["manifest_sha256"] = _manifest_sha(metadata)
    metadata_path = output / "metadata.json"
    metadata_path.write_bytes(
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    validate_metadata(metadata, output)
    print(json.dumps({
        "checkpoint_sha256": _sha256_file(checkpoint),
        "metadata_sha256": _sha256_file(metadata_path),
        "parameter_count": bridge_parameter_count(),
    }, indent=2, sort_keys=True))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--layercake-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train(
        artifact=args.artifact.resolve(), parent=args.parent.resolve(),
        layercake_root=args.layercake_root.resolve(), output=args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
