"""Continue R91 with length-diverse, structurally complete normalization."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from safetensors.torch import load_file, save_file

from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core
from abi.layercake_full_core_acquisition import _manifest_sha, load_english_training_rows
from abi.layercake_host import _sha256_file
from experiments.joint_span_r88.core_v1 import (
    FEEDFORWARD_WIDTH, HEADS, LAYERS, MAX_SPAN_TOKENS, MAX_TOKENS, WIDTH,
    JointSpanBridge, bridge_parameter_count, joint_span_scores,
)
from experiments.stateful_span_r85.train_candidate_v1 import _batch
from experiments.structural_span_r91.train_v1 import CODE, _gold, _span_example
from .core_v1 import ARCHITECTURE, PACKAGE_FORMAT, validate_metadata


PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
ARTIFACT_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
R91_SHA256 = "04b350e0a7f22ac4facb380f6bf99edfc0022697a312248972491434234a4fba"
ROUTE = CAPABILITY_CAKE_ORDER.index("domain_independent_reasoning")
STEPS, BATCH_SIZE, LEARNING_RATE, SEED = 3_000, 32, 3.0e-4, 96_001
STEM_LENGTHS, DIGIT_LENGTHS, STRUCTURAL_FORMS, INTERFACES, LIST_PLACEMENTS = (2, 3, 4, 5), (4, 5, 6, 7, 8, 9), 32, 8, 4


class R96TrainingError(RuntimeError):
    pass


def _stem(rng: random.Random, length: int) -> str:
    return "".join(chr(65 + rng.randrange(26)) for _ in range(length))


def _relation(form: int, a: str, b: str, c: str, subject: str) -> str:
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
        f"Given {subject} : {a}, and the chain {a} : {b}, {b} : {c}, follow both relations.",
        f"Initial membership for {subject} is {a}. One hop reaches {b}; a second hop reaches {c}.",
        f"{a} objects qualify as {b} objects, and {b} objects qualify as {c} objects. {subject} qualifies as {a}.",
        f"Read {a} <= {b} and {b} <= {c} as containment. The item {subject} is contained by {a}.",
        f"For {subject}, start label = {a}. Apply label ({a}) = {b}, followed by label ({b}) = {c}.",
        f"{subject} bears {a}; bearing {a} entails {b}; bearing {b} entails {c}.",
        f"In a nested hierarchy, {subject} lies in {a}, {a} is a subset of {b}, and {b} is a subset of {c}.",
        f"Membership ledger: {subject} has {a}. The {a} ledger points to {b}. The {b} ledger points to {c}.",
        f"Trace this classification: {subject} begins under {a}; the next broader type is {b}; the type broader than {b} is {c}.",
        f"Record {subject} under {a}. The parent of {a} is {b}; the parent of {b} is {c}.",
        f"Taxonomy entries state {subject} in {a}, then {a} in {b}, then {b} in {c}.",
        f"Starting at {subject}:{a}, follow successor {a}:{b} and successor {b}:{c}.",
        f"The type path for {subject} reads {a}, then {b}, then {c}.",
        f"Inclusions are {subject} within {a}, {a} within {b}, and {b} within {c}.",
        f"Classification table: {subject} => {a}; {a} => {b}; {b} => {c}.",
        f"Assign {subject} to {a}. Promote {a} to {b}, and promote {b} to {c}.",
        f"For this ontology, {subject} occupies {a}; {a} descends from {b}; {b} descends from {c}.",
        f"The chain attached to {subject} is {a} / {b} / {c}, ordered from narrow to broad.",
        f"{subject} matches {a}. Anything matching {a} matches {b}; anything matching {b} matches {c}.",
        f"Resolve the link from {subject} to {a}, then from {a} to {b}, and finally from {b} to {c}.",
        f"Closed type facts: {subject} has {a}; {a} implies {b}; {b} implies {c}.",
    )
    return forms[form]


def _render(tokenizer: Any, row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    source_codes = list(dict.fromkeys(CODE.findall(str(row["prompt"]))))
    if len(source_codes) != 4 or str(row["response"]) != source_codes[2]:
        raise R96TrainingError("source factorization changed")
    length = rng.choice(STEM_LENGTHS)
    stems = []
    while len(stems) < 4:
        value = _stem(rng, length)
        if value not in stems:
            stems.append(value)
    digits = rng.choice(DIGIT_LENGTHS)
    number = rng.randrange(10 ** (digits - 1), 10**digits)
    a, b, c, subject = [f"{stem}{number:0{digits}d}" for stem in stems]
    form, interface, placement = rng.randrange(STRUCTURAL_FORMS), rng.randrange(INTERFACES), rng.randrange(LIST_PLACEMENTS)
    relation = _relation(form, a, b, c, subject)
    candidates = [a, b, c]
    rng.shuffle(candidates)
    choices = "Candidate labels in arbitrary order: " + " | ".join(candidates) + "."
    question = f"Return only the terminal class for {subject} after both links."
    body = (
        f"{relation} {question}" if placement == 0 else
        f"{choices} {relation} {question}" if placement == 1 else
        f"{relation} {choices} {question}" if placement == 2 else
        f"{relation} {question} {choices}"
    )
    wrappers = (
        body,
        f"Solve this self-contained classification task. {body}",
        f"Use only these invented relations. {body}",
        f"Reason from every supplied statement and answer directly. {body}",
        f"A fictional type system follows. {body}",
        f"Closed-world record {rng.randrange(10_000_000)}. {body}",
        f"No outside facts are relevant. {body}",
        f"Complete the full nonce hierarchy. {body}",
    )
    prompt = wrappers[interface]
    transformed = {**row, "record_id": f"{row['record_id']}:{number}:{'-'.join(stems)}:{form}:{interface}:{placement}", "prompt": prompt, "response": c}
    example = _span_example(tokenizer, transformed)
    example.update({
        "base_record_id": str(row["record_id"]), "structural_form": form, "interface": interface,
        "list_placement": placement, "stem_length": length, "digit_length": digits,
        "synthetic_prompt_utf8_bytes": len(prompt.encode()), "synthetic_response_utf8_bytes": len(c.encode()),
    })
    return example


def train(*, artifact: Path, parent: Path, initial: Path, layercake_root: Path, output: Path) -> dict[str, Any]:
    if output.exists() or not torch.cuda.is_available():
        raise R96TrainingError("R96 output exists or CUDA unavailable")
    for path, digest in ((artifact, ARTIFACT_SHA256), (parent / "model.safetensors", PARENT_SHA256), (parent / "metadata.json", PARENT_METADATA_SHA256), (initial / "bridge.safetensors", R91_SHA256)):
        if not path.is_file() or _sha256_file(path) != digest:
            raise R96TrainingError("immutable R96 input changed")
    rows, budget, bundle = load_english_training_rows(artifact, budget_index=-1)
    if len(rows) != 8_129 or bundle["verification"]["training_eligible"] is not True:
        raise R96TrainingError("source coverage changed")
    device = torch.device("cuda")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); random.seed(SEED)
    rng = random.Random(SEED)
    process = psutil.Process(); rss_before = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, parent_metadata = load_layercake_core(parent, layercake_root=layercake_root, device=device)
    model.eval().requires_grad_(False)
    bridge = JointSpanBridge().to(device)
    bridge.load_state_dict(load_file(str(initial / "bridge.safetensors"), device="cuda"), strict=True)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    order = list(range(len(rows))); rng.shuffle(order); cursor = 0
    seen, forms, interfaces, placements, stem_lengths, digit_lengths = set(), set(), set(), set(), set(), set()
    synthetic_examples = prompt_bytes = response_bytes = 0; curves = []
    started = time.perf_counter(); cpu_before = process.cpu_times()
    for step in range(1, STEPS + 1):
        indexes = []
        while len(indexes) < BATCH_SIZE:
            take = min(BATCH_SIZE - len(indexes), len(order) - cursor)
            indexes.extend(order[cursor:cursor + take]); cursor += take
            if cursor == len(order): rng.shuffle(order); cursor = 0
        examples = [_render(tokenizer, rows[index], rng) for index in indexes]
        seen.update(row["base_record_id"] for row in examples); forms.update(row["structural_form"] for row in examples)
        interfaces.update(row["interface"] for row in examples); placements.update(row["list_placement"] for row in examples)
        stem_lengths.update(row["stem_length"] for row in examples); digit_lengths.update(row["digit_length"] for row in examples)
        synthetic_examples += len(examples); prompt_bytes += sum(row["synthetic_prompt_utf8_bytes"] for row in examples); response_bytes += sum(row["synthetic_response_utf8_bytes"] for row in examples)
        ids, attention, _, _ = _batch(examples, pad_token_id=tokenizer.pad_token_id, device=device)
        routes = torch.full((len(examples),), ROUTE, dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model(ids, attention_mask=attention, prompt_lengths=attention.long().sum(dim=1), task_routes=routes, use_cache=False)["hidden"]
        traces = [tuple(getattr(block, "_abi_last_deep_adapter_routes", ())) for block in model.transformer.h]
        if tuple(model.last_cake_calls) != (ROUTE,) or any(trace != (ROUTE,) for trace in traces):
            raise R96TrainingError("sparse parent execution changed")
        starts, ends = bridge(hidden.detach(), attention, routes)
        scores = joint_span_scores(starts, ends, attention).float(); gold = _gold(examples, scores.shape[1], device)
        loss = (torch.logsumexp(scores.flatten(1), dim=-1) - torch.logsumexp(scores.masked_fill(~gold, -torch.inf).flatten(1), dim=-1)).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 100 == 0 or step == STEPS:
            predicted = scores.flatten(1).argmax(dim=-1); accuracy = gold.flatten(1).gather(1, predicted[:, None]).float().mean()
            point = {"step": step, "loss": float(loss.detach()), "batch_joint_span_accuracy": float(accuracy.item()), "wall_seconds": time.perf_counter() - started}
            curves.append(point); print(json.dumps(point), flush=True)
    elapsed = time.perf_counter() - started; cpu_after = process.cpu_times()
    if len(seen) != len(rows) or len(forms) != STRUCTURAL_FORMS or len(interfaces) != INTERFACES or len(placements) != LIST_PLACEMENTS or stem_lengths != set(STEM_LENGTHS) or digit_lengths != set(DIGIT_LENGTHS):
        raise R96TrainingError("normalization coverage failed")
    output.mkdir(parents=True)
    checkpoint = output / "bridge.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in bridge.state_dict().items()}, str(checkpoint))
    metadata = {
        "format": PACKAGE_FORMAT, "status": "TRAINED_FROZEN_DEVELOPMENT_UNSCREENED",
        "bridge": {"architecture": ARCHITECTURE, "width": WIDTH, "layers": LAYERS, "heads": HEADS, "feedforward_width": FEEDFORWARD_WIDTH, "maximum_tokens": MAX_TOKENS, "maximum_span_tokens": MAX_SPAN_TOKENS, "parameter_count": bridge_parameter_count(), "active_parameter_count": bridge_parameter_count(), "contains_prompt_templates": False, "contains_rules": False, "contains_output_lookup_table": False},
        "checkpoint": {"path": checkpoint.name, "sha256": _sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "initialization": {"r91_checkpoint_sha256": R91_SHA256, "r91_parameters_loaded": bridge_parameter_count()},
        "parent_layercake": {"checkpoint_sha256": PARENT_SHA256, "metadata_sha256": PARENT_METADATA_SHA256, "architecture": parent_metadata["architecture"]["architecture_version"], "all_parameters_frozen": True, "changed_on_disk": False},
        "imported_artifact": {"sha256": ARTIFACT_SHA256, "records": len(rows), "budget_id": budget["budget_id"], "teacher_tokens": sum(int(row["teacher_tokens"]) for row in rows), "all_records_seen": True, "artifact_unchanged": True},
        "normalization": {"method": "bijective_length_diverse_structural_paraphrase_augmentation", "structural_forms": STRUCTURAL_FORMS, "neutral_interfaces": INTERFACES, "candidate_list_placements": LIST_PLACEMENTS, "stem_lengths": list(STEM_LENGTHS), "digit_lengths": list(DIGIT_LENGTHS), "synthetic_examples_consumed": synthetic_examples, "synthetic_prompt_utf8_bytes_consumed": prompt_bytes, "synthetic_response_utf8_bytes_consumed": response_bytes, "adds_teacher_claims": False, "synthetic_teacher_tokens": 0, "normalized_text_stored_in_package": False, "r95_exact_prompts_consumed": 0},
        "source_boundary": {"teacher_present_during_training": False, "teacher_present_at_inference": False, "source_parameters_copied": 0, "source_transformer_blocks_retained": 0, "source_logits_stored": 0, "source_hidden_activations_stored": 0, "source_generated_text_in_package": False},
        "training": {"seed": SEED, "device": "cuda", "successful_optimizer_steps": STEPS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": 0.01, "unique_records_seen": len(seen), "wall_seconds": elapsed, "gpu_hours": elapsed / 3600, "cpu_seconds": cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system, "active_parameter_seconds": bridge_parameter_count() * elapsed, "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "rss_before_bytes": rss_before, "rss_after_bytes": process.memory_info().rss, "curves": curves},
        "development": {"r95_outputs_used_for_training": 0, "candidate_outputs_observed": 0, "screen_status": "UNSCREENED"},
        "implementation": {"core_sha256": _sha256_file(Path(__file__).with_name("core_v1.py")), "trainer_sha256": _sha256_file(Path(__file__).resolve()), "protocol_sha256": _sha256_file(Path(__file__).with_name("PROTOCOL.md"))},
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "promotion_eligible": False, "full_abi_moonshot": "OPEN", "claim_boundary": "Unscreened length- and structure-invariant reasoning bridge; not broad English.",
    }
    metadata["manifest_sha256"] = _manifest_sha(metadata)
    metadata_path = output / "metadata.json"; metadata_path.write_bytes((json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    validate_metadata(metadata, output)
    print(json.dumps({"checkpoint_sha256": _sha256_file(checkpoint), "metadata_sha256": _sha256_file(metadata_path), "parameter_count": bridge_parameter_count()}, indent=2, sort_keys=True))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "parent", "initial", "layercake-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    train(artifact=args.artifact.resolve(), parent=args.parent.resolve(), initial=args.initial.resolve(), layercake_root=args.layercake_root.resolve(), output=args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
