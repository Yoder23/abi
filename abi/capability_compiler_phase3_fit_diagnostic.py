"""Read-only V26 fit, generalization, and autonomous-drift attribution."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, repetition_collapse, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_direct_core import _json, _layercake_api
from .capability_compiler_phase3_pointer_core import _copy_lexemes


FORMAT = "abi-capability-compiler-phase3-fit-diagnostic/1"
REPAIR_FORMAT = "abi-capability-compiler-phase3-fit-diagnostic-runtime-repair/1"
SYSTEMS = ("V23", "V24")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    document = _json(path)
    if document.get("format") == REPAIR_FORMAT:
        if (
            document.get("status") != "AUTHORIZED_IMPLEMENTATION_RETRY"
            or document.get("scientific_design_changed") is not False
            or document.get("attempt") != 2
        ):
            raise Phase3Error("fit diagnostic repair governance changed")
        base_path = (root / document["base_protocol"]).resolve()
        if not base_path.is_file() or sha256_file(base_path) != document["base_protocol_sha256"]:
            raise Phase3Error("fit diagnostic base protocol changed")
        protocol = _json(base_path)
        bindings = dict(protocol.get("bindings", {}))
        for relative, expected in document.get("binding_overrides", {}).items():
            if relative not in bindings:
                raise Phase3Error(f"repair attempted a new binding override: {relative}")
            bindings[relative] = expected
        bindings.update(document.get("additional_bindings", {}))
        protocol["bindings"] = bindings
    else:
        protocol = document
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY"
        or protocol.get("training_allowed") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("fit diagnostic governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"fit diagnostic binding changed: {relative}")
    return protocol, sha256_file(path)


def _normalized_training_rows(root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    return [
        {
            "record_id": str(row["ir_record_id"]),
            "capability": str(row["capability"]),
            "prompt": str(row["normalized_acquisition_prompt"]),
            "output": str(row["normalized_output"]),
        }
        for row in rows
    ]


def _development_rows(root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    teacher = {
        str(row["probe_id"]): row
        for row in (
            json.loads(line)
            for line in (root / protocol["teacher_reference"]).read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    rows = []
    for probe in probes:
        probe_id = str(probe["probe_id"])
        if probe_id not in teacher:
            raise Phase3Error(f"teacher target missing: {probe_id}")
        rows.append(
            {
                "record_id": probe_id,
                "capability": str(probe["canonical_capability"]),
                "prompt": str(probe["prompt"]),
                "output": str(teacher[probe_id]["output"]),
            }
        )
    if len(rows) != 1400:
        raise Phase3Error("development target depth changed")
    return rows


def _candidate(root: Path, protocol: Mapping[str, Any], system: str, device: torch.device):
    specification = protocol["systems"][system]
    directory = (root / specification["candidate_dir"]).resolve()
    metadata = _json(directory / "metadata.json")
    if (
        sha256_file(directory / "metadata.json") != specification["metadata_sha256"]
        or sha256_file(directory / "model.safetensors") != specification["checkpoint_sha256"]
        or metadata["checkpoint"]["sha256"] != specification["checkpoint_sha256"]
        or metadata.get("teacher_present_at_inference") is not False
        or metadata.get("source_blocks_retained") != 0
    ):
        raise Phase3Error(f"sealed candidate identity changed: {system}")
    _, _, tokenizer_type, model_type = _layercake_api(root, protocol)
    tokenizer = tokenizer_type.from_document(_json(directory / "tokenizer.json"))
    config = _json(directory / "model_config.json")
    model = model_type(**config).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(directory / "model.safetensors"), device=str(device)), strict=True)
    model.to(device).eval()
    return model, tokenizer, metadata


def _examples(rows: list[Mapping[str, Any]], tokenizer: Any, *, pointer_supervision: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["prompt"])
        output = str(row["output"])
        source_lexemes = tokenizer.split(prompt)
        output_lexemes = tokenizer.split(output)
        source_ids = [tokenizer.lexeme_to_id.get(piece, 3) for piece in source_lexemes]
        try:
            if pointer_supervision:
                copies = _copy_lexemes(source_lexemes, output_lexemes)
                targets = tokenizer.encode_target(output, copy_lexemes=[piece.decode("ascii") for piece in copies], source_lexemes=source_lexemes)
            else:
                targets = [tokenizer.lexeme_to_id[piece] for piece in output_lexemes] + [2]
        except (KeyError, ValueError, UnicodeDecodeError) as exc:
            rejected.append({"record_id": str(row["record_id"]), "capability": str(row["capability"]), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if tokenizer.decode_actions(targets, source_lexemes) != output.encode("utf-8"):
            raise Phase3Error(f"diagnostic target is not lossless: {row['record_id']}")
        examples.append(
            {
                "record_id": str(row["record_id"]),
                "capability": str(row["capability"]),
                "prompt": prompt,
                "output_bytes": output.encode("utf-8"),
                "source_ids": source_ids,
                "source_lexemes": source_lexemes,
                "target_actions": targets,
            }
        )
    return examples, rejected


def _collate(rows: list[Mapping[str, Any]], device: torch.device):
    source_width = max(len(row["source_ids"]) for row in rows)
    target_width = max(len(row["target_actions"]) for row in rows)
    source = torch.zeros((len(rows), source_width), dtype=torch.long, device=device)
    target = torch.full((len(rows), target_width), -100, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        source[index, : len(row["source_ids"])] = torch.tensor(row["source_ids"], device=device)
        target[index, : len(row["target_actions"])] = torch.tensor(row["target_actions"], device=device)
    return source, target


def summarize_counts(values: Counter) -> dict[str, Any]:
    """Summarize a stratum while preserving an empty stratum explicitly."""
    actions = values["actions"]
    sequences = values["sequences"]
    return {
        **dict(values),
        "action_accuracy": values["correct_actions"] / actions if actions else None,
        "exact_sequence_rate": values["exact_sequences"] / sequences if sequences else None,
        "fixed_action_accuracy": values["correct_fixed_actions"] / values["fixed_actions"] if values["fixed_actions"] else None,
        "pointer_action_accuracy": values["correct_pointer_actions"] / values["pointer_actions"] if values["pointer_actions"] else None,
        "action_type_accuracy": values["correct_action_types"] / actions if actions else None,
    }


@torch.inference_mode()
def teacher_forced(model, tokenizer, examples: list[dict[str, Any]], *, batch_size: int) -> dict[str, Any]:
    device = next(model.parameters()).device
    totals = Counter()
    per_capability = {capability: Counter() for capability in CAPABILITIES}
    nll_sum = 0.0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        source, targets = _collate(batch, device)
        log_probs = model(source, targets)["log_probs"]
        mask = targets.ge(0)
        safe = targets.clamp(min=0)
        chosen = log_probs.gather(-1, safe[:, :, None]).squeeze(-1)
        predictions = log_probs.argmax(dim=-1)
        correct = predictions.eq(targets) & mask
        target_pointer = targets.ge(tokenizer.vocab_size) & mask
        predicted_pointer = predictions.ge(tokenizer.vocab_size) & mask
        for index, row in enumerate(batch):
            row_mask = mask[index]
            row_correct = correct[index]
            count = int(row_mask.sum().item())
            right = int(row_correct.sum().item())
            pointer_count = int(target_pointer[index].sum().item())
            pointer_right = int((row_correct & target_pointer[index]).sum().item())
            fixed_count = count - pointer_count
            fixed_right = right - pointer_right
            type_right = int((predicted_pointer[index].eq(target_pointer[index]) & row_mask).sum().item())
            values = {
                "sequences": 1,
                "exact_sequences": int(right == count),
                "actions": count,
                "correct_actions": right,
                "fixed_actions": fixed_count,
                "correct_fixed_actions": fixed_right,
                "pointer_actions": pointer_count,
                "correct_pointer_actions": pointer_right,
                "correct_action_types": type_right,
            }
            totals.update(values)
            per_capability[row["capability"]].update(values)
        nll_sum += float((-chosen.masked_select(mask)).sum().item())
    result = summarize_counts(totals)
    result["mean_action_nll"] = nll_sum / totals["actions"]
    result["per_capability"] = {name: summarize_counts(values) for name, values in per_capability.items()}
    return result


def fixed_sample(examples: list[dict[str, Any]], observations_per_capability: int) -> list[dict[str, Any]]:
    selected = []
    for capability in CAPABILITIES:
        rows = sorted((row for row in examples if row["capability"] == capability), key=lambda row: row["record_id"])
        if len(rows) < observations_per_capability:
            raise Phase3Error(f"insufficient sample rows: {capability}")
        selected.extend(rows[:observations_per_capability])
    return selected


@torch.inference_mode()
def autonomous(model, tokenizer, examples: list[dict[str, Any]], *, batch_size: int, maximum_actions: int) -> dict[str, Any]:
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        source_width = max(len(row["source_ids"]) for row in batch)
        source = torch.zeros((len(batch), source_width), dtype=torch.long, device=device)
        for index, row in enumerate(batch):
            source[index, : len(row["source_ids"])] = torch.tensor(row["source_ids"], device=device)
        generated = model.generate_actions(source, maximum_actions=maximum_actions)
        for row, actions in zip(batch, generated):
            target = row["target_actions"]
            common = 0
            for left, right in zip(actions, target):
                if left != right:
                    break
                common += 1
            error = None
            try:
                raw = tokenizer.decode_actions(actions, row["source_lexemes"])
                text = raw.decode("utf-8", errors="strict")
            except Exception as exc:
                raw = b""
                text = ""
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "exact_actions": actions == target,
                    "exact_response_bytes": raw == row["output_bytes"] and error is None,
                    "common_prefix_actions": common,
                    "target_actions": len(target),
                    "common_prefix_fraction": common / len(target),
                    "repetition_collapse": repetition_collapse(text),
                    "generation_error": error,
                }
            )
    per_capability = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        per_capability[capability] = {
            "observations": len(values),
            "exact_action_sequences": sum(row["exact_actions"] for row in values),
            "exact_response_bytes": sum(row["exact_response_bytes"] for row in values),
            "mean_common_prefix_fraction": sum(row["common_prefix_fraction"] for row in values) / len(values),
            "repetition_collapses": sum(row["repetition_collapse"] for row in values),
            "generation_errors": sum(row["generation_error"] is not None for row in values),
        }
    return {
        "observations": len(rows),
        "exact_action_sequences": sum(row["exact_actions"] for row in rows),
        "exact_response_bytes": sum(row["exact_response_bytes"] for row in rows),
        "mean_common_prefix_fraction": sum(row["common_prefix_fraction"] for row in rows) / len(rows),
        "repetition_collapses": sum(row["repetition_collapse"] for row in rows),
        "generation_errors": sum(row["generation_error"] is not None for row in rows),
        "per_capability": per_capability,
        "row_sequence_sha256": hashlib.sha256(b"".join(canonical_json_bytes(row) for row in rows)).hexdigest(),
    }


def classify(system: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, bool]:
    train = system["training_teacher_forced"]
    autonomous_train = system["training_autonomous_sample"]
    development = system["development_teacher_forced"]
    return {
        "train_fit_or_capacity_limit": train["action_accuracy"] < thresholds["training_action_accuracy_minimum"] or train["exact_sequence_rate"] < thresholds["training_exact_sequence_rate_minimum"],
        "autonomous_state_drift": train["action_accuracy"] >= thresholds["state_drift_training_action_accuracy_minimum"] and autonomous_train["exact_response_bytes"] / autonomous_train["observations"] < thresholds["training_autonomous_exact_response_minimum"],
        "held_out_generalization_limit": development["action_accuracy"] + thresholds["train_to_development_action_accuracy_drop_minimum"] < train["action_accuracy"],
    }


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("V26 diagnostic requires the declared CUDA device")
    set_determinism(int(protocol["diagnostic"]["seed"]))
    training_rows = _normalized_training_rows(root, protocol)
    development_rows = _development_rows(root, protocol)
    result_systems = {}
    for system in SYSTEMS:
        model, tokenizer, metadata = _candidate(root, protocol, system, torch.device("cuda"))
        pointer = bool(protocol["systems"][system]["pointer_supervision"])
        train_examples, train_rejected = _examples(training_rows, tokenizer, pointer_supervision=pointer)
        dev_examples, dev_rejected = _examples(development_rows, tokenizer, pointer_supervision=pointer)
        train_tf = teacher_forced(model, tokenizer, train_examples, batch_size=int(protocol["diagnostic"]["teacher_forced_batch_size"]))
        dev_tf = teacher_forced(model, tokenizer, dev_examples, batch_size=int(protocol["diagnostic"]["teacher_forced_batch_size"]))
        sample = fixed_sample(train_examples, int(protocol["diagnostic"]["training_autonomous_observations_per_capability"]))
        train_auto = autonomous(model, tokenizer, sample, batch_size=int(protocol["diagnostic"]["autonomous_batch_size"]), maximum_actions=int(protocol["diagnostic"]["maximum_actions"]))
        result_systems[system] = {
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "pointer_supervision": pointer,
            "training_representable": len(train_examples),
            "training_rejected": train_rejected,
            "development_representable": len(dev_examples),
            "development_rejected": dev_rejected,
            "training_teacher_forced": train_tf,
            "development_teacher_forced": dev_tf,
            "training_autonomous_sample": train_auto,
        }
        result_systems[system]["classification"] = classify(result_systems[system], protocol["classification_thresholds"])
        del model
        torch.cuda.empty_cache()
    owners = {
        "both_systems_train_fit_limited": all(result_systems[name]["classification"]["train_fit_or_capacity_limit"] for name in SYSTEMS),
        "either_system_autonomous_state_drift": any(result_systems[name]["classification"]["autonomous_state_drift"] for name in SYSTEMS),
        "either_system_held_out_generalization_limited": any(result_systems[name]["classification"]["held_out_generalization_limit"] for name in SYSTEMS),
        "layercake_host_regression": False,
        "abi_training_or_checkpoint_changed": False,
    }
    result: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-fit-diagnostic-result/1",
        "status": "PASS_READ_ONLY_ATTRIBUTION",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED",
        "phase3_certified": False,
        "phase4_through_8": "LOCKED",
        "systems": result_systems,
        "ownership": owners,
        "decision": {
            "training_authorized": False,
            "rule": protocol["post_diagnostic_rule"],
            "next_architecture_selected": False,
        },
        "final_test_accessed": False,
        "negative_evidence_preserved": True,
        "claim_boundary": "Read-only development attribution on failed sealed checkpoints. It cannot promote V23/V24, certify Phase 3, or establish ABI superiority."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify(root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    stored = _json(output_path)
    expected = execute(root, protocol_path)
    if stored != expected:
        raise Phase3Error("stored V26 diagnostic differs from recomputation")
    return {"status": "PASS", "evidence_sha256": expected["evidence_sha256"], "phase3_certified": False}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FIT_DIAGNOSTIC_PROTOCOL_V26.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_fit_diagnostic/fit_generalization_v26.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); protocol = (root / args.protocol).resolve(); output = (root / args.output).resolve()
    if args.command == "execute":
        if output.exists():
            raise Phase3Error(f"V26 output is immutable: {output}")
        result = execute(root, protocol)
        _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    else:
        result = verify(root, protocol, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
