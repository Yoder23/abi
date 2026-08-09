"""No-model feasibility gate for compact causal probability-field transfer.

This gate inspects only immutable acquisition evidence and the frozen tokenizer.
It never loads teacher weights, extracts logits, or trains a candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import zipfile

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir


FORMAT = "abi-capability-compiler-phase3-causal-field-feasibility/1"
HOST_SPECIAL_ACTIONS = 4


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n",
    )


def _archive_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        return [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line.strip()]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_MODEL_FEASIBILITY"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("logit_extraction_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("causal-field feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"causal-field feasibility binding changed: {relative}")
    return protocol, sha256_file(path)


def _rows(root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    original = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    targeted = _archive_rows((root / protocol["targeted_ir"]).resolve())
    rows: list[dict[str, Any]] = []
    for row in original:
        body = "\n".join(str(row["normalized_acquisition_prompt"]).splitlines()[1:]).strip()
        rows.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "host_prompt": f"Capability route: {row['capability']}\n{body}",
                "rendered_prompt": str(row["rendered_generation_prompt"]),
                "output": str(row["normalized_output"]),
                "generated_ids": [int(value) for value in row["authoritative_generated_token_ids"]],
                "teacher_input_tokens": int(row["teacher_input_tokens"]),
                "teacher_output_tokens": int(row["authoritative_teacher_tokens"]),
            }
        )
    for row in targeted:
        rows.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "host_prompt": f"Capability route: {row['capability']}\n{row['host_conformant_acquisition_prompt']}",
                "rendered_prompt": str(row["rendered_generation_prompt"]),
                "output": str(row["normalized_output"]),
                "generated_ids": [int(value) for value in row["authoritative_generated_token_ids"]],
                "teacher_input_tokens": int(row["teacher_input_tokens"]),
                "teacher_output_tokens": int(row["authoritative_teacher_tokens"]),
            }
        )
    return rows


def _causal_decoder_parameters(*, vocabulary: int, width: int, layers: int, feedforward: int, maximum_sequence: int) -> int:
    # Tied token input/output table; learned positions; PyTorch-style MHA,
    # two affine feed-forward layers, two norms per block, final norm, output bias.
    embedding = vocabulary * width
    position = maximum_sequence * width
    attention = 4 * width * width + 4 * width
    feed_forward = 2 * width * feedforward + feedforward + width
    norms = 4 * width
    final_norm = 2 * width
    output_bias = vocabulary
    return embedding + position + layers * (attention + feed_forward + norms) + final_norm + output_bias


def evaluate(root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    from tokenizers import Tokenizer

    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_path.exists():
        raise Phase3Error("causal-field feasibility output already exists")
    tokenizer = Tokenizer.from_file(str(Path(protocol["source"]["tokenizer_json"])))
    rows = _rows(root, protocol)
    if len(rows) != int(protocol["expected"]["records"]) or len({row["record_id"] for row in rows}) != len(rows):
        raise Phase3Error("causal-field acquisition inventory changed")

    terminal_id = int(protocol["source"]["terminal_token_id"])
    external_vocabulary = int(protocol["source"]["external_vocabulary"])
    top_k = int(protocol["probability_field"]["top_k"])
    counts = Counter(row["capability"] for row in rows)
    source_maximum = target_maximum = combined_maximum = 0
    teacher_input_tokens = teacher_output_tokens = 0
    raw_prompt_bytes = output_bytes = 0
    unique_bytes: set[bytes] = set()
    order = hashlib.sha256()

    for row in rows:
        rendered_ids = tokenizer.encode(row["rendered_prompt"], add_special_tokens=False).ids
        if len(rendered_ids) != row["teacher_input_tokens"]:
            raise Phase3Error(f"teacher input token count changed: {row['record_id']}")
        generated = row["generated_ids"]
        if len(generated) != row["teacher_output_tokens"] or not generated or generated[-1] != terminal_id:
            raise Phase3Error(f"authoritative response boundary changed: {row['record_id']}")
        if tokenizer.decode(generated[:-1], skip_special_tokens=False) != row["output"]:
            raise Phase3Error(f"authoritative response no longer decodes exactly: {row['record_id']}")
        host_source = tokenizer.encode(row["host_prompt"], add_special_tokens=False).ids
        if tokenizer.decode(host_source, skip_special_tokens=False) != row["host_prompt"]:
            raise Phase3Error(f"host prompt does not round-trip: {row['record_id']}")
        host_targets = generated[:-1] + [terminal_id]
        if any(not 0 <= value < external_vocabulary for value in host_source + host_targets):
            raise Phase3Error(f"external action is outside vocabulary: {row['record_id']}")
        source_maximum = max(source_maximum, len(host_source))
        target_maximum = max(target_maximum, len(host_targets))
        combined_maximum = max(combined_maximum, len(host_source) + 1 + len(host_targets))
        teacher_input_tokens += len(rendered_ids)
        teacher_output_tokens += len(generated)
        raw_prompt_bytes += len(row["rendered_prompt"].encode("utf-8"))
        output_bytes += len(row["output"].encode("utf-8"))
        unique_bytes.add(row["rendered_prompt"].encode("utf-8"))
        order.update(row["record_id"].encode("ascii") + b"\n")

    if set(counts) != set(CAPABILITIES) or any(value != int(protocol["expected"]["records_per_capability"]) for value in counts.values()):
        raise Phase3Error("causal-field capability balance changed")

    actions = teacher_output_tokens
    id_bytes = int(protocol["probability_field"]["id_bytes"])
    probability_bytes = int(protocol["probability_field"]["probability_bytes"])
    residual_bytes = int(protocol["probability_field"]["residual_mass_bytes"])
    offset_bytes = (len(rows) + 1) * int(protocol["probability_field"]["offset_bytes"])
    tensor_payload = actions * (top_k * (id_bytes + probability_bytes) + residual_bytes) + offset_bytes
    architecture = protocol["architecture"]
    host_vocabulary = external_vocabulary + HOST_SPECIAL_ACTIONS
    parameters = _causal_decoder_parameters(
        vocabulary=host_vocabulary,
        width=int(architecture["model_width"]),
        layers=int(architecture["decoder_layers"]),
        feedforward=int(architecture["feedforward_width"]),
        maximum_sequence=int(architecture["maximum_sequence_actions"]),
    )
    gates = {
        "all_records_exact": True,
        "balanced_capabilities": True,
        "source_bound": source_maximum <= int(architecture["maximum_source_actions"]),
        "target_bound": target_maximum <= int(architecture["maximum_target_actions"]),
        "combined_context_bound": combined_maximum <= int(architecture["maximum_sequence_actions"]),
        "probability_payload_bound": tensor_payload <= int(protocol["probability_field"]["payload_bytes_maximum"]),
        "parameter_floor": parameters >= int(architecture["parameter_minimum"]),
        "parameter_ceiling": parameters <= int(architecture["parameter_maximum"]),
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-causal-field-feasibility-result/1",
        "status": "PASS_CAUSAL_FIELD_FEASIBILITY_ONLY" if passed else "FAIL_CAUSAL_FIELD_FEASIBILITY",
        "protocol_sha256": protocol_sha,
        "inventory": {
            "records": len(rows),
            "records_per_capability": dict(sorted(counts.items())),
            "record_order_sha256": order.hexdigest(),
            "raw_source_prompts": len(rows),
            "raw_prompt_bytes": raw_prompt_bytes,
            "unique_raw_prompt_utf8_bytes": sum(len(value) for value in unique_bytes),
            "teacher_input_tokens": teacher_input_tokens,
            "teacher_output_tokens": teacher_output_tokens,
            "teacher_output_bytes": output_bytes,
            "source_action_maximum": source_maximum,
            "target_action_maximum": target_maximum,
            "combined_context_maximum": combined_maximum,
        },
        "probability_field": {
            "top_k": top_k,
            "prediction_positions": actions,
            "token_id_dtype": "uint16",
            "probability_dtype": "float16",
            "residual_mass_dtype": "float16",
            "ragged_offset_dtype": "int64",
            "tensor_payload_bytes": tensor_payload,
            "payload_bytes_maximum": int(protocol["probability_field"]["payload_bytes_maximum"]),
            "stored_logits": actions * top_k,
            "stored_probability_scalars": actions * (top_k + 1),
            "source_parameters_copied": 0,
            "hidden_activations_stored": 0,
        },
        "architecture": {
            **architecture,
            "host_vocabulary": host_vocabulary,
            "tied_input_output_embedding": True,
            "calculated_parameters": parameters,
            "private_execution": "decoder_only_causal_transformer_with_persistent_incremental_state",
        },
        "gates": gates,
        "teacher_model_loaded": False,
        "teacher_forward_passes": 0,
        "logit_extraction_performed": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "claim_boundary": "No-model feasibility and accounting only; this is not extracted teacher knowledge, a trained candidate, quality evidence, or a Phase 3 certificate.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_json(output_path, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CAUSAL_FIELD_FEASIBILITY_PROTOCOL_V177.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3/causal_field_v177/feasibility_v178.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = evaluate(root, (root / args.protocol).resolve(), (root / args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
