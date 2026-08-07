"""No-training feasibility and accounting for richer frozen-teacher signals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir


FORMAT = "abi-capability-compiler-phase3-teacher-representation-feasibility/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_NO_TRAINING_FEASIBILITY" or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("training_authorized") is not False:
        raise Phase3Error("teacher representation feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"teacher representation feasibility binding changed: {relative}")
    source_config = Path(protocol["source_config"]["path"])
    if not source_config.is_file() or sha256_file(source_config) != protocol["source_config"]["sha256"]:
        raise Phase3Error("frozen teacher config changed")
    return protocol, sha256_file(path)


def _candidate(name: str, *, payload_bytes: int, vectors: int, scalars: int, per_record: bool, covers_prompt: bool, covers_response: bool, tokenizer_independent: bool, direct_layercake_alignment: bool, standard_method: str) -> dict[str, Any]:
    return {"name": name, "payload_bytes": int(payload_bytes), "vectors": int(vectors), "scalars": int(scalars), "per_record": per_record, "covers_prompt": covers_prompt, "covers_response": covers_response, "tokenizer_independent": tokenizer_independent, "direct_layercake_alignment": direct_layercake_alignment, "standard_method": standard_method}


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("teacher representation feasibility output exists")
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    config = _json(Path(protocol["source_config"]["path"]))
    records = len(rows)
    input_tokens = sum(int(row["teacher_input_tokens"]) for row in rows)
    output_tokens = sum(int(row["authoritative_teacher_tokens"]) for row in rows)
    hidden = int(config["hidden_size"])
    vocabulary = int(config["vocab_size"])
    layers = int(config["num_hidden_layers"])
    fp16 = 2
    candidates = [
        _candidate("full_output_logits_fp16", payload_bytes=output_tokens * vocabulary * fp16, vectors=output_tokens, scalars=output_tokens * vocabulary, per_record=True, covers_prompt=False, covers_response=True, tokenizer_independent=False, direct_layercake_alignment=False, standard_method="full-logit knowledge distillation"),
    ]
    for top_k in (4, 8, 16, 32):
        candidates.append(_candidate(f"top{top_k}_output_logits_uint16_fp16_plus_logz", payload_bytes=output_tokens * (top_k * 4 + 2), vectors=output_tokens, scalars=output_tokens * (top_k + 1), per_record=True, covers_prompt=False, covers_response=True, tokenizer_independent=False, direct_layercake_alignment=False, standard_method="top-k logit knowledge distillation"))
    candidates.extend([
        _candidate("response_token_final_hidden_fp16", payload_bytes=output_tokens * hidden * fp16, vectors=output_tokens, scalars=output_tokens * hidden, per_record=True, covers_prompt=False, covers_response=True, tokenizer_independent=True, direct_layercake_alignment=True, standard_method="token-level representation distillation"),
        _candidate("prompt_response_token_final_hidden_fp16", payload_bytes=(input_tokens + output_tokens) * hidden * fp16, vectors=input_tokens + output_tokens, scalars=(input_tokens + output_tokens) * hidden, per_record=True, covers_prompt=True, covers_response=True, tokenizer_independent=True, direct_layercake_alignment=True, standard_method="token-level representation distillation"),
        _candidate("dual_pooled_final_hidden_fp16", payload_bytes=records * 2 * hidden * fp16, vectors=records * 2, scalars=records * 2 * hidden, per_record=True, covers_prompt=True, covers_response=True, tokenizer_independent=True, direct_layercake_alignment=True, standard_method="pooled representation distillation"),
        _candidate("dual_pooled_layers_16_24_32_fp16", payload_bytes=records * 2 * hidden * fp16 * 3, vectors=records * 2 * 3, scalars=records * 2 * hidden * 3, per_record=True, covers_prompt=True, covers_response=True, tokenizer_independent=True, direct_layercake_alignment=True, standard_method="multi-layer pooled representation distillation"),
        _candidate("capability_prototypes_final_hidden_fp16", payload_bytes=len(CAPABILITIES) * 2 * hidden * fp16, vectors=len(CAPABILITIES) * 2, scalars=len(CAPABILITIES) * 2 * hidden, per_record=False, covers_prompt=True, covers_response=True, tokenizer_independent=True, direct_layercake_alignment=True, standard_method="capability-prototype representation distillation"),
    ])
    maximum = int(protocol["selection"]["payload_bytes_maximum"])
    eligible = [row for row in candidates if row["payload_bytes"] <= maximum and row["per_record"] and row["covers_prompt"] and row["covers_response"] and row["tokenizer_independent"] and row["direct_layercake_alignment"]]
    eligible.sort(key=lambda row: (row["payload_bytes"], row["name"]))
    selected = eligible[0] if eligible else None
    expected = protocol["selection"]["expected_selection"]
    if selected is None or selected["name"] != expected:
        status = "FAIL_NO_QUALIFYING_REPRESENTATION"
        extraction_authorized = False
    else:
        status = "PASS_FEASIBILITY_SELECTION_ONLY"
        extraction_authorized = False
    result = {
        "format": "abi-capability-compiler-phase3-teacher-representation-feasibility-result/1",
        "status": status,
        "protocol_sha256": protocol_sha,
        "teacher": {"model": protocol["teacher"]["model"], "revision": protocol["teacher"]["revision"], "hidden_size": hidden, "layers": layers, "vocabulary": vocabulary, "dtype": config["torch_dtype"], "config_sha256": protocol["source_config"]["sha256"]},
        "corpus": {"records": records, "teacher_input_tokens": input_tokens, "teacher_output_tokens": output_tokens, "teacher_forward_tokens": input_tokens + output_tokens, "capabilities": len(CAPABILITIES)},
        "candidates": candidates,
        "selection_rule": protocol["selection"],
        "selected": selected,
        "selected_imported_information": None if selected is None else {"stored_vectors": selected["vectors"], "stored_scalars": selected["scalars"], "payload_bytes": selected["payload_bytes"], "logits_stored": 0, "hidden_activations_stored": selected["vectors"], "source_parameters_copied": 0},
        "extraction_authorized": extraction_authorized,
        "training_authorized": False,
        "teacher_present_at_final_inference": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
        "next_step": "Preregister one extraction-only run for the selected representation, including exact model load, pooling, dtype, record order, hashes, GPU/RAM/time accounting, and hostile verification." if selected is not None else "Stop richer-representation work; no extraction is authorized.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_FEASIBILITY_PROTOCOL_V56.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_teacher_representation/feasibility_v56.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, (root / args.protocol).resolve(), (root / args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
