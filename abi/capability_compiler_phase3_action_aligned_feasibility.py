"""No-training feasibility study for causal, LayerCake-action-aligned teacher states."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_bpe_core import _layercake_api, _tokenizer


FORMAT = "abi-capability-compiler-phase3-action-aligned-feasibility/1"


def _piece_spans(text: str, pieces: Sequence[bytes], start: int = 0) -> list[tuple[int, int]]:
    decoded = [piece.decode("utf-8", errors="strict") for piece in pieces]
    if "".join(decoded) != text:
        raise Phase3Error("LayerCake pieces do not reconstruct the aligned text")
    spans = []
    cursor = start
    for piece in decoded:
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return spans


def _overlaps(offsets: Sequence[Sequence[int]], start: int, end: int) -> list[int]:
    selected = [index for index, (left, right) in enumerate(offsets) if int(right) > start and int(left) < end and int(right) > int(left)]
    if not selected:
        raise Phase3Error("LayerCake action has no teacher-token overlap")
    return selected


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_NO_TRAINING_FEASIBILITY" or protocol.get("extraction_authorized") is not False or protocol.get("training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("V64 action-aligned feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V64 binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    protocol, protocol_sha = _load_protocol(root, protocol_path)
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    teacher = AutoTokenizer.from_pretrained(protocol["teacher"]["snapshot_path"], local_files_only=True, trust_remote_code=False)
    _, _, tokenizer_type, _, _ = _layercake_api(root, protocol)
    layercake = _tokenizer(root, protocol, tokenizer_type)
    totals = Counter()
    capability_vectors = Counter()
    maximum_source = maximum_target = maximum_teacher = 0
    order = hashlib.sha256()
    for row in rows:
        rendered = str(row["rendered_generation_prompt"])
        semantic = str(row["normalized_acquisition_prompt"])
        output_text = str(row["normalized_output"])
        prompt_ids = teacher(rendered, add_special_tokens=False)["input_ids"]
        output_ids = [int(value) for value in row["authoritative_generated_token_ids"]]
        if not output_ids or output_ids[-1] != int(protocol["teacher"]["terminal_response_token_id"]):
            raise Phase3Error("V64 response terminal changed")
        combined = teacher(rendered + output_text, add_special_tokens=False, return_offsets_mapping=True)
        if [int(value) for value in combined["input_ids"]] != [int(value) for value in prompt_ids] + output_ids[:-1]:
            raise Phase3Error(f"V64 contextual response tokenization changed: {row['ir_record_id']}")
        semantic_start = rendered.find(semantic)
        if semantic_start < 0 or rendered.find(semantic, semantic_start + 1) >= 0:
            raise Phase3Error("V64 semantic prompt span is not unique")
        lines = semantic.splitlines()
        body = "\n".join(lines[1:]).strip()
        body_text = "\n" + body
        body_relative = semantic.find(body_text)
        if body_relative < 0 or semantic.find(body_text, body_relative + 1) >= 0:
            raise Phase3Error("V64 routed body span is not unique")
        source_pieces = layercake.split(body_text)
        target_pieces = layercake.split(output_text)
        source_spans = _piece_spans(body_text, source_pieces, semantic_start + body_relative)
        target_spans = _piece_spans(output_text, target_pieces, len(rendered))
        source_overlaps = [_overlaps(combined["offset_mapping"], start, end) for start, end in source_spans]
        target_overlaps = [_overlaps(combined["offset_mapping"], start, end) for start, end in target_spans]
        if any(min(indices) < len(prompt_ids) for indices in target_overlaps):
            raise Phase3Error("V64 target action overlaps prompt tokens")
        if any(min(indices) <= 0 for indices in target_overlaps):
            raise Phase3Error("V64 target action lacks a causal predecessor state")
        source_vectors = 1 + len(source_pieces)
        target_vectors = len(target_pieces)
        totals["source_vectors"] += source_vectors
        totals["target_vectors"] += target_vectors
        totals["teacher_prompt_tokens_referenced"] += sum(len(indices) for indices in source_overlaps)
        totals["teacher_response_tokens_referenced"] += sum(len(indices) for indices in target_overlaps)
        totals["source_actions_with_teacher_boundary_straddle"] += sum(any(int(combined["offset_mapping"][index][0]) < start or int(combined["offset_mapping"][index][1]) > end for index in indices) for (start, end), indices in zip(source_spans, source_overlaps))
        totals["target_actions_with_teacher_boundary_straddle"] += sum(any(int(combined["offset_mapping"][index][0]) < start or int(combined["offset_mapping"][index][1]) > end for index in indices) for (start, end), indices in zip(target_spans, target_overlaps))
        capability_vectors[str(row["capability"])] += source_vectors + target_vectors
        maximum_source = max(maximum_source, source_vectors)
        maximum_target = max(maximum_target, target_vectors)
        maximum_teacher = max(maximum_teacher, len(combined["input_ids"]))
        order.update(str(row["ir_record_id"]).encode("ascii") + b"\n")
    vectors = totals["source_vectors"] + totals["target_vectors"]
    width = int(protocol["representation"]["projected_width"])
    payload = vectors * width * 2
    passed = len(rows) == 7000 and len(capability_vectors) == 14 and payload <= int(protocol["representation"]["payload_bytes_maximum"])
    result = {
        "format": "abi-capability-compiler-phase3-action-aligned-feasibility-result/1",
        "status": "PASS_FEASIBILITY_EXTRACTION_PROTOCOL_DESIGN_AUTHORIZED" if passed else "FAIL_FEASIBILITY_BRANCH_CLOSED",
        "protocol_sha256": protocol_sha,
        "records": len(rows),
        "record_order_sha256": order.hexdigest(),
        "source_vectors": totals["source_vectors"],
        "target_vectors": totals["target_vectors"],
        "vectors": vectors,
        "projected_width": width,
        "tensor_payload_bytes": payload,
        "teacher_prompt_tokens_referenced": totals["teacher_prompt_tokens_referenced"],
        "teacher_response_tokens_referenced": totals["teacher_response_tokens_referenced"],
        "source_actions_with_teacher_boundary_straddle": totals["source_actions_with_teacher_boundary_straddle"],
        "target_actions_with_teacher_boundary_straddle": totals["target_actions_with_teacher_boundary_straddle"],
        "maximum_source_vectors_per_record": maximum_source,
        "maximum_target_vectors_per_record": maximum_target,
        "maximum_teacher_sequence_tokens": maximum_teacher,
        "vectors_by_capability": dict(sorted(capability_vectors.items())),
        "causal_response_state": "For every teacher response token overlapped by a LayerCake target action, use the final hidden state immediately preceding that teacher token; average within the LayerCake action after fixed projection.",
        "route_state": "Use the projected mean semantic-prompt hidden state for the first trained route-control action.",
        "teacher_model_loaded": False,
        "teacher_forward_passes": 0,
        "extraction_performed": False,
        "training_performed": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
        "claim_boundary": "V64 proves mapping and payload feasibility only; it does not extract, train, verify quality, or certify Phase 3."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_FEASIBILITY_PROTOCOL_V64.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_action_aligned/feasibility_v64.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    print(json.dumps(run(root, (root / args.protocol).resolve(), (root / args.output).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
