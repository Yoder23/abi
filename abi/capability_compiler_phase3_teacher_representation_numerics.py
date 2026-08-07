"""Read-only V60 diagnosis of V59 pooled-vector numerical disagreement."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_teacher_representation_verify import (
    _load_jsonl,
    _load_teacher,
    _select_samples,
    _token_span,
)


FORMAT = "abi-capability-compiler-phase3-teacher-representation-numerics/1"


def _batch_groups(indices: Sequence[int], batch_size: int) -> list[int]:
    if batch_size <= 0:
        raise Phase3Error("invalid extraction batch size")
    return sorted({(int(index) // batch_size) * batch_size for index in indices})


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_DIAGNOSTIC"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_verification_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("teacher representation numerics governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"teacher representation numerics binding changed: {relative}")
    return protocol, sha256_file(path)


def _prepare(row: Mapping[str, Any], tokenizer: Any, terminal: int) -> dict[str, Any]:
    rendered = str(row["rendered_generation_prompt"])
    semantic = str(row["normalized_acquisition_prompt"])
    char_start = rendered.find(semantic)
    if char_start < 0 or rendered.find(semantic, char_start + 1) >= 0:
        raise Phase3Error("diagnostic semantic span is not unique")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    prompt_start, prompt_count = _token_span(
        encoded["offset_mapping"], char_start, char_start + len(semantic)
    )
    output_ids = [int(value) for value in row["authoritative_generated_token_ids"]]
    if not output_ids or output_ids[-1] != terminal:
        raise Phase3Error("diagnostic response boundary changed")
    return {
        "input_ids": [int(value) for value in encoded["input_ids"]] + output_ids,
        "prompt_start": prompt_start,
        "prompt_count": prompt_count,
        "response_start": len(encoded["input_ids"]),
        "response_count": len(output_ids) - 1,
    }


def _measure(stored: torch.Tensor, recomputed: torch.Tensor) -> dict[str, float]:
    delta = (stored.float() - recomputed.float()).abs()
    return {
        "maximum_absolute_error": float(delta.max()),
        "mean_absolute_error": float(delta.mean()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(
            stored.float().unsqueeze(0), recomputed.float().unsqueeze(0)
        )),
        "exact_scalar_fraction": float((stored == recomputed).float().mean()),
    }


def _aggregate(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        "vectors": len(values),
        "maximum_absolute_error": max(value["maximum_absolute_error"] for value in values),
        "mean_absolute_error": sum(value["mean_absolute_error"] for value in values) / len(values),
        "minimum_cosine_similarity": min(value["cosine_similarity"] for value in values),
        "mean_exact_scalar_fraction": sum(value["exact_scalar_fraction"] for value in values) / len(values),
    }


@torch.inference_mode()
def diagnose(root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = _load_protocol(root, protocol_path)
    artifact_dir = (root / protocol["artifact"]["directory"]).resolve()
    tensors = load_file(str(artifact_dir / protocol["artifact"]["tensor"]), device="cpu")
    artifact_rows = _load_jsonl(artifact_dir / protocol["artifact"]["records"])
    phase1_rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    indices = _select_samples(
        artifact_rows,
        int(protocol["sampling"]["records_per_capability"]),
        int(protocol["sampling"]["seed"]),
    )
    model, tokenizer, manifest = _load_teacher(protocol)
    terminal = int(protocol["source"]["terminal_response_token_id"])
    prepared = [_prepare(row, tokenizer, terminal) for row in phase1_rows]
    singleton_values: list[dict[str, float]] = []
    for index in indices:
        row = prepared[index]
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
        hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True).last_hidden_state
        recomputed = {
            "prompt_pooled": hidden[0, row["prompt_start"]:row["prompt_start"] + row["prompt_count"]].float().mean(0).half().cpu(),
            "response_pooled": hidden[0, row["response_start"]:row["response_start"] + row["response_count"]].float().mean(0).half().cpu(),
        }
        singleton_values.extend(_measure(tensors[name][index], value) for name, value in recomputed.items())
    batch_size = int(protocol["sampling"]["original_batch_size"])
    selected = set(indices)
    batch_values: list[dict[str, float]] = []
    for start in _batch_groups(indices, batch_size):
        batch = prepared[start:start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        ids = torch.full((len(batch), width), int(protocol["source"]["pad_token_id"]), dtype=torch.long, device="cuda")
        mask = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        for offset, row in enumerate(batch):
            ids[offset, :len(row["input_ids"])] = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
            mask[offset, :len(row["input_ids"])] = 1
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        for offset, row in enumerate(batch):
            index = start + offset
            if index not in selected:
                continue
            recomputed = {
                "prompt_pooled": hidden[offset, row["prompt_start"]:row["prompt_start"] + row["prompt_count"]].float().mean(0).half().cpu(),
                "response_pooled": hidden[offset, row["response_start"]:row["response_start"] + row["response_count"]].float().mean(0).half().cpu(),
            }
            batch_values.extend(_measure(tensors[name][index], value) for name, value in recomputed.items())
    result = {
        "format": "abi-capability-compiler-phase3-teacher-representation-numerics-result/1",
        "status": "DIAGNOSTIC_COMPLETE_NONPROMOTIONAL",
        "protocol_sha256": protocol_sha,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "selected_records": len(indices),
        "selected_vectors": len(indices) * 2,
        "singleton_recomputation": _aggregate(singleton_values),
        "original_batch_recomputation": _aggregate(batch_values),
        "training_performed": False,
        "artifact_verified": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
        "claim_boundary": "V60 diagnoses numerical reproduction only and cannot verify the substrate, authorize training, or certify Phase 3."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output_path, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_NUMERICS_PROTOCOL_V60.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_teacher_representation/numerics_v60.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    print(json.dumps(diagnose(root, (root / args.protocol).resolve(), (root / args.output).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
