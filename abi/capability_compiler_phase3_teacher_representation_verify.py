"""Hostile verifier for the V58 pooled frozen-teacher substrate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_pipeline import build_source_model_manifest


FORMAT = "abi-capability-compiler-phase3-teacher-representation-verifier/1"
ARTIFACT_FORMAT = "abi-capability-compiler-phase3-dual-pooled-teacher-substrate/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_HOSTILE_VERIFICATION_ONLY"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("teacher representation verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"teacher representation verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def _verify_evidence_hash(metadata: Mapping[str, Any]) -> None:
    payload = dict(metadata)
    claimed = payload.pop("evidence_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if claimed != actual:
        raise Phase3Error("teacher substrate evidence hash changed")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase3Error("teacher substrate record is not an object")
            rows.append(value)
    return rows


def _verify_tensor_structure(
    tensors: Mapping[str, torch.Tensor], records: int, width: int
) -> dict[str, Any]:
    if set(tensors) != {"prompt_pooled", "response_pooled"}:
        raise Phase3Error("teacher substrate tensor keys changed")
    summary: dict[str, Any] = {}
    for name in ("prompt_pooled", "response_pooled"):
        tensor = tensors[name]
        if tensor.dtype != torch.float16 or tuple(tensor.shape) != (records, width):
            raise Phase3Error(f"teacher substrate {name} shape or dtype changed")
        if not bool(torch.isfinite(tensor).all()):
            raise Phase3Error(f"teacher substrate {name} contains nonfinite values")
        norms = tensor.float().norm(dim=1)
        if bool((norms <= 0).any()) or float(norms.std()) <= 0:
            raise Phase3Error(f"teacher substrate {name} is degenerate")
        summary[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "minimum_norm": float(norms.min()),
            "maximum_norm": float(norms.max()),
            "mean_norm": float(norms.mean()),
            "norm_stddev": float(norms.std()),
        }
    return summary


def _select_samples(
    rows: Sequence[Mapping[str, Any]], per_capability: int, seed: int
) -> list[int]:
    by_capability: dict[str, list[tuple[str, int]]] = {}
    for index, row in enumerate(rows):
        score = hashlib.sha256(
            f"{seed}\0{row['record_id']}".encode("ascii")
        ).hexdigest()
        by_capability.setdefault(str(row["capability"]), []).append((score, index))
    selected: list[int] = []
    for capability in sorted(by_capability):
        candidates = sorted(by_capability[capability])
        if len(candidates) < per_capability:
            raise Phase3Error("insufficient verifier samples in capability")
        selected.extend(index for _, index in candidates[:per_capability])
    return sorted(selected)


def _token_span(offsets: Sequence[Sequence[int]], start: int, end: int) -> tuple[int, int]:
    selected = [
        index
        for index, (left, right) in enumerate(offsets)
        if int(right) > start and int(left) < end and int(right) > int(left)
    ]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise Phase3Error("verifier found noncontiguous semantic token offsets")
    covered = [(int(offsets[index][0]), int(offsets[index][1])) for index in selected]
    if covered[0][0] != start or covered[-1][1] != end:
        raise Phase3Error("verifier found a semantic boundary straddle")
    if any(left[1] != right[0] for left, right in zip(covered, covered[1:])):
        raise Phase3Error("verifier found a semantic token offset gap")
    return selected[0], len(selected)


def _verify_records(
    phase1_rows: Sequence[Mapping[str, Any]], artifact_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(phase1_rows) != len(artifact_rows):
        raise Phase3Error("teacher substrate record count changed")
    capabilities = Counter()
    order_hash = hashlib.sha256()
    for source, artifact in zip(phase1_rows, artifact_rows):
        expected = {
            "record_id": str(source["ir_record_id"]),
            "capability": str(source["capability"]),
            "prompt_sha256": str(source["normalized_acquisition_prompt_sha256"]),
            "output_sha256": str(source["normalized_output_sha256"]),
        }
        if any(artifact.get(key) != value for key, value in expected.items()):
            raise Phase3Error("teacher substrate provenance row changed")
        if int(artifact.get("prompt_count", 0)) <= 0 or int(artifact.get("response_count", 0)) <= 0:
            raise Phase3Error("teacher substrate contains an empty pooled span")
        capabilities[expected["capability"]] += 1
        order_hash.update(expected["record_id"].encode("ascii") + b"\n")
    if len(capabilities) != 14 or set(capabilities.values()) != {500}:
        raise Phase3Error("teacher substrate capability balance changed")
    return {
        "records": len(artifact_rows),
        "capabilities": dict(sorted(capabilities.items())),
        "record_order_sha256": order_hash.hexdigest(),
    }


def _load_teacher(protocol: Mapping[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = protocol["source"]
    snapshot = Path(source["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda").eval()
    weights = [
        {
            "relative_path": path.relative_to(snapshot).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(snapshot.glob("*.safetensors"))
    ]
    manifest = build_source_model_manifest(
        model_id=source["model"],
        revision=source["revision"],
        revision_is_immutable=True,
        architecture=model.config.architectures[0],
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tokenizer_id=source["model"],
        tokenizer_revision=source["revision"],
        license_id=source["license"],
        weight_files=weights,
        trust_remote_code=False,
    )
    if manifest["source_manifest_sha256"] != source["source_manifest_sha256"]:
        raise Phase3Error("verifier teacher source manifest changed")
    return model, tokenizer, manifest


@torch.inference_mode()
def _recompute_samples(
    protocol: Mapping[str, Any],
    phase1_rows: Sequence[Mapping[str, Any]],
    artifact_rows: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise Phase3Error("CUDA is required for registered teacher recomputation")
    indices = _select_samples(
        artifact_rows,
        int(protocol["sampling"]["records_per_capability"]),
        int(protocol["sampling"]["seed"]),
    )
    model, tokenizer, manifest = _load_teacher(protocol)
    maximum_error = 0.0
    minimum_cosine = 1.0
    verified_vectors = 0
    terminal = int(protocol["source"]["terminal_response_token_id"])
    for index in indices:
        row = phase1_rows[index]
        rendered = str(row["rendered_generation_prompt"])
        semantic = str(row["normalized_acquisition_prompt"])
        char_start = rendered.find(semantic)
        if char_start < 0 or rendered.find(semantic, char_start + 1) >= 0:
            raise Phase3Error("verifier semantic text span is not unique")
        encoded = tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
        prompt_start, prompt_count = _token_span(
            encoded["offset_mapping"], char_start, char_start + len(semantic)
        )
        output_ids = [int(value) for value in row["authoritative_generated_token_ids"]]
        if not output_ids or output_ids[-1] != terminal:
            raise Phase3Error("verifier response boundary changed")
        input_ids = [int(value) for value in encoded["input_ids"]] + output_ids
        ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
        hidden = model.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        response_start = len(encoded["input_ids"])
        expected = {
            "prompt_pooled": hidden[0, prompt_start:prompt_start + prompt_count]
            .float().mean(0).half().cpu(),
            "response_pooled": hidden[0, response_start:response_start + len(output_ids) - 1]
            .float().mean(0).half().cpu(),
        }
        for name, recomputed in expected.items():
            stored = tensors[name][index]
            error = float((stored.float() - recomputed.float()).abs().max())
            cosine = float(torch.nn.functional.cosine_similarity(
                stored.float().unsqueeze(0), recomputed.float().unsqueeze(0)
            ))
            maximum_error = max(maximum_error, error)
            minimum_cosine = min(minimum_cosine, cosine)
            verified_vectors += 1
    if maximum_error > float(protocol["sampling"]["maximum_absolute_error"]):
        raise Phase3Error("teacher substrate sample values do not reproduce")
    if minimum_cosine < float(protocol["sampling"]["minimum_cosine_similarity"]):
        raise Phase3Error("teacher substrate sample directions do not reproduce")
    return {
        "sample_records": len(indices),
        "sample_vectors": verified_vectors,
        "records_per_capability": int(protocol["sampling"]["records_per_capability"]),
        "selected_record_ids": [str(artifact_rows[index]["record_id"]) for index in indices],
        "maximum_absolute_error": maximum_error,
        "minimum_cosine_similarity": minimum_cosine,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
    }


def verify(root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = _load_protocol(root, protocol_path)
    artifact_dir = (root / protocol["artifact"]["directory"]).resolve()
    metadata_path = artifact_dir / "metadata.json"
    metadata = _json(metadata_path)
    if metadata.get("format") != ARTIFACT_FORMAT or metadata.get("status") != "EXTRACTED_UNVERIFIED_TRAINING_PROHIBITED":
        raise Phase3Error("teacher substrate metadata governance changed")
    _verify_evidence_hash(metadata)
    tensor_path = artifact_dir / metadata["artifact"]["path"]
    records_path = artifact_dir / metadata["records"]["path"]
    for path, entry in ((tensor_path, metadata["artifact"]), (records_path, metadata["records"])):
        expected_bytes = entry["file_bytes"] if "file_bytes" in entry else entry["bytes"]
        if not path.is_file() or sha256_file(path) != entry["sha256"] or path.stat().st_size != int(expected_bytes):
            raise Phase3Error(f"teacher substrate file changed: {path.name}")
    if sha256_file(metadata_path) != protocol["artifact"]["metadata_sha256"]:
        raise Phase3Error("teacher substrate metadata file changed")
    tensors = load_file(str(tensor_path), device="cpu")
    structure = _verify_tensor_structure(
        tensors,
        int(protocol["expected"]["records"]),
        int(protocol["expected"]["hidden_width"]),
    )
    phase1_rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    artifact_rows = _load_jsonl(records_path)
    provenance = _verify_records(phase1_rows, artifact_rows)
    if provenance["record_order_sha256"] != protocol["expected"]["record_order_sha256"]:
        raise Phase3Error("teacher substrate record order changed")
    accounting = metadata["imported_information"]
    if (
        accounting.get("logits_stored") != 0
        or accounting.get("source_parameters_copied") != 0
        or accounting.get("hidden_activation_bytes") != protocol["expected"]["tensor_payload_bytes"]
        or metadata.get("teacher_present_in_artifact") is not False
        or metadata.get("training_performed") is not False
        or metadata.get("final_test_accessed") is not False
    ):
        raise Phase3Error("teacher substrate imported-information boundary changed")
    started = time.perf_counter()
    recomputation = _recompute_samples(
        protocol, phase1_rows, artifact_rows, tensors
    )
    result = {
        "format": "abi-capability-compiler-phase3-teacher-representation-verification-result/1",
        "status": "PASS_ARTIFACT_VERIFIED_TRAINING_STILL_PROHIBITED",
        "protocol_sha256": protocol_sha,
        "artifact_metadata_sha256": sha256_file(metadata_path),
        "tensor_sha256": sha256_file(tensor_path),
        "records_sha256": sha256_file(records_path),
        "tensor_structure": structure,
        "provenance": provenance,
        "teacher_sample_recomputation": recomputation,
        "verification_seconds": time.perf_counter() - started,
        "stored_logits": 0,
        "copied_source_parameters": 0,
        "teacher_present_in_artifact": False,
        "training_performed": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
        "claim_boundary": "The teacher substrate is reproducible and provenance-bound. This does not prove that a LayerCake candidate can learn from it or pass Phase 3."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        output_path,
        json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_VERIFIER_PROTOCOL_V59.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_teacher_representation/verification_v59.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root, (root / args.protocol).resolve(), (root / args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
