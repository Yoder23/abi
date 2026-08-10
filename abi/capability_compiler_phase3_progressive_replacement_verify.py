"""Hostile exact verifier for the copied progressive-replacement substrate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open
import torch

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_progressive_replacement_extract import runtime_source_rows, substrate_parameter_count


FORMAT = "abi-capability-compiler-phase3-progressive-replacement-verifier/1"


def expected_keys(layers: int) -> set[str]:
    keys = {"token_embedding.weight", "lm_head.weight", "final_norm.weight"}
    for layer in range(layers):
        keys.add(f"layers.{layer}.input_norm.weight")
        keys.add(f"layers.{layer}.post_attention_norm.weight")
    return keys


def _source_slice(snapshot: Path, weight_map: Mapping[str, str], key: str, start: int | None = None, end: int | None = None) -> torch.Tensor:
    relative = weight_map.get(key)
    if relative is None:
        raise Phase3Error(f"source index lacks verifier tensor: {key}")
    with safe_open(str(snapshot / relative), framework="pt", device="cpu") as handle:
        view = handle.get_slice(key)
        return view[:] if start is None else view[start:end]


def _artifact_slice(path: Path, key: str, start: int | None = None, end: int | None = None) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        view = handle.get_slice(key)
        return view[:] if start is None else view[start:end]


def _equal_gpu(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.cuda(), right.cuda()))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_HOSTILE_EXACT_GPU_VERIFICATION"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("progressive substrate verifier governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"progressive substrate verifier binding changed: {name}")
    return protocol, sha256_file(path)


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("hostile verifier requires preregistered CUDA")
    source = protocol["source"]
    target = protocol["target"]
    artifact_path = root / protocol["artifact"]["path"]
    result = json.loads((root / protocol["artifact"]["result_path"]).read_text(encoding="utf-8"))
    if sha256_file(artifact_path) != protocol["artifact"]["sha256"] or result.get("tensor_sha256") != protocol["artifact"]["sha256"]:
        raise Phase3Error("copied substrate identity changed")
    snapshot = Path(source["snapshot_path"])
    weight_map = json.loads(Path(source["index_path"]).read_text(encoding="utf-8"))["weight_map"]
    layers = int(target["replacement_layers"])
    required = expected_keys(layers)
    with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
        actual = set(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in handle.keys()}
        dtypes = {key: str(handle.get_slice(key).get_dtype()) for key in handle.keys()}
    static = {
        "exact_key_set": actual == required,
        "tensor_count": len(actual) == 67,
        "all_bfloat16": set(dtypes.values()) == {"BF16"},
        "lexical_shapes": shapes.get("token_embedding.weight") == [32_015, 3_072]
        and shapes.get("lm_head.weight") == [32_015, 3_072],
        "normalization_shapes": all(shapes.get(key) == [3_072] for key in required if key not in {"token_embedding.weight", "lm_head.weight"}),
    }

    special_rows = list(target["host_special_source_rows"])
    external_actions = int(target["external_actions"])
    rows = runtime_source_rows(external_actions=external_actions, host_special_source_rows=special_rows)
    chunk_rows = int(protocol["verification"]["chunk_rows"])
    verified_scalars = 0
    lexical_equal = True
    for source_key, artifact_key in (("model.embed_tokens.weight", "token_embedding.weight"), ("lm_head.weight", "lm_head.weight")):
        for index, source_row in enumerate(special_rows):
            source_value = _source_slice(snapshot, weight_map, source_key, source_row, source_row + 1)
            artifact_value = _artifact_slice(artifact_path, artifact_key, index, index + 1)
            lexical_equal = lexical_equal and _equal_gpu(source_value, artifact_value)
            verified_scalars += source_value.numel()
        offset = len(special_rows)
        for start in range(0, external_actions, chunk_rows):
            end = min(start + chunk_rows, external_actions)
            source_value = _source_slice(snapshot, weight_map, source_key, start, end)
            artifact_value = _artifact_slice(artifact_path, artifact_key, offset + start, offset + end)
            lexical_equal = lexical_equal and _equal_gpu(source_value, artifact_value)
            verified_scalars += source_value.numel()
        print(json.dumps({"verified_tensor": artifact_key, "verified_scalars": verified_scalars}), flush=True)

    norms_equal = True
    for layer in range(layers):
        for source_name, target_name in (("input_layernorm", "input_norm"), ("post_attention_layernorm", "post_attention_norm")):
            source_key = f"model.layers.{layer}.{source_name}.weight"
            artifact_key = f"layers.{layer}.{target_name}.weight"
            source_value = _source_slice(snapshot, weight_map, source_key)
            artifact_value = _artifact_slice(artifact_path, artifact_key)
            norms_equal = norms_equal and _equal_gpu(source_value, artifact_value)
            verified_scalars += source_value.numel()
    source_final = _source_slice(snapshot, weight_map, "model.norm.weight")
    artifact_final = _artifact_slice(artifact_path, "final_norm.weight")
    norms_equal = norms_equal and _equal_gpu(source_final, artifact_final)
    verified_scalars += source_final.numel()

    expected_scalars = substrate_parameter_count(runtime_vocabulary=len(rows), full_width=int(target["full_width"]), layers=layers)
    mapping = {
        "runtime_row_slots": len(rows),
        "unique_source_rows": len(set(rows)),
        "duplicate_host_mappings": len(rows) - len(set(rows)),
        "distinct_unused_source_rows": int(source["vocab_size"]) - len(set(rows)),
        "deployed_row_slots_fewer_than_source": int(source["vocab_size"]) - len(rows),
        "special_rows_exact": rows[:4] == [32_000, 32_001, 32_007, 0],
    }
    gates = {
        **static,
        "lexical_values_exact": lexical_equal,
        "normalization_values_exact": norms_equal,
        "every_artifact_scalar_recomputed": verified_scalars == expected_scalars == int(target["copied_substrate_parameters"]),
        "runtime_row_slots": mapping["runtime_row_slots"] == 32_015,
        "unique_source_rows": mapping["unique_source_rows"] == 32_011,
        "duplicate_host_mappings": mapping["duplicate_host_mappings"] == 4,
        "distinct_unused_source_rows": mapping["distinct_unused_source_rows"] == 53,
        "deployed_row_slots_fewer_than_source": mapping["deployed_row_slots_fewer_than_source"] == 49,
        "special_rows_exact": mapping["special_rows_exact"],
        "zero_source_transformer_blocks": result["source"]["complete_blocks_retained"] == 0,
        "zero_teacher_forward_or_training": result["accounting"]["teacher_forward_tokens"] == 0 and result["training_performed"] is False,
    }
    passed = all(gates.values())
    return {
        "format": FORMAT,
        "status": "PASS_VERIFIED_TRAINING_PROTOCOL_MAY_BE_DESIGNED" if passed else "FAIL_VERIFICATION_BRANCH_CLOSED",
        "protocol_sha256": protocol_sha,
        "artifact_sha256": sha256_file(artifact_path),
        "verified_scalars": verified_scalars,
        "expected_scalars": expected_scalars,
        "mapping": mapping,
        "gates": gates,
        "teacher_model_loaded": False,
        "teacher_forward_tokens": 0,
        "training_performed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "terminology_correction": "There are 49 fewer deployed row slots than source weight rows, 53 distinct unused source rows, and four deliberate duplicate source mappings for host-special positions.",
        "next_gate": "Design one bounded layer-local progressive replacement training protocol with frozen verified substrate and explicit per-layer fit gates.",
        "claim_boundary": "Exact copied-substrate verification only; no replacement fit, English quality, measured speed, transfer, or superiority claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_PROGRESSIVE_REPLACEMENT_VERIFIER_PROTOCOL_V223.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_progressive_replacement/verification_v224.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error("progressive verifier output exists")
    result = execute(root, root / args.protocol)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_VERIFIED_TRAINING_PROTOCOL_MAY_BE_DESIGNED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
