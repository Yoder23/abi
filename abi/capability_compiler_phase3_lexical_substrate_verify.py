"""Hostile verification of the V82 projected lexical substrate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_lexical_substrate_extract import project_rows


def selected_rows(key: str, count: int, rows: int) -> list[int]:
    values = set()
    nonce = 0
    while len(values) < count:
        digest = hashlib.sha256(f"v84:{key}:{nonce}".encode()).digest()
        values.add(int.from_bytes(digest[:8], "big") % rows)
        nonce += 1
    return sorted(values)


def _read_tensor(snapshot: Path, index: dict[str, Any], key: str) -> torch.Tensor:
    with safe_open(str(snapshot / index["weight_map"][key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_HOSTILE_VERIFICATION" or protocol.get("neural_training_authorized") is not False:
        raise Phase3Error("lexical verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"lexical verifier binding changed: {relative}")
    artifact = load_file(str((root / protocol["artifact"]["path"]).resolve()), device="cpu")
    expected_keys = {"input_embedding_rows_fp16", "output_head_rows_fp16"}
    static = set(artifact) == expected_keys and all(value.shape == (32011, 192) and value.dtype == torch.float16 and bool(torch.isfinite(value).all()) for value in artifact.values())
    projection_cfg = protocol["projection"]
    generator = torch.Generator(device="cpu").manual_seed(int(projection_cfg["seed"]))
    projection = torch.randn(int(projection_cfg["source_width"]), int(projection_cfg["target_width"]), generator=generator, dtype=torch.float32)
    projection = (projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-12)).cuda()
    source = protocol["source"]
    snapshot = Path(source["snapshot_path"])
    index = json.loads(Path(source["index_path"]).read_text(encoding="utf-8"))
    chunk = int(protocol["chunk_rows"])
    comparisons = {}
    all_exact = True
    for artifact_key, source_key, target_norm in (
        ("input_embedding_rows_fp16", "model.embed_tokens.weight", math.sqrt(192)),
        ("output_head_rows_fp16", "lm_head.weight", 1.0 / math.sqrt(3.0)),
    ):
        source_tensor = _read_tensor(snapshot, index, source_key)
        rows = selected_rows(artifact_key, int(protocol["sample_rows_per_table"]), 32011)
        exact_scalars = 0
        maximum_error = 0.0
        for chunk_start in sorted({(row // chunk) * chunk for row in rows}):
            stop = min(32011, chunk_start + chunk)
            recomputed = project_rows(source_tensor[chunk_start:stop].cuda(), projection, target_norm).cpu()
            for row in [value for value in rows if chunk_start <= value < stop]:
                expected = artifact[artifact_key][row]
                observed = recomputed[row - chunk_start]
                difference = (expected.float() - observed.float()).abs()
                maximum_error = max(maximum_error, float(difference.max()))
                exact_scalars += int(expected.eq(observed).sum())
        total = len(rows) * 192
        table_exact = exact_scalars == total and maximum_error == 0.0
        all_exact &= table_exact
        comparisons[artifact_key] = {"selected_rows": rows, "scalars": total, "exact_scalars": exact_scalars, "maximum_absolute_error": maximum_error, "exact": table_exact}
    norms = {key: {"minimum": float(value.float().norm(dim=1).min()), "median": float(value.float().norm(dim=1).median()), "maximum": float(value.float().norm(dim=1).max())} for key, value in artifact.items()}
    norm_gate = abs(norms["input_embedding_rows_fp16"]["median"] - math.sqrt(192)) <= 0.003 and abs(norms["output_head_rows_fp16"]["median"] - 1.0 / math.sqrt(3.0)) <= 0.0002
    passed = static and all_exact and norm_gate
    return {"format": "abi-capability-compiler-phase3-lexical-substrate-verification/1", "status": "PASS_VERIFIED" if passed else "FAIL_VERIFICATION", "static_checks_pass": static, "norm_gate_pass": norm_gate, "norms": norms, "recomputation": comparisons, "sampled_scalars": sum(value["scalars"] for value in comparisons.values()), "exact_scalars": sum(value["exact_scalars"] for value in comparisons.values()), "maximum_absolute_error": max(value["maximum_absolute_error"] for value in comparisons.values()), "teacher_model_loaded": False, "teacher_inference_performed": False, "neural_training_performed": False, "phase3_certified": False, "final_test_accessed": False, "next_gate": "Preregister one bridge-only candidate with both imported tables frozen and post-training identity verification." if passed else "Reject lexical substrate."
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_LEXICAL_SUBSTRATE_VERIFIER_PROTOCOL_V84.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_lexical_substrate/verification_v84.json")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise Phase3Error("lexical verification output exists")
    result = run(root, (root / args.protocol).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
