"""Diagnose public R30 capabilities from foreign hidden representations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from experiments.factual_semantic_r16.public_qualification import _load_source, _render_chat
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once
from .acquire import MODEL_ID, REVISION
from .diagnose_v2 import _catalog


SYSTEM = (
    "Attend only to the abstract language operation requested by the instruction. "
    "Ignore subject matter, names, and numbers."
)
LAYERS = (4, 8, 12, 16, 20, 24, 28)
POOLS = ("mean", "last")


@torch.inference_mode()
def _represent(tokenizer, model, instruction: str) -> dict[str, torch.Tensor]:
    rendered = _render_chat(tokenizer, SYSTEM, instruction)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
    output = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    attention = encoded["attention_mask"].to(dtype=torch.bool)[0]
    result = {}
    for layer in LAYERS:
        hidden = output.hidden_states[layer][0].float()
        result[f"layer{layer}_mean"] = hidden[attention].mean(dim=0).cpu().contiguous()
        result[f"layer{layer}_last"] = hidden[-1].cpu().contiguous()
    return result


def _clusters(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    model = KMeans(n_clusters=12, random_state=30_003, n_init=20, algorithm="lloyd")
    labels = model.fit_predict(normalized)
    return labels, float(silhouette_score(normalized, labels, metric="cosine"))


def run(v2: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"immutable R30 v3 diagnosis exists: {output}")
    v2_result = v2 / "result.json"
    if not v2_result.is_file():
        raise RuntimeError("R30 v2 result missing")
    prior = json.loads(v2_result.read_text(encoding="utf-8"))
    if prior.get("verdict") != "FAIL_DIAGNOSIS":
        raise RuntimeError("R30 v3 requires the preserved v2 diagnosis failure")
    catalog, oracle = _catalog()
    ids = [item["id"] for item in catalog]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(MODEL_ID, REVISION)
    representations: dict[str, list[torch.Tensor]] = {
        f"layer{layer}_{pool}": [] for layer in LAYERS for pool in POOLS
    }
    input_tokens = 0
    for index, item in enumerate(catalog, 1):
        rendered = _render_chat(tokenizer, SYSTEM, item["instruction"])
        input_tokens += len(tokenizer.encode(rendered, add_special_tokens=False))
        current = _represent(tokenizer, model, item["instruction"])
        for key, value in current.items():
            representations[key].append(value)
        if index == 1 or index % 6 == 0:
            print(json.dumps({"represented": index, "seconds": time.perf_counter() - started}), flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    output.mkdir(parents=True)
    tensors = {key: torch.stack(value) for key, value in representations.items()}
    tensor_path = output / "instruction_representations.safetensors"
    save_file(tensors, str(tensor_path), metadata={"source_revision": REVISION})
    candidates = []
    assignments: dict[str, np.ndarray] = {}
    for key, tensor in tensors.items():
        labels, silhouette = _clusters(tensor.numpy())
        assignments[key] = labels
        candidates.append({"view": key, "silhouette": silhouette})
    candidates.sort(key=lambda item: (-item["silhouette"], item["view"]))
    selected = candidates[0]["view"]
    labels = assignments[selected]
    groups = []
    pure = 0
    sizes = []
    for label in sorted(set(int(item) for item in labels)):
        positions = [index for index, item in enumerate(labels) if int(item) == label]
        group_ids = [ids[index] for index in positions]
        tasks = [oracle[item] for item in group_ids]
        is_pure = len(set(tasks)) == 1
        pure += int(is_pure)
        sizes.append(len(group_ids))
        opaque_name = "capability-" + hashlib.sha256("|".join(sorted(group_ids)).encode()).hexdigest()[:12]
        groups.append({"name": opaque_name, "ids": group_ids, "oracle_tasks": tasks, "oracle_pure": is_pure})
    passed = len(groups) == 12 and pure == 12 and sorted(sizes) == [3] * 12
    result = {
        "format": "abi-r30-foreign-representation-diagnosis/3",
        "verdict": "PASS_PUBLIC_NEURAL_DIAGNOSIS" if passed else "FAIL_PUBLIC_NEURAL_DIAGNOSIS",
        "v2_result_sha256": sha256_file(v2_result),
        "selection": {"rule": "maximum_unsupervised_cosine_silhouette", "selected_view": selected, "candidates": candidates},
        "metrics": {"instructions": len(ids), "clusters": len(groups), "pure_clusters": pure, "cluster_sizes": sizes, "full_coverage": sum(sizes) == len(ids)},
        "groups": groups,
        "source": {"model_id": MODEL_ID, "revision": REVISION, "snapshot": str(snapshot), "training_steps": 0},
        "information_accounting": {"source_calls": len(ids), "source_input_tokens": input_tokens, "teacher_generated_tokens": 0, "hidden_activations_stored": sum(t.numel() for t in tensors.values()), "hidden_activation_bytes": tensor_path.stat().st_size, "source_parameters_copied": 0, "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())},
        "artifacts": {"representations": {"path": tensor_path.name, "sha256": sha256_file(tensor_path), "bytes": tensor_path.stat().st_size}},
        "claim_ceiling": "DISCLOSED_PUBLIC_INSTRUCTION_CLUSTERING_NOT_HELDOUT_ENGLISH",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.v2.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
