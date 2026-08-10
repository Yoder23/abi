"""Leakage-free analytic three-route classifier from source-prompt embeddings only."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-source-prompt-router-audit/1"


def _route(capability: str, specialists: tuple[str, ...]) -> str:
    return capability if capability in specialists else "generic"


def _solve_dual_ridge(
    features: torch.Tensor, targets: torch.Tensor, relative_ridge: float
) -> tuple[torch.Tensor, float]:
    if features.shape[0] != targets.shape[0]:
        raise Phase3Error("router feature/target observations changed")
    gram = features @ features.transpose(0, 1)
    scale = float(torch.trace(gram) / gram.shape[0])
    ridge = relative_ridge * scale
    dual = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype),
        targets,
    )
    return features.transpose(0, 1) @ dual, ridge


def _prompt_feature(embedding: torch.Tensor, source_ids: list[int]) -> torch.Tensor:
    if not source_ids:
        raise Phase3Error("empty source prompt")
    feature = embedding[torch.tensor(source_ids, dtype=torch.long)].float().mean(dim=0)
    feature = feature / torch.linalg.vector_norm(feature).clamp_min(1e-8)
    return torch.cat((feature, torch.ones(1, dtype=feature.dtype)))


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SOURCE_PROMPT_ONLY_ANALYTIC_ROUTER_AUDIT"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("target_token_access_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("source-prompt router governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"source-prompt router binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")

    output.mkdir(parents=True)
    device = torch.device("cuda")
    set_determinism(int(protocol["seed"]))
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    _, tokenizer_type = sequential._types(root, base)
    tokenizer = sequential.field._tokenizer(base, tokenizer_type)
    examples = sequential.field._examples(root, base, tokenizer)
    cfg = base["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    example_by_id = {str(row["record_id"]): row for row in examples}
    substrate = load_file(str(root / base["substrate"]["path"]), device="cpu")
    embedding = substrate["token_embedding.weight"]
    specialists = tuple(str(value) for value in protocol["specialist_routes"])
    route_names = ("generic",) + specialists
    route_index = {name: index for index, name in enumerate(route_names)}

    train_features = torch.stack(
        [
            _prompt_feature(embedding, list(example_by_id[str(row["record_id"])]["source_ids"]))
            for row in train_rows
        ]
    )
    train_labels = [
        _route(str(row["capability"]), specialists) for row in train_rows
    ]
    counts = Counter(train_labels)
    targets = torch.zeros(len(train_rows), len(route_names), dtype=torch.float32)
    weights = torch.empty(len(train_rows), dtype=torch.float32)
    for index, label in enumerate(train_labels):
        targets[index, route_index[label]] = 1.0
        weights[index] = len(train_rows) / (len(route_names) * counts[label])
    root_weights = torch.sqrt(weights).unsqueeze(1)
    weighted_features = (train_features * root_weights).to(device)
    weighted_targets = (targets * root_weights).to(device)
    solution, effective_ridge = _solve_dual_ridge(
        weighted_features, weighted_targets, float(protocol["relative_ridge"])
    )

    validation_features = torch.stack(
        [
            _prompt_feature(embedding, list(example_by_id[str(row["record_id"])]["source_ids"]))
            for row in validation_rows
        ]
    ).to(device)
    scores = validation_features @ solution
    predictions = scores.argmax(dim=1).cpu().tolist()
    rows = []
    exact = 0
    confusion: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in route_names} for expected in route_names
    }
    for row, predicted_index, row_scores in zip(validation_rows, predictions, scores.cpu()):
        expected = _route(str(row["capability"]), specialists)
        predicted = route_names[predicted_index]
        exact += int(expected == predicted)
        confusion[expected][predicted] += 1
        rows.append(
            {
                "record_id": row["record_id"],
                "capability": row["capability"],
                "expected_route": expected,
                "predicted_route": predicted,
                "scores": {name: float(row_scores[index]) for index, name in enumerate(route_names)},
                "exact": expected == predicted,
            }
        )
    exact_required = int(protocol["gate"]["exact_validation_routes_required"])
    passed = exact == exact_required == len(validation_rows)
    checkpoint = {
        "router.weight": solution[:-1].transpose(0, 1).detach().to(torch.float16).cpu().contiguous(),
        "router.bias": solution[-1].detach().to(torch.float16).cpu().contiguous(),
    }
    checkpoint_path = output / "source_prompt_router.safetensors"
    save_file(checkpoint, str(checkpoint_path), metadata={"format": FORMAT, "protocol_sha256": sha256_file(protocol_path)})
    oracle = json.loads((root / protocol["oracle_metadata"]["path"]).read_text(encoding="utf-8"))
    inherited = oracle["validation"] if passed else None
    result = {
        "format": FORMAT,
        "status": "PASS_EXACT_SOURCE_PROMPT_ROUTER" if passed else "FAIL_SOURCE_PROMPT_ROUTER",
        "protocol_sha256": sha256_file(protocol_path),
        "teacher_model_loaded": False,
        "target_tokens_accessed": False,
        "source_prompt_features": "mean_l2_normalized_copied_token_embeddings_plus_bias",
        "class_balancing": "inverse_route_frequency",
        "effective_ridge": effective_ridge,
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "calibration_tokens_not_used_by_router": calibration_tokens,
        "exact_validation_routes": exact,
        "exact_validation_routes_required": exact_required,
        "route_accuracy": exact / len(validation_rows),
        "confusion": confusion,
        "record_routes": rows,
        "checkpoint": {"path": checkpoint_path.name, "sha256": sha256_file(checkpoint_path), "parameters": sum(value.numel() for value in checkpoint.values())},
        "oracle_downstream_validation_inherited": inherited,
        "artifact_promoted": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Held-out source-prompt router audit and exact-route oracle inheritance only; no host, full artifact, English quality, physical runtime, certificate, or superiority claim.",
    }
    _write_immutable(output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_SOURCE_PROMPT_ROUTER_AUDIT_PROTOCOL_V301.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_three_route/router_audit_v302")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
