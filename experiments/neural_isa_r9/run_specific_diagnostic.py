"""Run R9 Gate A: a capability-specific Pythia realization diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    generate_rows,
    public_capabilities,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    module_sha256,
    sha256_file,
)
from experiments.native_transfer_r8.recipient_worker import (
    _random_latent,
    _shuffled_latent,
)

from .backend import PackageConditionedGRUBackend


class R9DiagnosticError(RuntimeError):
    """Raised when Gate A inputs, execution, or evidence are inadmissible."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R9DiagnosticError(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise R9DiagnosticError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise R9DiagnosticError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise R9DiagnosticError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise R9DiagnosticError(f"registered path escapes repository: {value}") from exc
    return path


def _bind_r8(root: Path, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    reference = config["r8_reference"]
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("extraction_receipt", "extraction_receipt_sha256"),
        ("canonical_latents", "canonical_latents_sha256"),
    ):
        path = _resolve(root, str(reference[path_key]))
        if not path.is_file() or sha256_file(path) != str(reference[hash_key]):
            raise R9DiagnosticError(f"registered R8 input changed: {path_key}")
    sealed = _resolve(root, str(reference["sealed_public_verification"]))
    sealed_value = _json(sealed)
    if sealed_value.get("exact_question_answer") != "NO" or sealed_value.get("verdict_level") != 0:
        raise R9DiagnosticError("sealed R8 Level 0 result changed")
    return _resolve(root, str(reference["canonical_latents"])), sealed_value


def _bind_implementation(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    registered = config.get("implementation")
    if registered is None:
        return {}
    if not isinstance(registered, dict) or not registered:
        raise R9DiagnosticError("implementation registration changed")
    result = {}
    for relative, expected in registered.items():
        path = _resolve(root, str(relative))
        actual = sha256_file(path)
        if actual != expected:
            raise R9DiagnosticError(f"registered implementation changed: {relative}")
        result[str(relative)] = actual
    return result


@torch.inference_mode()
def _recipient_features(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    state_layers: Sequence[str],
) -> list[torch.Tensor]:
    features: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [str(row["prompt"]) for row in batch]
        encoded = host.encode(prompts)
        lengths = encoded["attention_mask"].sum(dim=1).detach().cpu().tolist()
        _, output = host.logits(prompts, prefix=None, output_hidden_states=True)
        states = _select_states(output, state_layers).detach().float().cpu()
        for index, length in enumerate(lengths):
            features.append(states[index, : int(length)].contiguous())
    if len(features) != len(rows):
        raise R9DiagnosticError("recipient feature extraction depth changed")
    return features


def _select_states(output: Any, state_layers: Sequence[str]) -> torch.Tensor:
    if not state_layers or any(value not in {"embedding", "final"} for value in state_layers):
        raise R9DiagnosticError("recipient state-layer registration changed")
    selected = [
        output.hidden_states[0] if value == "embedding" else output.hidden_states[-1]
        for value in state_layers
    ]
    if any(value.shape[:2] != selected[0].shape[:2] for value in selected):
        raise R9DiagnosticError("recipient state sequence geometry changed")
    return torch.cat(selected, dim=-1)


def _batch_features(
    features: Sequence[torch.Tensor], indices: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [features[index] for index in indices]
    lengths = torch.tensor([value.shape[0] for value in selected], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(selected, batch_first=True)
    return padded.to(device), lengths.to(device)


def _teacher_distribution(latent: torch.Tensor, row: Mapping[str, Any]) -> torch.Tensor:
    state = F.one_hot(torch.tensor(int(row["start"])), num_classes=8).float()
    for operation in row["program"]:
        state = state @ latent[int(operation)].float()
    return state / state.sum().clamp_min(torch.finfo(torch.float32).tiny)


def _condition_packages(
    after: torch.Tensor,
    before: torch.Tensor,
    wrong: torch.Tensor,
    capability_id: str,
) -> dict[str, torch.Tensor | None]:
    return {
        "BASE": None,
        "AFTER": after,
        "BEFORE": before,
        "WRONG": wrong,
        "ZERO": torch.zeros_like(after),
        "RANDOM": _random_latent(after, capability_id + "-r9"),
        "SHUFFLED": _shuffled_latent(after, capability_id + "-r9"),
        "REMOVED": None,
    }


def _normalized_packages(packages: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    result = []
    for package in packages:
        value = package.float().clamp_min(0)
        denominator = value.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(value, 1.0 / 8.0)
        result.append(torch.where(denominator > 0, value / denominator.clamp_min(1e-12), uniform))
    return result


def _train_backend(
    backend: PackageConditionedGRUBackend,
    features: Sequence[torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
    *,
    after: torch.Tensor,
    negative_packages: Sequence[torch.Tensor],
    settings: Mapping[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], float]:
    generator = random.Random(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        backend.parameters(), lr=float(settings["learning_rate"]), weight_decay=0.0
    )
    steps = int(settings["steps"])
    batch_size = int(settings["batch_size"])
    negative_weight = float(settings["negative_control_weight"])
    clip = float(settings["gradient_clip"])
    after = after.to(next(backend.parameters()).device)
    negatives = [value.to(after.device) for value in _normalized_packages(negative_packages)]
    curves = []
    started = time.perf_counter()
    backend.train()
    for step in range(1, steps + 1):
        indices = [generator.randrange(len(rows)) for _ in range(batch_size)]
        states, lengths = _batch_features(features, indices, after.device)
        targets = torch.tensor(
            [int(rows[index]["answer"]) for index in indices],
            dtype=torch.long,
            device=after.device,
        )
        optimizer.zero_grad(set_to_none=True)
        positive = backend(states, lengths, after)
        positive_loss = F.cross_entropy(positive, targets)
        negative_package = negatives[generator.randrange(len(negatives))]
        negative = backend(states, lengths, negative_package)
        negative_loss = negative.square().mean()
        loss = positive_loss + negative_weight * negative_loss
        if not torch.isfinite(loss):
            raise R9DiagnosticError("backend loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backend.parameters(), clip)
        optimizer.step()
        if step == 1 or step % max(1, steps // 20) == 0 or step == steps:
            item = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "positive_loss": float(positive_loss.detach().cpu()),
                "negative_loss": float(negative_loss.detach().cpu()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
    return curves, time.perf_counter() - started


@torch.inference_mode()
def _evaluate(
    host: FrozenNeuralHost,
    backend: PackageConditionedGRUBackend,
    rows: Sequence[Mapping[str, Any]],
    conditions: Mapping[str, torch.Tensor | None],
    *,
    after: torch.Tensor,
    batch_size: int,
    residual_scale: float,
    state_layers: Sequence[str],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    backend.eval()
    target_ids = torch.tensor(host.target_token_ids, dtype=torch.long, device=host.device)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [str(row["prompt"]) for row in batch]
        encoded = host.encode(prompts)
        lengths = encoded["attention_mask"].sum(dim=1)
        base_logits, output = host.logits(prompts, prefix=None, output_hidden_states=True)
        states = _select_states(output, state_layers).detach().float()
        for condition, package in conditions.items():
            modified = base_logits.clone()
            if package is not None:
                residual = backend(states, lengths, package.to(host.device))
                modified[:, target_ids] += float(residual_scale) * residual
            probabilities = torch.softmax(modified.index_select(-1, target_ids), dim=-1)
            predictions = modified.argmax(dim=-1)
            for index, row in enumerate(batch):
                teacher = _teacher_distribution(after, row)
                tv = 0.5 * torch.abs(probabilities[index].cpu() - teacher).sum()
                observations.append(
                    {
                        "row_id": str(row["row_id"]),
                        "capability_id": str(row["capability_id"]),
                        "prompt_sha256": str(row["prompt_sha256"]),
                        "depth": int(row["depth"]),
                        "flavor": str(row["flavor"]),
                        "condition": condition,
                        "prediction_token_id": int(predictions[index].cpu()),
                        "canonical_output_probabilities": [
                            float(value) for value in probabilities[index].cpu()
                        ],
                        "teacher_canonical_probabilities": [float(value) for value in teacher],
                        "teacher_recipient_tv": float(tv),
                    }
                )
    return observations


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R9DiagnosticError(f"immutable output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = _json(config_path)
    if not str(config.get("status", "")).startswith("PREREGISTERED_BEFORE_GATE_A"):
        raise R9DiagnosticError("R9 preregistration status changed")
    latent_path, _ = _bind_r8(root, config)
    implementation = _bind_implementation(root, config)
    tensors = load_file(str(latent_path), device="cpu")
    gate = config["gate_a"]
    capability_index = int(gate["development_capability_index"])
    wrong_index = int(gate["wrong_capability_index"])
    r8_config = _json(_resolve(root, str(config["r8_reference"]["config"])))
    split = r8_config["splits"]
    capabilities = public_capabilities(
        int(split["development_seed"]),
        split="development",
        count=int(split["development_capabilities"]),
    )
    capability = capabilities[capability_index]
    after = tensors["development_after"][capability_index].float()
    before = tensors["before"].float()
    wrong = tensors["development_after"][wrong_index].float()
    settings = gate["backend"]
    state_layers = tuple(str(value) for value in settings.get("recipient_state_layers", ["final"]))
    seed = int(gate["seed"])
    train_rows = generate_rows(
        capability,
        split="r9_specific_train",
        rows=int(gate["train_rows"]),
        depths=gate["train_depths"],
        seed=seed + 1,
    )
    evaluation_rows = generate_rows(
        capability,
        split="r9_specific_evaluation",
        rows=int(gate["evaluation_rows"]),
        depths=gate["evaluation_depths"],
        seed=seed + 2,
    )
    if {row["row_id"] for row in train_rows} & {row["row_id"] for row in evaluation_rows}:
        raise R9DiagnosticError("train/evaluation rows overlap")

    torch.manual_seed(seed)
    host = FrozenNeuralHost(SPECS[str(gate["host"])], device="cuda")
    host_state_before = host.model_state_sha256
    backend = PackageConditionedGRUBackend(
        host.hidden_width * len(state_layers), hidden_width=int(settings["hidden_width"])
    ).to(host.device)
    train_features = _recipient_features(
        host,
        train_rows,
        batch_size=int(settings["batch_size"]),
        state_layers=state_layers,
    )
    conditions = _condition_packages(after, before, wrong, capability.capability_id)
    negative_packages = [
        value for name, value in conditions.items() if name not in {"BASE", "AFTER", "REMOVED"}
        and value is not None
    ]
    curves, training_seconds = _train_backend(
        backend,
        train_features,
        train_rows,
        after=after,
        negative_packages=negative_packages,
        settings=settings,
        seed=seed + 3,
    )
    output.mkdir(parents=True)
    backend_path = output / "capability_specific_backend.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in backend.state_dict().items()},
        str(backend_path),
    )
    backend_hash_before = module_sha256(backend)
    started = time.perf_counter()
    observations = _evaluate(
        host,
        backend,
        evaluation_rows,
        conditions,
        after=after,
        batch_size=int(settings["batch_size"]),
        residual_scale=float(settings["residual_scale"]),
        state_layers=state_layers,
    )
    evaluation_seconds = time.perf_counter() - started
    raw_path = output / "observations.jsonl"
    _write_jsonl_once(raw_path, observations)
    training_observations = _evaluate(
        host,
        backend,
        train_rows,
        {"AFTER": after},
        after=after,
        batch_size=int(settings["batch_size"]),
        residual_scale=float(settings["residual_scale"]),
        state_layers=state_layers,
    )
    training_raw_path = output / "training_observations.jsonl"
    _write_jsonl_once(training_raw_path, training_observations)
    backend_hash_after = module_sha256(backend)
    host.verify_frozen()
    host_state_after = module_sha256(host.model)
    if host_state_before != host_state_after or backend_hash_before != backend_hash_after:
        raise R9DiagnosticError("frozen model identity changed during evaluation")
    receipt = {
        "format": "abi-neural-isa-r9-capability-specific-diagnostic/1",
        "config_sha256": sha256_file(config_path),
        "r8_config_sha256": sha256_file(_resolve(root, str(config["r8_reference"]["config"]))),
        "r8_extraction_receipt_sha256": sha256_file(
            _resolve(root, str(config["r8_reference"]["extraction_receipt"]))
        ),
        "r8_canonical_latents_sha256": sha256_file(latent_path),
        "implementation_sha256": implementation,
        "host_key": host.spec.key,
        "host_model_id": host.spec.model_id,
        "host_revision": host.spec.revision,
        "host_model_state_sha256_before": host_state_before,
        "host_model_state_sha256_after": host_state_after,
        "target_token_ids": list(host.target_token_ids),
        "target_text": list(host.target_text),
        "capability_id": capability.capability_id,
        "capability_specific_weights_allowed": True,
        "universal_decoder_claim_allowed": False,
        "recipient_optimizer_steps": 0,
        "backend_optimizer_steps": int(settings["steps"]),
        "backend": {
            "path": backend_path.name,
            "sha256": sha256_file(backend_path),
            "state_sha256_before_evaluation": backend_hash_before,
            "state_sha256_after_evaluation": backend_hash_after,
            "parameters": sum(value.numel() for value in backend.parameters()),
            "kind": settings["kind"],
            "recipient_state_layers": list(state_layers),
        },
        "training": {
            "rows": len(train_rows),
            "depths": list(gate["train_depths"]),
            "wall_seconds": training_seconds,
            "curves": curves,
        },
        "evaluation": {
            "rows_per_condition": len(evaluation_rows),
            "conditions": list(conditions),
            "wall_seconds": evaluation_seconds,
        },
        "observations": {
            "path": raw_path.name,
            "sha256": sha256_file(raw_path),
            "rows": len(observations),
        },
        "training_observations": {
            "path": training_raw_path.name,
            "sha256": sha256_file(training_raw_path),
            "rows": len(training_observations),
        },
        "hardware": {
            "device": str(host.device),
            "cuda_device_name": torch.cuda.get_device_name(host.device),
            "cuda_total_memory_bytes": torch.cuda.get_device_properties(host.device).total_memory,
        },
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    _write_once(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = run(Path(args.config).resolve(), Path(args.output).resolve())
    except (OSError, ValueError, R9DiagnosticError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
