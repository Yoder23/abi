"""Run matched R8 baseline shards and consolidate their raw observations."""

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

from .capability_generator import (
    OpaqueCapability,
    canonical_json_bytes,
    generate_rows,
    public_capabilities,
)
from .native_host import (
    SPECS,
    FrozenNeuralHost,
    NativeHostError,
    sha256_file,
    tensor_state_sha256,
)
from .recipient_worker import _disable_network, _json, _jsonl


class BaselineError(RuntimeError):
    """Raised when a matched baseline is incomplete or not reproducible."""


class DenseLinearBridge(torch.nn.Module):
    """Unstructured linear latent-to-prefix baseline."""

    def __init__(self, input_width: int, prefix_length: int, hidden_width: int) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_width = hidden_width
        self.projection = torch.nn.Linear(input_width, prefix_length * hidden_width)
        torch.nn.init.normal_(self.projection.weight, mean=0.0, std=0.002)
        torch.nn.init.zeros_(self.projection.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        value = self.projection(latent.float().flatten(1))
        return value.reshape(-1, self.prefix_length, self.hidden_width)


class LoRALinear(torch.nn.Module):
    """Minimal target-specific LoRA wrapper around one frozen linear layer."""

    def __init__(self, base: torch.nn.Linear, *, rank: int) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.a = torch.nn.Parameter(torch.empty(rank, base.in_features))
        self.b = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        self.reset()

    def reset(self) -> None:
        torch.nn.init.normal_(self.a, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.b)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        adapted = F.linear(F.linear(inputs.float(), self.a.float()), self.b.float())
        return base + adapted.to(base.dtype) / self.rank


def _capabilities(path: Path) -> list[OpaqueCapability]:
    value = _json(path)
    return [
        OpaqueCapability(
            capability_id=str(row["capability_id"]),
            offsets=tuple(int(item) for item in row["offsets"]),
            seed_commitment=str(row["seed_commitment"]),
        )
        for row in value["capabilities"]
    ]


def _fit_prefix(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    teacher_probabilities: Mapping[str, Sequence[float]] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = random.Random(seed)
    torch.manual_seed(seed)
    prefix = torch.nn.Parameter(
        torch.empty(24, host.hidden_width, dtype=torch.float32, device=host.device)
    )
    torch.nn.init.normal_(prefix, mean=0.0, std=0.02)
    optimizer = torch.optim.AdamW([prefix], lr=learning_rate, weight_decay=0.0)
    started = time.perf_counter()
    for _ in range(steps):
        batch = [rows[generator.randrange(len(rows))] for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=prefix)
        if teacher_probabilities is None:
            targets = host.target_ids([int(row["answer"]) for row in batch])
            loss = F.cross_entropy(logits, targets)
        else:
            indices = torch.tensor(host.target_token_ids, dtype=torch.long, device=host.device)
            selected = logits.index_select(-1, indices)
            teacher = torch.tensor(
                [teacher_probabilities[str(row["row_id"])] for row in batch],
                dtype=torch.float32,
                device=host.device,
            )
            loss = F.kl_div(F.log_softmax(selected, dim=-1), teacher, reduction="batchmean")
        if not torch.isfinite(loss):
            raise BaselineError("target-specific prefix baseline became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([prefix], 1.0)
        optimizer.step()
    return prefix.detach(), {
        "optimizer_steps": steps,
        "wall_seconds": time.perf_counter() - started,
    }


def _fit_dense_linear(
    host: FrozenNeuralHost,
    config: Mapping[str, Any],
    meta_latents: torch.Tensor,
) -> tuple[DenseLinearBridge, DenseLinearBridge, dict[str, Any]]:
    meta_capabilities = public_capabilities(
        int(config["splits"]["meta_seed"]),
        split="meta_train",
        count=int(config["splits"]["meta_train_capabilities"]),
    )
    rows = [
        generate_rows(
            capability,
            split="linear_baseline_meta",
            rows=int(config["splits"]["source_train_rows_per_capability"]),
            depths=config["capability_family"]["source_train_depths"],
            seed=int(config["training"]["seed"]) + 3109 * index,
        )
        for index, capability in enumerate(meta_capabilities)
    ]
    seed = int(config["training"]["seed"]) + 8081
    torch.manual_seed(seed)
    learned = DenseLinearBridge(192, 24, host.hidden_width).to(host.device)
    random_projection = DenseLinearBridge(192, 24, host.hidden_width).to(host.device)
    random_projection.load_state_dict(learned.state_dict())
    for parameter in random_projection.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        learned.parameters(),
        lr=float(config["training"]["bridge_learning_rate"]),
        weight_decay=0.0,
    )
    generator = random.Random(seed)
    latents = meta_latents.to(host.device)
    steps = int(config["training"]["baseline_linear_meta_steps"])
    batch_size = int(config["training"]["batch_size"])
    started = time.perf_counter()
    for _ in range(steps):
        indices = [generator.randrange(len(rows)) for _ in range(batch_size)]
        selected = [rows[index][generator.randrange(len(rows[index]))] for index in indices]
        index_tensor = torch.tensor(indices, dtype=torch.long, device=host.device)
        optimizer.zero_grad(set_to_none=True)
        prefix = learned(latents.index_select(0, index_tensor))
        logits, _ = host.logits([str(row["prompt"]) for row in selected], prefix=prefix)
        targets = host.target_ids([int(row["answer"]) for row in selected])
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise BaselineError("linear mapping baseline became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learned.parameters(), 1.0)
        optimizer.step()
    learned.eval()
    for parameter in learned.parameters():
        parameter.requires_grad_(False)
    return learned, random_projection, {
        "optimizer_steps": steps,
        "wall_seconds": time.perf_counter() - started,
        "learned_parameters": sum(value.numel() for value in learned.parameters()),
    }


def _find_lora_target(model: torch.nn.Module) -> tuple[str, torch.nn.Linear]:
    priorities = ("q_proj", "query_key_value", ".q")
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    for suffix in priorities:
        for name, module in candidates:
            if name.endswith(suffix):
                return name, module
    if not candidates:
        raise BaselineError("recipient exposes no linear layer for LoRA")
    return candidates[0]


def _replace_module(model: torch.nn.Module, name: str, replacement: torch.nn.Module) -> None:
    parent = model
    pieces = name.split(".")
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    setattr(parent, pieces[-1], replacement)


def _augmented_prompts(
    capability: OpaqueCapability,
    evaluation: Sequence[Mapping[str, Any]],
    training: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> list[str]:
    if method == "natural_language_context_equal_byte":
        context = (
            f"Hidden rule: vok adds {capability.offsets[0]}, narel adds "
            f"{capability.offsets[1]}, tem adds {capability.offsets[2]}, all modulo 8. "
        )
    elif method == "retrieval_in_context_equal_byte":
        examples = training[:3]
        context = "Examples: " + " ".join(
            f"[{row['prompt']} {row['answer']}]" for row in examples
        ) + " "
    else:
        raise BaselineError(f"unknown context baseline: {method}")
    return [context + str(row["prompt"]) for row in evaluation]


@torch.inference_mode()
def _observe(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[str],
    prefix: torch.Tensor | None,
    *,
    host_key: str,
    capability_id: str,
    method: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    output = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        logits, _ = host.logits(batch_prompts, prefix=prefix)
        predicted = logits.argmax(dim=-1)
        probabilities = host.canonical_probabilities(logits)
        for row, token, probability in zip(batch_rows, predicted, probabilities):
            output.append(
                {
                    "host": host_key,
                    "capability_id": capability_id,
                    "method": method,
                    "row_id": row["row_id"],
                    "prediction_token_id": int(token.item()),
                    "canonical_output_probabilities": [
                        float(value) for value in probability.cpu().tolist()
                    ],
                }
            )
    return output


def _teacher_probabilities(
    config: Mapping[str, Any],
    campaign_root: Path,
    capabilities: Sequence[OpaqueCapability],
    train_rows: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, list[float]]]:
    source_dir = campaign_root / "heldout_source"
    source_receipt = _json(source_dir / "receipt.json")
    prefixes = load_file(
        str(source_dir / source_receipt["prefixes"]["path"]), device="cpu"
    )["after"]
    source = FrozenNeuralHost(SPECS["source"], device=config["training"]["device"])
    result = []
    batch_size = int(config["training"]["batch_size"])
    with torch.inference_mode():
        for index, (capability, rows) in enumerate(zip(capabilities, train_rows)):
            values = {}
            prefix = prefixes[index].to(source.device)
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                logits, _ = source.logits([str(row["prompt"]) for row in batch], prefix=prefix)
                probabilities = source.canonical_probabilities(logits)
                for row, probability in zip(batch, probabilities):
                    values[str(row["row_id"])] = [
                        float(value) for value in probability.cpu().tolist()
                    ]
            result.append(values)
    del source
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_host(
    config_path: Path,
    campaign_root: Path,
    output: Path,
    *,
    host_key: str,
) -> dict[str, Any]:
    if output.exists():
        raise BaselineError(f"immutable baseline shard exists: {output}")
    config = _json(config_path)
    capabilities = _capabilities(campaign_root / "evaluator_private/capabilities.json")
    train_rows = [
        generate_rows(
            capability,
            split="source_train",
            rows=int(config["splits"]["source_train_rows_per_capability"]),
            depths=config["capability_family"]["source_train_depths"],
            seed=int(config["training"]["seed"]) + 8009 * index,
        )
        for index, capability in enumerate(capabilities)
    ]
    teacher = _teacher_probabilities(config, campaign_root, capabilities, train_rows)
    host = FrozenNeuralHost(SPECS[host_key], device=config["training"]["device"])
    _disable_network()
    extraction_receipt = _json(campaign_root / "pre_reveal/meta_extraction/receipt.json")
    meta_tensors = load_file(
        str(
            campaign_root
            / "pre_reveal/meta_extraction"
            / extraction_receipt["latents"]["path"]
        ),
        device="cpu",
    )
    learned_linear, random_projection, linear_receipt = _fit_dense_linear(
        host, config, meta_tensors["meta_after"].float()
    )
    observations = []
    checkpoints: dict[str, torch.Tensor] = {}
    training_receipts = []
    batch_size = int(config["training"]["batch_size"])
    steps = int(config["training"]["baseline_target_steps"])
    learning_rate = float(config["training"]["baseline_target_learning_rate"])
    heldout_latents = {}
    from .recipient_worker import _load_package

    for capability in capabilities:
        _, latent = _load_package(
            campaign_root / "heldout_source/packages" / capability.capability_id / "after.abipkg"
        )
        heldout_latents[capability.capability_id] = latent
    started = time.perf_counter()
    for index, capability in enumerate(capabilities):
        evaluation = _jsonl(
            campaign_root / "heldout_source/worker_inputs" / f"{capability.capability_id}.jsonl"
        )
        plain_prompts = [str(row["prompt"]) for row in evaluation]
        for method in (
            "natural_language_context_equal_byte",
            "retrieval_in_context_equal_byte",
        ):
            observations.extend(
                _observe(
                    host,
                    evaluation,
                    _augmented_prompts(capability, evaluation, train_rows[index], method=method),
                    None,
                    host_key=host_key,
                    capability_id=capability.capability_id,
                    method=method,
                    batch_size=batch_size,
                )
            )
        soft, soft_receipt = _fit_prefix(
            host,
            train_rows[index],
            steps=steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=int(config["training"]["seed"]) + 40009 * index,
        )
        distill, distill_receipt = _fit_prefix(
            host,
            train_rows[index],
            steps=steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=int(config["training"]["seed"]) + 50021 * index,
            teacher_probabilities=teacher[index],
        )
        checkpoints[f"soft_prompt_{index}"] = soft.cpu().contiguous()
        checkpoints[f"distilled_prompt_{index}"] = distill.cpu().contiguous()
        for method, prefix in (
            ("target_specific_soft_prompt", soft),
            ("teacher_to_target_distillation", distill),
            (
                "linear_mapping",
                learned_linear(heldout_latents[capability.capability_id].to(host.device)).squeeze(0),
            ),
            (
                "random_projection",
                random_projection(
                    heldout_latents[capability.capability_id].to(host.device)
                ).squeeze(0),
            ),
        ):
            observations.extend(
                _observe(
                    host,
                    evaluation,
                    plain_prompts,
                    prefix,
                    host_key=host_key,
                    capability_id=capability.capability_id,
                    method=method,
                    batch_size=batch_size,
                )
            )
        training_receipts.append(
            {
                "capability_id": capability.capability_id,
                "soft_prompt": soft_receipt,
                "distillation": distill_receipt,
            }
        )

    target_name, target_layer = _find_lora_target(host.model)
    target_weight_before = hashlib.sha256(
        target_layer.weight.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    lora = LoRALinear(target_layer, rank=int(config["training"]["baseline_lora_rank"])).to(
        host.device
    )
    _replace_module(host.model, target_name, lora)
    for index, capability in enumerate(capabilities):
        seed = int(config["training"]["seed"]) + 60013 * index
        torch.manual_seed(seed)
        lora.reset()
        optimizer = torch.optim.AdamW((lora.a, lora.b), lr=learning_rate, weight_decay=0.0)
        generator = random.Random(seed)
        lora_started = time.perf_counter()
        for _ in range(steps):
            batch = [train_rows[index][generator.randrange(len(train_rows[index]))] for _ in range(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
            targets = host.target_ids([int(row["answer"]) for row in batch])
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise BaselineError("target-specific LoRA became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((lora.a, lora.b), 1.0)
            optimizer.step()
        evaluation = _jsonl(
            campaign_root / "heldout_source/worker_inputs" / f"{capability.capability_id}.jsonl"
        )
        observations.extend(
            _observe(
                host,
                evaluation,
                [str(row["prompt"]) for row in evaluation],
                None,
                host_key=host_key,
                capability_id=capability.capability_id,
                method="target_specific_lora",
                batch_size=batch_size,
            )
        )
        checkpoints[f"lora_a_{index}"] = lora.a.detach().cpu().contiguous()
        checkpoints[f"lora_b_{index}"] = lora.b.detach().cpu().contiguous()
        training_receipts[index]["lora"] = {
            "optimizer_steps": steps,
            "wall_seconds": time.perf_counter() - lora_started,
            "rank": lora.rank,
        }
    target_weight_after = hashlib.sha256(
        lora.base.weight.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    checkpoints.update(
        {
            "linear_" + name: value.detach().cpu().contiguous()
            for name, value in learned_linear.state_dict().items()
        }
    )
    checkpoints.update(
        {
            "random_projection_" + name: value.detach().cpu().contiguous()
            for name, value in random_projection.state_dict().items()
        }
    )
    output.mkdir(parents=True)
    checkpoint_path = output / "baseline_parameters.safetensors"
    save_file(checkpoints, str(checkpoint_path))
    raw_path = output / "observations.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in observations))
    manifest = {
        "format": "abi-native-transfer-r8-baseline-shard/1",
        "host": host_key,
        "architecture_family": host.spec.architecture_family,
        "revision": host.spec.revision,
        "config_sha256": sha256_file(config_path),
        "methods": list(config["required_baselines"]),
        "target_token_ids": host.target_token_ids,
        "linear_mapping": linear_receipt,
        "target_specific_training": training_receipts,
        "lora_target_layer": target_name,
        "lora_base_weight_sha256_before": target_weight_before,
        "lora_base_weight_sha256_after": target_weight_after,
        "parameter_artifact_sha256": sha256_file(checkpoint_path),
        "parameter_state_sha256": tensor_state_sha256(checkpoints),
        "observations_sha256": sha256_file(raw_path),
        "rows": len(observations),
        "wall_seconds": time.perf_counter() - started,
    }
    manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    (output / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return manifest


def consolidate(config_path: Path, campaign_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise BaselineError(f"immutable consolidated baseline output exists: {output}")
    config = _json(config_path)
    all_rows = []
    shards = {}
    target_ids = {}
    for host in sorted(config["models"]["recipients"]):
        shard = campaign_root / "baselines/shards" / host
        manifest = _json(shard / "manifest.json")
        payload = dict(manifest)
        stored = payload.pop("evidence_sha256", None)
        if stored != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
            raise BaselineError(f"stale baseline shard: {host}")
        raw_path = shard / "observations.jsonl"
        if manifest["observations_sha256"] != sha256_file(raw_path):
            raise BaselineError(f"baseline shard raw rows changed: {host}")
        rows = _jsonl(raw_path)
        all_rows.extend(rows)
        shards[host] = {
            "manifest_sha256": sha256_file(shard / "manifest.json"),
            "observations_sha256": sha256_file(raw_path),
            "rows": len(rows),
        }
        target_ids[host] = manifest["target_token_ids"]
    output.mkdir(parents=True)
    raw_path = output / "observations.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in all_rows))
    manifest = {
        "format": "abi-native-transfer-r8-baselines/1",
        "config_sha256": sha256_file(config_path),
        "methods": list(config["required_baselines"]),
        "target_token_ids": target_ids,
        "shards": shards,
        "observations_sha256": sha256_file(raw_path),
        "rows": len(all_rows),
    }
    manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    (output / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", choices=("qwen2", "pythia", "t5"))
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    if not args.consolidate and args.host is None:
        parser.error("--host is required unless --consolidate is used")
    try:
        value = (
            consolidate(
                Path(args.config).resolve(),
                Path(args.campaign_root).resolve(),
                Path(args.output).resolve(),
            )
            if args.consolidate
            else run_host(
                Path(args.config).resolve(),
                Path(args.campaign_root).resolve(),
                Path(args.output).resolve(),
                host_key=str(args.host),
            )
        )
    except (BaselineError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
