"""Meta-train and freeze a prompt-blind generic bridge for one recipient."""

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
    canonical_json_bytes,
    generate_composition_rows,
    generate_rows,
    public_capabilities,
)
from .native_host import (
    SPECS,
    CanonicalLatentBridge,
    FrozenNeuralHost,
    NativeHostError,
    build_bridge,
    module_sha256,
    sha256_file,
)


class MetaInterfaceError(RuntimeError):
    """Raised when bridge training observes held-out data or changes its host."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MetaInterfaceError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise MetaInterfaceError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


@torch.inference_mode()
def _evaluate(
    host: FrozenNeuralHost,
    bridge: CanonicalLatentBridge,
    latents: torch.Tensor,
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    result = []
    for index, capability_rows in enumerate(rows):
        correct = 0
        losses = 0.0
        latent = latents[index].to(host.device)
        prefix = bridge(latent).squeeze(0)
        for start in range(0, len(capability_rows), batch_size):
            batch = capability_rows[start : start + batch_size]
            logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=prefix)
            targets = host.target_ids([int(row["answer"]) for row in batch])
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            losses += float(F.cross_entropy(logits, targets, reduction="sum").item())
        result.append(
            {
                "capability_index": index,
                "rows": len(capability_rows),
                "correct": correct,
                "accuracy": correct / len(capability_rows),
                "mean_nll": losses / len(capability_rows),
            }
        )
    return result


@torch.inference_mode()
def _evaluate_composition(
    host: FrozenNeuralHost,
    bridge: CanonicalLatentBridge,
    latents: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    result = []
    for pair_index, ((first, second), pair_rows) in enumerate(zip(pairs, rows)):
        prefix = torch.cat(
            (
                bridge(latents[first].to(host.device)).squeeze(0),
                bridge(latents[second].to(host.device)).squeeze(0),
            ),
            dim=0,
        )
        correct = 0
        losses = 0.0
        for start in range(0, len(pair_rows), batch_size):
            batch = pair_rows[start : start + batch_size]
            logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=prefix)
            targets = host.target_ids([int(row["answer"]) for row in batch])
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            losses += float(F.cross_entropy(logits, targets, reduction="sum").item())
        result.append(
            {
                "pair_index": pair_index,
                "first_capability_index": first,
                "second_capability_index": second,
                "rows": len(pair_rows),
                "correct": correct,
                "accuracy": correct / len(pair_rows),
                "mean_nll": losses / len(pair_rows),
            }
        )
    return result


def train(
    config_path: Path,
    extraction_dir: Path,
    output: Path,
    *,
    host_key: str,
    campaign_root: Path,
) -> dict[str, Any]:
    config = _json(config_path)
    reveal_path = campaign_root / "heldout_reveal.json"
    package_paths = list(campaign_root.rglob("*.abipkg")) if campaign_root.exists() else []
    if output.exists() or reveal_path.exists() or package_paths:
        raise MetaInterfaceError(
            "bridge freeze requires a fresh output with held-out reveal and packages absent"
        )
    if host_key not in config["models"]["recipients"] or host_key not in SPECS:
        raise MetaInterfaceError(f"unregistered recipient: {host_key}")
    extraction_receipt = _json(extraction_dir / "receipt.json")
    latent_path = extraction_dir / extraction_receipt["latents"]["path"]
    if sha256_file(latent_path) != extraction_receipt["latents"]["sha256"]:
        raise MetaInterfaceError("meta latent identity changed")
    tensors = load_file(str(latent_path), device="cpu")
    meta_latents = tensors["meta_after"].float()
    development_latents = tensors["development_after"].float()
    split = config["splits"]
    meta_capabilities = public_capabilities(
        int(split["meta_seed"]),
        split="meta_train",
        count=int(split["meta_train_capabilities"]),
    )
    development_capabilities = public_capabilities(
        int(split["development_seed"]),
        split="development",
        count=int(split["development_capabilities"]),
    )
    train_rows = [
        generate_rows(
            capability,
            split="bridge_meta_train",
            rows=int(split["source_train_rows_per_capability"]),
            depths=config["training"].get(
                "bridge_meta_train_depths",
                config["capability_family"]["source_train_depths"],
            ),
            seed=int(config["training"]["seed"]) + 2003 * index,
        )
        for index, capability in enumerate(meta_capabilities)
    ]
    development_rows = [
        generate_rows(
            capability,
            split="bridge_development",
            rows=256,
            depths=config["capability_family"]["evaluation_depths"],
            seed=int(config["training"]["seed"]) + 4001 * index,
        )
        for index, capability in enumerate(development_capabilities)
    ]
    meta_pairs = [
        (index, index + 1) for index in range(0, len(meta_capabilities) - 1, 2)
    ]
    meta_composition_rows = [
        generate_composition_rows(
            meta_capabilities[first],
            meta_capabilities[second],
            split="bridge_meta_composition",
            rows=int(split["source_train_rows_per_capability"]),
            first_depths=config["training"].get(
                "bridge_meta_composition_depths",
                config["capability_family"]["composition_train_depths"],
            ),
            second_depths=config["training"].get(
                "bridge_meta_composition_depths",
                config["capability_family"]["composition_train_depths"],
            ),
            seed=int(config["training"]["seed"]) + 6007 * pair_index,
        )
        for pair_index, (first, second) in enumerate(meta_pairs)
    ]
    development_pairs = [(0, 1), (2, 3)]
    development_composition_rows = [
        generate_composition_rows(
            development_capabilities[first],
            development_capabilities[second],
            split="bridge_development_composition",
            rows=256,
            first_depths=config["capability_family"]["composition_evaluation_depths"],
            second_depths=config["capability_family"]["composition_evaluation_depths"],
            seed=int(config["training"]["seed"]) + 7001 * pair_index,
        )
        for pair_index, (first, second) in enumerate(development_pairs)
    ]
    host = FrozenNeuralHost(SPECS[host_key], device=config["training"]["device"])
    bridge = build_bridge(host, config).to(host.device).train()
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(config["training"]["bridge_learning_rate"]),
        weight_decay=0.0,
    )
    host_seed = int.from_bytes(hashlib.sha256(host_key.encode("ascii")).digest()[:4], "big")
    generator = random.Random(int(config["training"]["seed"]) + host_seed)
    batch_size = int(config["training"]["batch_size"])
    steps = int(config["training"]["bridge_steps"])
    composition_fraction = float(config["training"]["bridge_composition_step_fraction"])
    composition_interval = round(1.0 / composition_fraction)
    if composition_fraction <= 0 or composition_fraction > 0.5 or composition_interval < 2:
        raise MetaInterfaceError("invalid preregistered composition step fraction")
    meta_latents_device = meta_latents.to(host.device)
    curves = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        composition_step = step % composition_interval == 0
        if composition_step:
            pair_indices = [generator.randrange(len(meta_pairs)) for _ in range(batch_size)]
            selected = [
                meta_composition_rows[index][
                    generator.randrange(len(meta_composition_rows[index]))
                ]
                for index in pair_indices
            ]
            first_indices = torch.tensor(
                [meta_pairs[index][0] for index in pair_indices],
                dtype=torch.long,
                device=host.device,
            )
            second_indices = torch.tensor(
                [meta_pairs[index][1] for index in pair_indices],
                dtype=torch.long,
                device=host.device,
            )
            prefix = torch.cat(
                (
                    bridge(meta_latents_device.index_select(0, first_indices)),
                    bridge(meta_latents_device.index_select(0, second_indices)),
                ),
                dim=1,
            )
        else:
            indices = [generator.randrange(len(meta_capabilities)) for _ in range(batch_size)]
            selected = [
                train_rows[index][generator.randrange(len(train_rows[index]))]
                for index in indices
            ]
            index_tensor = torch.tensor(indices, dtype=torch.long, device=host.device)
            prefix = bridge(meta_latents_device.index_select(0, index_tensor))
        logits, _ = host.logits([str(row["prompt"]) for row in selected], prefix=prefix)
        targets = host.target_ids([int(row["answer"]) for row in selected])
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise MetaInterfaceError("bridge loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, steps // 20) == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "wall_seconds": time.perf_counter() - started,
                "composition_step": composition_step,
            }
            curves.append(row)
            print(json.dumps(row), flush=True)
    development = _evaluate(
        host,
        bridge,
        development_latents,
        development_rows,
        batch_size=batch_size,
    )
    development_composition = _evaluate_composition(
        host,
        bridge,
        development_latents,
        development_pairs,
        development_composition_rows,
        batch_size=batch_size,
    )
    bridge.freeze()
    bridge.verify_frozen()
    host.verify_frozen()
    output.mkdir(parents=True)
    bridge_path = output / "bridge.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in bridge.state_dict().items()},
        str(bridge_path),
    )
    receipt = {
        "format": "abi-native-transfer-r8-frozen-generic-bridge/1",
        "host_key": host_key,
        "architecture_family": host.spec.architecture_family,
        "model_id": host.spec.model_id,
        "revision": host.spec.revision,
        "config_sha256": sha256_file(config_path),
        "meta_extraction_receipt_sha256": sha256_file(extraction_dir / "receipt.json"),
        "host_snapshot_inventory": host.inventory,
        "host_model_state_sha256_before": host.model_state_sha256,
        "host_model_state_sha256_after": host.model_state_sha256,
        "bridge": {
            "path": bridge_path.name,
            "sha256": sha256_file(bridge_path),
            "state_sha256": module_sha256(bridge),
            "parameters": sum(parameter.numel() for parameter in bridge.parameters()),
            "prompt_inputs": 0,
            "answer_inputs": 0,
            "method": config["training"].get("bridge_method", "static_prefix"),
            "adapter_inventory": (
                bridge.adapter_inventory()
                if hasattr(bridge, "adapter_inventory")
                else None
            ),
        },
        "training": {
            "optimizer_steps": steps,
            "meta_capabilities": len(meta_capabilities),
            "meta_composition_pairs": len(meta_pairs),
            "composition_step_fraction": composition_fraction,
            "heldout_capabilities": 0,
            "wall_seconds": time.perf_counter() - started,
            "curves": curves,
        },
        "development": development,
        "development_composition": development_composition,
        "heldout_reveal_present_during_training": False,
        "capability_packages_present_during_training": 0,
        "recipient_parameters_trainable": 0,
        "recipient_optimizer_steps": 0,
        "frozen": True,
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    _write_once(output / "receipt.json", receipt)
    del host, bridge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--extraction-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--host", required=True, choices=tuple(SPECS[key].key for key in ("qwen2", "pythia", "t5"))
    )
    args = parser.parse_args()
    try:
        value = train(
            Path(args.config).resolve(),
            Path(args.extraction_dir).resolve(),
            Path(args.output).resolve(),
            host_key=args.host,
            campaign_root=Path(args.campaign_root).resolve(),
        )
    except (MetaInterfaceError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
