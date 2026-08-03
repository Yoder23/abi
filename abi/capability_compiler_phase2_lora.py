"""Train and development-screen the matched Phase 2 Phi-3 LoRA baselines."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import psutil
import torch

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    Phase2Error,
    canonical_json_bytes,
    capture_lora,
    evaluate_functional,
    install_lora,
    load_catalog,
    load_lora,
    lora_modules,
    repetition_collapse,
    reset_lora,
    response_cross_entropy,
    set_determinism,
    sha256_file,
    stable_seed,
    state_sha256,
)
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot, reconstruct_packs


SYSTEMS = ("L0", "L1")
ROUTER_DIMENSIONS = 4096


def _feature_vector(text: str) -> np.ndarray:
    normalized = " ".join(text.casefold().split())
    vector = np.zeros(ROUTER_DIMENSIONS, dtype=np.float32)
    for index in range(max(0, len(normalized) - 2)):
        gram = normalized[index:index + 3].encode("utf-8")
        bucket = int.from_bytes(__import__("hashlib").sha256(gram).digest()[:4], "big") % ROUTER_DIMENSIONS
        vector[bucket] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


def train_router(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    grouped = {capability: [] for capability in CAPABILITIES}
    for row in records:
        grouped[str(row["capability"])].append(_feature_vector(str(row["normalized_acquisition_prompt"])))
    centroids = {}
    for capability, values in grouped.items():
        if len(values) != 500:
            raise Phase2Error("router training depth changed")
        centroid = np.mean(values, axis=0)
        norm = float(np.linalg.norm(centroid))
        centroids[capability] = centroid / norm if norm else centroid
    return centroids


def route_prompt(prompt: str, centroids: Mapping[str, np.ndarray]) -> str:
    vector = _feature_vector(prompt)
    return max(CAPABILITIES, key=lambda capability: (float(vector @ centroids[capability]), -CAPABILITIES.index(capability)))


def _development_probes(catalog_path: Path, *, per_capability: int) -> list[dict[str, Any]]:
    catalog = load_catalog(catalog_path)
    grouped: dict[str, list[dict[str, Any]]] = {capability: [] for capability in CAPABILITIES}
    for probe in catalog["probes"]:
        if probe.get("split") == "validation" and probe.get("canonical_capability") in grouped:
            grouped[str(probe["canonical_capability"])].append(probe)
    selected = []
    for capability in CAPABILITIES:
        rows = sorted(grouped[capability], key=lambda row: str(row["probe_id"]))
        if len(rows) != 100 or not 1 <= per_capability <= 100:
            raise Phase2Error("development suite depth changed")
        selected.extend(rows[:per_capability])
    return selected


def _render_prompt(tokenizer: Any, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return [int(value) for value in tokenizer(rendered, add_special_tokens=False).input_ids]


def evaluate_development(
    model: Any,
    adapter_states: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    system: str,
    centroids: Mapping[str, np.ndarray],
    tokenizer: Any,
    catalog_path: Path,
    per_capability: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise Phase2Error(f"immutable development outputs already exist: {output_path}")
    probes = _development_probes(catalog_path, per_capability=per_capability)
    rows = []
    router_correct = 0
    current_capability = None
    model.eval()
    model.config.use_cache = True
    started = time.perf_counter()
    for probe in probes:
        expected = str(probe["canonical_capability"])
        routed = expected if system == "L0" else route_prompt(str(probe["prompt"]), centroids)
        router_correct += int(routed == expected)
        if routed != current_capability:
            load_lora(model, adapter_states[routed])
            current_capability = routed
        prompt_ids = _render_prompt(tokenizer, str(probe["prompt"]))
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs,
                do_sample=False,
                max_new_tokens=int(probe["max_new_tokens"]),
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.eos_token_id),
                use_cache=True,
            )[0, len(prompt_ids):]
        output_ids = [int(value) for value in generated.tolist() if int(value) != int(tokenizer.eos_token_id)]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        rows.append({
            "probe_id": probe["probe_id"],
            "capability": expected,
            "routed_capability": routed,
            "output": output,
            "output_token_ids": output_ids,
            "functional_pass": evaluate_functional(output, probe["evaluator"]),
            "repetition_collapse": repetition_collapse(output),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    per_capability_metrics = {}
    for capability in CAPABILITIES:
        subset = [row for row in rows if row["capability"] == capability]
        per_capability_metrics[capability] = {
            "observations": len(subset),
            "functional_passes": sum(bool(row["functional_pass"]) for row in subset),
            "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in subset),
            "route_correct": sum(row["routed_capability"] == capability for row in subset),
        }
    return {
        "observations": len(rows),
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "router_correct": router_correct,
        "per_capability": per_capability_metrics,
        "seconds": time.perf_counter() - started,
        "outputs_path": output_path.as_posix(),
        "outputs_sha256": sha256_file(output_path),
    }


def train_lora(
    *,
    root: Path,
    manifest_path: Path,
    system: str,
    rank: int,
    learning_rate: float,
    exposures: int,
    seed: int,
    development_per_capability: int,
    output_dir: Path,
    save_checkpoint: bool,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    allowed_ranks = {"L0": {16, 64}, "L1": {8, 32}}
    if system not in SYSTEMS or rank not in allowed_ranks[system]:
        raise Phase2Error("LoRA system or rank is outside the frozen grid")
    if learning_rate not in {1e-4, 3e-4} or exposures not in {1, 4}:
        raise Phase2Error("LoRA configuration is outside the frozen grid")
    receipt_path = output_dir / "receipt.json"
    if receipt_path.exists():
        raise Phase2Error(f"immutable receipt already exists: {receipt_path}")
    set_determinism(seed)
    packs, manifest = reconstruct_packs(root=root, manifest_path=manifest_path)
    from .capability_compiler_phase2_common import load_phase1_records

    phase1_records = load_phase1_records(root / "results/abi_capability_compiler_phase1/final/normalized_acquisition_ir_v1.abicir")
    centroids = train_router(phase1_records)
    router_parameters = len(CAPABILITIES) * ROUTER_DIMENSIONS
    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    from transformers import AutoModelForCausalLM

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    source_load_seconds = time.perf_counter() - load_started
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = install_lora(model, rank=rank, alpha=2.0 * rank, dropout=0.05)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    per_capability_training = {}
    successful_steps = 0
    response_tokens_seen = 0
    peak_rss = psutil.Process().memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for capability in CAPABILITIES:
        reset_lora(model, seed=seed, capability=capability)
        parameters = [parameter for _, module in lora_modules(model) for parameter in (module.lora_a, module.lora_b)]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        cap_packs = [pack for pack in packs if pack.capability == capability]
        losses = []
        cap_steps = 0
        cap_tokens = 0
        model.train()
        for exposure in range(exposures):
            order = list(range(len(cap_packs)))
            random.Random(stable_seed(seed, exposure, capability, "lora-order")).shuffle(order)
            for pack_index in order:
                pack = cap_packs[pack_index]
                inputs = torch.tensor([pack.input_ids], dtype=torch.long, device="cuda")
                labels = torch.tensor([pack.labels], dtype=torch.long, device="cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(inputs).logits
                    loss = response_cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise Phase2Error("non-finite LoRA loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                cap_steps += 1
                cap_tokens += sum(label != -100 for label in pack.labels)
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        adapters[capability] = capture_lora(model)
        per_capability_training[capability] = {
            "successful_optimizer_steps": cap_steps,
            "response_tokens_seen": cap_tokens,
            "mean_loss": float(np.mean(losses)),
            "adapter_sha256": state_sha256(adapters[capability]),
        }
        successful_steps += cap_steps
        response_tokens_seen += cap_tokens
        del optimizer
        torch.cuda.empty_cache()
    training_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    development = evaluate_development(
        model,
        adapters,
        system=system,
        centroids=centroids,
        tokenizer=tokenizer,
        catalog_path=root / "catalogs/capability_compiler_phase1_frozen_v1.json",
        per_capability=development_per_capability,
        output_path=output_dir / "development_outputs.jsonl",
    )
    combined_state = {
        f"{capability}.{name}": value
        for capability, state in adapters.items()
        for name, value in state.items()
    }
    checkpoint_path = None
    checkpoint_sha = None
    if save_checkpoint:
        from safetensors.torch import save_file

        checkpoint_path = output_dir / "adapters.safetensors"
        save_file(combined_state, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
    base_parameter_count = sum(parameter.numel() for parameter in model.parameters()) - sum(parameter.numel() for parameter in parameters)
    if base_parameter_count != 3_821_079_552:
        raise Phase2Error("source parameter count changed")
    adapter_parameters = sum(parameter.numel() for parameter in parameters)
    installed_adapter_parameters = adapter_parameters if system == "L0" else adapter_parameters * len(CAPABILITIES)
    receipt = {
        "format": "abi-capability-compiler-phase2-lora-run/1",
        "status": "PASS",
        "system": system,
        "role": "single_capability_adapter_evaluated_per_capability" if system == "L0" else "prompt_routed_fourteen_adapter_system",
        "rank": rank,
        "alpha": 2.0 * rank,
        "dropout": 0.05,
        "learning_rate": learning_rate,
        "target_token_exposures": exposures,
        "seed": seed,
        "optimizer": "AdamW",
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "compute_precision": "bfloat16",
        "source_base": "microsoft/Phi-3-mini-4k-instruct",
        "source_revision": "f39ac1d28e925b323eae81227eaba4464caced4e",
        "source_base_present_at_inference": True,
        "base_parameters": base_parameter_count,
        "adapter_parameters_per_capability": adapter_parameters,
        "installed_adapter_parameters": installed_adapter_parameters,
        "router_parameters": router_parameters if system == "L1" else 0,
        "router_parameter_ratio_vs_source": (router_parameters / base_parameter_count) if system == "L1" else 0.0,
        "target_modules": targets,
        "pack_manifest_sha256": sha256_file(manifest_path),
        "pack_content_sha256": manifest["content_sha256"],
        "successful_optimizer_steps": successful_steps,
        "response_tokens_seen": response_tokens_seen,
        "per_capability_training": per_capability_training,
        "training_seconds": training_seconds,
        "total_wall_seconds_including_verification_and_evaluation": time.perf_counter() - run_started,
        "source_load_seconds": source_load_seconds,
        "development": development,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_bytes": int(peak_rss),
        "combined_adapter_sha256": state_sha256(combined_state),
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix() if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_sha,
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "final_prompts_accessed": False,
        "abi_candidate_trained": False,
    }
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--exposures", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--development-per-capability", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    receipt = train_lora(
        root=root,
        manifest_path=Path(args.manifest).resolve(),
        system=args.system,
        rank=args.rank,
        learning_rate=args.learning_rate,
        exposures=args.exposures,
        seed=args.seed,
        development_per_capability=args.development_per_capability,
        output_dir=Path(args.output_dir).resolve(),
        save_checkpoint=args.save_checkpoint,
    )
    print(json.dumps({
        "status": receipt["status"],
        "system": receipt["system"],
        "steps": receipt["successful_optimizer_steps"],
        "development_passes": receipt["development"]["functional_passes"],
        "development_observations": receipt["development"]["observations"],
        "repetition_collapses": receipt["development"]["repetition_collapses"],
        "router_correct": receipt["development"]["router_correct"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
