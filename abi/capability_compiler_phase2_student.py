"""Train and development-screen the matched Phase 2 transformer students."""

from __future__ import annotations

import argparse
import json
import os
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
    CompactTransformerLM,
    Phase2Error,
    canonical_json_bytes,
    evaluate_functional,
    load_catalog,
    pack_manifest,
    repetition_collapse,
    response_cross_entropy,
    set_determinism,
    sha256_bytes,
    sha256_file,
    sparse_topk_kl,
    stable_seed,
    state_sha256,
)
from .capability_compiler_phase2_prepare import (
    TEMPERATURE,
    _verified_snapshot,
    _tokenizer,
    reconstruct_packs,
)


METHODS = ("D0", "D1", "D2")


def _load_topk(summary_path: Path, *, root: Path) -> tuple[dict[str, Path], Mapping[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "PASS"
        or int(summary.get("topk", 0)) != 64
        or float(summary.get("temperature", 0.0)) != TEMPERATURE
    ):
        raise Phase2Error("invalid top-k teacher cache")
    files: dict[str, Path] = {}
    for row in summary["files"]:
        path = (root / str(row["path"])).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise Phase2Error("unsafe or missing top-k cache path")
        if sha256_file(path) != row["sha256"]:
            raise Phase2Error("top-k cache file changed")
        pack_id = str(row["pack_id"])
        if pack_id in files:
            raise Phase2Error("duplicate top-k cache pack")
        files[pack_id] = path
    return files, summary


def _development_probes(catalog_path: Path, *, per_capability: int) -> list[dict[str, Any]]:
    catalog = load_catalog(catalog_path)
    grouped: dict[str, list[dict[str, Any]]] = {capability: [] for capability in CAPABILITIES}
    for probe in catalog["probes"]:
        if probe.get("split") == "validation" and probe.get("canonical_capability") in grouped:
            grouped[str(probe["canonical_capability"])].append(probe)
    selected: list[dict[str, Any]] = []
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


def _generate_student(
    model: CompactTransformerLM,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_id: int,
    device: torch.device,
) -> list[int]:
    sequence = list(prompt_ids)
    output: list[int] = []
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            inputs = torch.tensor([sequence[-768:]], dtype=torch.long, device=device)
            token = int(model(inputs)[0, -1].argmax().item())
            if token == eos_token_id:
                break
            sequence.append(token)
            output.append(token)
    return output


def evaluate_development(
    model: CompactTransformerLM,
    *,
    tokenizer: Any,
    catalog_path: Path,
    per_capability: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise Phase2Error(f"immutable development outputs already exist: {output_path}")
    probes = _development_probes(catalog_path, per_capability=per_capability)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for probe in probes:
        prompt_ids = _render_prompt(tokenizer, str(probe["prompt"]))
        output_ids = _generate_student(
            model,
            prompt_ids,
            max_new_tokens=int(probe["max_new_tokens"]),
            eos_token_id=int(tokenizer.eos_token_id),
            device=next(model.parameters()).device,
        )
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        rows.append({
            "probe_id": probe["probe_id"],
            "capability": probe["canonical_capability"],
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
        }
    return {
        "observations": len(rows),
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "per_capability": per_capability_metrics,
        "seconds": time.perf_counter() - started,
        "outputs_path": output_path.as_posix(),
        "outputs_sha256": sha256_file(output_path),
    }


def train_student(
    *,
    root: Path,
    manifest_path: Path,
    topk_summary_path: Path | None,
    method: str,
    learning_rate: float,
    exposures: int,
    seed: int,
    development_per_capability: int,
    output_dir: Path,
    save_checkpoint: bool,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    if method not in METHODS:
        raise Phase2Error("unsupported student method")
    if learning_rate not in {1e-5, 3e-5} or exposures not in {1, 2, 4}:
        raise Phase2Error("student configuration is outside the frozen grid")
    receipt_path = output_dir / "receipt.json"
    if receipt_path.exists():
        raise Phase2Error(f"immutable receipt already exists: {receipt_path}")
    set_determinism(seed)
    packs, manifest = reconstruct_packs(root=root, manifest_path=manifest_path)
    if method == "D0" and topk_summary_path is not None:
        raise Phase2Error("D0 cannot silently import the teacher-logit channel")
    topk_files: dict[str, Path] = {}
    topk_summary: Mapping[str, Any] | None = None
    if method in {"D1", "D2"}:
        if topk_summary_path is None:
            raise Phase2Error("logit-distilled student requires top-k cache")
        topk_files, topk_summary = _load_topk(topk_summary_path, root=root)
        if set(topk_files) != {pack.pack_id for pack in packs}:
            raise Phase2Error("top-k cache pack set changed")
        if topk_summary["pack_content_sha256"] != manifest["content_sha256"]:
            raise Phase2Error("top-k cache was extracted for different packs")
    tokenizer = _tokenizer(_verified_snapshot(root))
    device = torch.device("cuda")
    model = CompactTransformerLM().to(device)
    if model.parameter_count != 11_060_800:
        raise Phase2Error("student parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    losses: list[dict[str, float]] = []
    successful_steps = 0
    response_tokens_seen = 0
    started = time.perf_counter()
    model.train()
    for exposure in range(exposures):
        order = list(range(len(packs)))
        random.Random(stable_seed(seed, exposure, "student-order")).shuffle(order)
        for pack_index in order:
            pack = packs[pack_index]
            inputs = torch.tensor([pack.input_ids], dtype=torch.long, device=device)
            labels = torch.tensor([pack.labels], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                sequence_loss = response_cross_entropy(logits, labels)
                logit_loss = None
                if method in {"D1", "D2"}:
                    with np.load(topk_files[pack.pack_id], allow_pickle=False) as cache:
                        if str(cache["pack_id"].item()) != pack.pack_id:
                            raise Phase2Error("top-k cache pack identity changed")
                        positions = torch.as_tensor(cache["positions"], dtype=torch.long, device=device)
                        indices = torch.as_tensor(cache["indices"], dtype=torch.long, device=device)
                        values = torch.as_tensor(cache["values"], dtype=torch.float16, device=device)
                    if positions.tolist() != list(pack.response_positions):
                        raise Phase2Error("top-k response positions changed")
                    logit_loss = sparse_topk_kl(
                        logits,
                        positions,
                        indices,
                        values,
                        temperature=TEMPERATURE,
                    )
                if method == "D0":
                    loss = sequence_loss
                elif method == "D1":
                    assert logit_loss is not None
                    loss = logit_loss
                else:
                    assert logit_loss is not None
                    loss = 0.5 * sequence_loss + 0.5 * logit_loss
            if not torch.isfinite(loss):
                raise Phase2Error("non-finite student loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            successful_steps += 1
            response_tokens_seen += sum(label != -100 for label in pack.labels)
            losses.append({
                "total": float(loss.detach()),
                "sequence": float(sequence_loss.detach()),
                "logit": float(logit_loss.detach()) if logit_loss is not None else 0.0,
            })
            peak_rss = max(peak_rss, process.memory_info().rss)
    training_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    development = evaluate_development(
        model,
        tokenizer=tokenizer,
        catalog_path=root / "catalogs/capability_compiler_phase1_frozen_v1.json",
        per_capability=development_per_capability,
        output_path=output_dir / "development_outputs.jsonl",
    )
    checkpoint_path = None
    checkpoint_sha = None
    if save_checkpoint:
        from safetensors.torch import save_file

        checkpoint_path = output_dir / "student.safetensors"
        save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
    state_hash = state_sha256(model.state_dict())
    loss_summary = {
        key: float(np.mean([row[key] for row in losses])) for key in ("total", "sequence", "logit")
    }
    receipt = {
        "format": "abi-capability-compiler-phase2-student-run/1",
        "status": "PASS",
        "method": method,
        "learning_rate": learning_rate,
        "target_token_exposures": exposures,
        "seed": seed,
        "optimizer": "AdamW",
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "compute_precision": "bfloat16",
        "parameter_storage_precision": "float32",
        "student_spec": model.spec,
        "student_parameters": model.parameter_count,
        "student_parameter_bytes_deployed_bfloat16": model.parameter_count * 2,
        "layercake_phase2_active_parameter_bytes_reference": 21_720_964,
        "deployed_parameter_byte_ratio_vs_layercake_reference": model.parameter_count * 2 / 21_720_964,
        "pack_manifest_sha256": sha256_file(manifest_path),
        "pack_content_sha256": manifest["content_sha256"],
        "topk_summary_sha256": sha256_file(topk_summary_path) if topk_summary_path else None,
        "successful_optimizer_steps": successful_steps,
        "response_tokens_seen": response_tokens_seen,
        "loss": loss_summary,
        "training_seconds": training_seconds,
        "total_wall_seconds_including_verification_and_evaluation": time.perf_counter() - run_started,
        "development": development,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_bytes": int(peak_rss),
        "state_sha256": state_hash,
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
    parser.add_argument("--topk-summary")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--exposures", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--development-per-capability", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    receipt = train_student(
        root=root,
        manifest_path=Path(args.manifest).resolve(),
        topk_summary_path=Path(args.topk_summary).resolve() if args.topk_summary else None,
        method=args.method,
        learning_rate=args.learning_rate,
        exposures=args.exposures,
        seed=args.seed,
        development_per_capability=args.development_per_capability,
        output_dir=Path(args.output_dir).resolve(),
        save_checkpoint=args.save_checkpoint,
    )
    print(json.dumps({
        "status": receipt["status"],
        "method": receipt["method"],
        "steps": receipt["successful_optimizer_steps"],
        "development_passes": receipt["development"]["functional_passes"],
        "development_observations": receipt["development"]["observations"],
        "repetition_collapses": receipt["development"]["repetition_collapses"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
