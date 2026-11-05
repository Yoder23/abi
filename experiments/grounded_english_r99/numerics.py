"""Two-step precision attribution only; no quality evaluation or promotion."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

from experiments.plan_english_r98 import run as engine

from . import core as c

OUT = c.ROOT / "results/grounded_english_r99_numerics"


def freeze():
    c.verify_binding()
    paths = [
        Path(__file__),
        c.HERE / "NUMERICAL_AUDIT.md",
        c.OUT / "freeze.json",
        c.OUT / "source/rows.jsonl",
        c.OUT / "source/result.json",
        c.OUT / "distillation/training.jsonl",
        c.OUT / "stop.json",
    ]
    c.write_once(
        OUT / "freeze.json",
        {
            "format": "abi-r99-numerics-freeze/1",
            "files": {str(p.relative_to(c.ROOT)).replace("\\", "/"): c.sha(p) for p in paths},
            "conditions": ["fp16_scaled", "fp16_unit", "fp32"],
            "steps_per_condition": 2,
            "quality_evaluation_authorized": False,
        },
    )


def run(condition):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER

    frozen = c.read(OUT / "freeze.json")
    for relative, digest in frozen["files"].items():
        if c.sha(c.ROOT / relative) != digest:
            raise RuntimeError("Numerical audit binding changed")
    if condition not in frozen["conditions"]:
        raise RuntimeError("Unregistered precision")
    engine.c = c
    c.verify_binding()
    model, tokenizer, _ = engine.load_parent()
    params, _ = engine.prepare_trainable(model, "distillation")
    model.eval()
    reference = c.read(c.OUT / "train.json")
    targets = {r["id"]: r for r in engine.source_rows() if r["split"] == "train"}
    rng = random.Random(c.SEED)
    order = list(range(len(reference)))
    rng.shuffle(order)
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    scaler = (
        None
        if condition == "fp32"
        else torch.amp.GradScaler("cuda", init_scale=65536.0 if condition == "fp16_scaled" else 1.0)
    )
    destination = OUT / condition
    destination.mkdir()
    start = time.perf_counter()
    with (destination / "rows.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
        for step in range(2):
            ids = torch.full(
                (2, c.CONTEXT), tokenizer.eos_token_id, device="cuda", dtype=torch.long
            )
            labels = torch.full_like(ids, -100)
            attention = torch.zeros_like(ids)
            lengths, routes, sample_ids = [], [], []
            for j in range(2):
                row = reference[order[2 * step + j]]
                prefix = tokenizer.encode(
                    c.model_prompt(row, "distillation"), add_special_tokens=False
                )
                suffix = tokenizer.encode(
                    targets[row["id"]]["teacher_output"], add_special_tokens=False
                ) + [tokenizer.eos_token_id]
                sequence = prefix + suffix
                if len(sequence) > c.CONTEXT:
                    raise RuntimeError("Unregistered overflow")
                ids[j, : len(sequence)] = torch.tensor(sequence, device="cuda")
                labels[j, len(prefix) : len(sequence)] = torch.tensor(suffix, device="cuda")
                attention[j, : len(sequence)] = 1
                lengths.append(len(prefix))
                routes.append(CAPABILITY_CAKE_ORDER.index(row["task"]))
                sample_ids.append(row["id"])
            optimizer.zero_grad(set_to_none=True)
            context = (
                contextlib.nullcontext()
                if condition == "fp32"
                else torch.autocast("cuda", dtype=torch.float16)
            )
            with context:
                output = model(
                    ids,
                    attention_mask=attention,
                    prompt_lengths=torch.tensor(lengths, device="cuda"),
                    task_routes=torch.tensor(routes, device="cuda"),
                    use_cache=False,
                )
                loss = F.cross_entropy(
                    output["logits"][:, :-1].float().reshape(-1, model.config.vocab_size),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise RuntimeError("Non-finite forward loss")
            scale = 1.0 if scaler is None else scaler.get_scale()
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            counts = {
                name: int((~torch.isfinite(p.grad)).sum())
                for name, p in model.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            bad = {name: count for name, count in counts.items() if count}
            payload_sha = None
            if bad:
                path = destination / f"nonfinite-gradients-step{step + 1}.safetensors"
                save_file(
                    {
                        name: p.grad.detach().cpu().contiguous()
                        for name, p in model.named_parameters()
                        if name in bad
                    },
                    str(path),
                )
                payload_sha = c.sha(path)
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            observation = {
                "step": step + 1,
                "condition": condition,
                "loss": loss_value,
                "scale": scale,
                "gradient_norm": norm if math.isfinite(norm) else None,
                "gradient_norm_nonfinite": not math.isfinite(norm),
                "nonfinite_counts": bad,
                "nonfinite_payload_sha256": payload_sha,
                "sample_ids": sample_ids,
                "elapsed_seconds": time.perf_counter() - start,
            }
            c.append(journal, observation)
            print(json.dumps(observation), flush=True)
            if bad or not math.isfinite(norm):
                break
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()


def verify():
    import torch
    from safetensors.torch import load_file

    all_rows = {
        name: c.rows(OUT / name / "rows.jsonl") for name in ("fp16_scaled", "fp16_unit", "fp32")
    }
    original = c.rows(c.OUT / "distillation/training.jsonl")[0]
    first = all_rows["fp16_scaled"][0]
    if first["loss"] != original["loss"] or first["sample_ids"] != original["sample_ids"]:
        raise RuntimeError("Original first update did not replay")
    for name, observations in all_rows.items():
        if len(observations) != 2 or [r["step"] for r in observations] != [1, 2]:
            raise RuntimeError("Incomplete numerical audit")
        if [r["sample_ids"] for r in observations] != [
            r["sample_ids"] for r in all_rows["fp16_scaled"]
        ]:
            raise RuntimeError("Unpaired numerical audit")
        for row in observations:
            if row["nonfinite_counts"]:
                path = OUT / name / f"nonfinite-gradients-step{row['step']}.safetensors"
                if c.sha(path) != row["nonfinite_payload_sha256"]:
                    raise RuntimeError("Non-finite gradient payload changed")
                observed = {
                    key: int((~torch.isfinite(value)).sum())
                    for key, value in load_file(str(path)).items()
                }
                if observed != row["nonfinite_counts"]:
                    raise RuntimeError("Unrecomputable non-finite counts")
    attributed = bool(all_rows["fp16_scaled"][1]["nonfinite_counts"]) and all(
        not r["nonfinite_counts"] and not r["gradient_norm_nonfinite"]
        for name in ("fp16_unit", "fp32")
        for r in all_rows[name]
    )
    result = {
        "format": "abi-r99-numerics-decision/1",
        "rows": all_rows,
        "verdict": "LOSS_SCALING_PRECISION_FAILURE_ISOLATED"
        if attributed
        else "ATTRIBUTION_UNRESOLVED",
        "quality_evaluated": False,
        "layercake_architectural_regression_established": False,
        "row_hashes": {name: c.sha(OUT / name / "rows.jsonl") for name in all_rows},
    }
    c.write_once(OUT / "decision.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "verify"))
    parser.add_argument("--condition", choices=("fp16_scaled", "fp16_unit", "fp32"))
    args = parser.parse_args()
    if args.command == "run":
        run(args.condition)
    else:
        {"freeze": freeze, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
