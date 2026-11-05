"""Single precision-repaired matched pilot; no new source or parameter sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

from experiments.grounded_english_r99 import run as parent_runner
from experiments.plan_english_r98 import run as engine

from . import core as c
from .train import student


def require_numerical_gate():
    import torch
    from safetensors.torch import load_file

    directory = c.ROOT / "results/grounded_english_r99_numerics"
    proof = c.read(directory / "decision.json")
    data = {}
    for condition in ("fp16_scaled", "fp16_unit", "fp32"):
        path = directory / condition / "rows.jsonl"
        if c.sha(path) != proof["row_hashes"][condition]:
            raise RuntimeError("Numerical proof rows changed")
        data[condition] = c.rows(path)
        if len(data[condition]) != 2 or [r["step"] for r in data[condition]] != [1, 2]:
            raise RuntimeError("Incomplete numerical proof")
    original = c.rows(c.original.OUT / "distillation/training.jsonl")[0]
    first = data["fp16_scaled"][0]
    if first["loss"] != original["loss"] or first["sample_ids"] != original["sample_ids"]:
        raise RuntimeError("Failed replay of initial update")
    bad = data["fp16_scaled"][1]
    path = directory / "fp16_scaled/nonfinite-gradients-step2.safetensors"
    if c.sha(path) != bad["nonfinite_payload_sha256"]:
        raise RuntimeError("Missing/changed non-finite gradient payload")
    counts = {
        key: int((~torch.isfinite(value)).sum()) for key, value in load_file(str(path)).items()
    }
    if counts != bad["nonfinite_counts"] or sum(counts.values()) != 50688:
        raise RuntimeError("Numerical failure not recomputable")
    for condition in ("fp16_unit", "fp32"):
        for i, row in enumerate(data[condition]):
            if row["sample_ids"] != data["fp16_scaled"][i]["sample_ids"]:
                raise RuntimeError("Unpaired numerical proof")
            if (
                row["nonfinite_counts"]
                or row["gradient_norm"] is None
                or not math.isfinite(row["gradient_norm"])
            ):
                raise RuntimeError("No finite numerical control")


def freeze():
    if c.OUT.exists():
        raise RuntimeError("Preserve previous FP32 outputs")
    c.original.verify_binding()
    require_numerical_gate()
    for relative in (
        "train.json",
        "development.json",
        "source/rows.jsonl",
        "source/raw.jsonl",
        "source/result.json",
    ):
        target = c.OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.original.OUT / relative, target)
    files = dict(c.read(c.original.OUT / "freeze.json")["files"])
    paths = list(c.HERE.glob("*.py")) + [
        c.HERE / "PROTOCOL.md",
        c.ROOT / "tests/test_grounded_english_r99_fp32.py",
    ]
    paths += [
        p for p in (c.ROOT / "results/grounded_english_r99_numerics").rglob("*") if p.is_file()
    ]
    paths += [p for p in c.OUT.rglob("*") if p.is_file()]
    paths += [
        c.original.OUT / "freeze.json",
        c.original.OUT / "launch.json",
        c.original.OUT / "stop.json",
        c.original.HERE / "numerics.py",
        c.original.HERE / "NUMERICAL_AUDIT.md",
    ]
    files.update({os.path.relpath(p, c.ROOT).replace("\\", "/"): c.sha(p) for p in paths})
    deadline = c.read(c.original.OUT / "launch.json")["start_unix"] + 21600
    c.write_once(
        c.OUT / "freeze.json",
        {
            "format": "abi-r99-fp32-freeze/1",
            "files": files,
            "abi_commit": engine.git(c.ROOT, "rev-parse", "HEAD"),
            "deadline_unix": deadline,
            "precision": "float32_no_autocast_no_scaler_no_tf32",
            "source_reused_byte_exact": True,
            "steps": c.STEPS,
            "seed": c.SEED,
            "full_moonshot": "NOT_PROVEN",
        },
    )
    print(
        json.dumps(
            {
                "freeze_sha256": c.sha(c.OUT / "freeze.json"),
                "files": len(files),
                "deadline_unix": deadline,
            }
        )
    )


def verify():
    import torch
    from safetensors import safe_open

    for arm in c.ARMS:
        log = c.rows(c.OUT / arm / "training.jsonl")
        if len(log) != c.STEPS or any(
            not math.isfinite(r["loss"]) or not math.isfinite(r["gradient_norm"]) for r in log
        ):
            raise RuntimeError("Non-finite/incomplete FP32 evidence")
        result = c.read(c.OUT / arm / "training_result.json")
        if result["precision"] != "float32" or any(
            result[key] for key in ("autocast", "loss_scaling", "tf32")
        ):
            raise RuntimeError("Wrong precision lineage")
        if result["loss_tokens"] != sum(r["loss_tokens"] for r in log):
            raise RuntimeError("Training token accounting mismatch")
        if result["padded_token_positions"] != c.STEPS * c.BATCH * c.CONTEXT:
            raise RuntimeError("Unmatched padded workload")
        with safe_open(c.OUT / arm / "model.safetensors", framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                tensor = tensors.get_tensor(key)
                if not torch.isfinite(tensor).all():
                    raise RuntimeError("Non-finite deployed tensor")
    parent_runner.verify()


def pilot():
    manifest = c.verify_binding()
    require_numerical_gate()
    order = list(c.ARMS)
    random.Random(c.SEED).shuffle(order)
    stages = [("preflight",), ("verify-source",)]
    stages += [("train", "--arm", arm) for arm in order]
    stages += [("evaluate", "--arm", arm) for arm in ("parent", *order)] + [("verify",)]
    c.write_once(
        c.OUT / "launch.json",
        {
            "command": sys.argv,
            "pid": os.getpid(),
            "stages": stages,
            "start_unix": time.time(),
            "deadline_unix": manifest["deadline_unix"],
            "source_queries": 0,
        },
    )
    for args in stages:
        remaining = manifest["deadline_unix"] - time.time()
        try:
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, 21600)
            done = subprocess.run(
                [sys.executable, "-u", "-m", "experiments.grounded_english_r99_fp32.run", *args],
                cwd=c.ROOT,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            c.write_once(
                c.OUT / "stop.json",
                {
                    "stage": args,
                    "reason": "EXPERIMENT_BUDGET_STOP",
                    "hard_resource_limit_proven": False,
                },
            )
            return
        if done.returncode:
            c.write_once(
                c.OUT / "stop.json",
                {
                    "stage": args,
                    "reason": "FAILED_PREREQUISITE_OR_OPERATIONAL_ERROR",
                    "returncode": done.returncode,
                    "layercake_failure_proven": False,
                },
            )
            return
        if sum(p.stat().st_size for p in c.OUT.rglob("*") if p.is_file()) > 20 * 1024**3:
            c.write_once(
                c.OUT / "stop.json", {"stage": args, "reason": "EXPERIMENT_DISK_BUDGET_STOP"}
            )
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "pilot", "preflight", "verify-source", "train", "evaluate", "verify"),
    )
    parser.add_argument("--arm", choices=(*c.ARMS, "parent"))
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    engine.c, parent_runner.c = c, c
    if args.command in {"train", "evaluate", "verify"} and not c.source_gate(
        parent_runner.verify_source()
    ):
        raise RuntimeError("ABI_NORMALIZATION_PREREQUISITE_FAILED")
    if args.command == "train":
        if args.arm not in c.ARMS:
            parser.error("Training requires registered arm")
        student(args.arm)
    elif args.command == "evaluate":
        if args.arm is None:
            parser.error("Evaluation requires an arm")
        engine.evaluate(args.arm)
    else:
        {
            "freeze": freeze,
            "pilot": pilot,
            "preflight": engine.preflight,
            "verify-source": parent_runner.verify_source,
            "verify": verify,
        }[args.command]()


if __name__ == "__main__":
    main()
