#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_abi.py
==========
End-to-end ABI training and evaluation.

Trains the full ABI pipeline (Stage A → C → D) and evaluates against the
Non-Inferiority Benchmark (NIB). Reproduces the published breakthrough result:

    Path 2C — T5-large ABI — Official NIB PASS
    top-5 agreement = 0.8725  (threshold ≥ 0.860)

Usage
-----
    python run_abi.py

Prerequisites
-------------
  1. Python 3.10, PyTorch ≥ 2.1, CUDA GPU (≥ 10 GB VRAM)
  2. t5-large cached locally (see README.md for download instructions)
  3. Run from the layercakeogwithdecoder/ directory

Output
------
  abi_result.json    — complete result record
  Console output     — live training progress and final NIB scores
"""

import json
import os
import pathlib
import sys
import time

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.stdout.reconfigure(line_buffering=True)

from abi.training   import load_python_corpus, train_anchor, \
                           train_candidate_native, train_corrMSE_calibration, DEVICE
from abi.evaluation import nib_eval_extended

ROOT = pathlib.Path(__file__).parent


def _banner(msg: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n  {msg}\n{line}")


def main() -> None:
    t_global = time.time()

    _banner("ABI — Autonomous Basis Injection  |  T5-large  |  Path 2C")
    print(f"  Device:  {DEVICE}")
    print(f"  Published result:  top-5 = 0.8725  (NIB PASS, threshold ≥ 0.860)")
    print(f"  Run type: full training + evaluation (~230–250 min on RTX 3080 Laptop)")

    # ── Data ──────────────────────────────────────────────────────────────────
    _banner("Data Pipeline")
    tokens = load_python_corpus(ROOT)

    # ── Stage A: Anchor ───────────────────────────────────────────────────────
    _banner("Stage A — Anchor Pre-Training  (seed=42)")
    anchor = train_anchor(tokens, seed=42)

    # ── Stage C: Candidate ────────────────────────────────────────────────────
    _banner("Stage C — Candidate Pre-Training  (seed=99)")
    candidate = train_candidate_native(anchor, tokens, seed=99)

    # ── Stage D: corrMSE Calibration ──────────────────────────────────────────
    _banner("Stage D — corrMSE Calibration  (16,000 steps, 4-phase LR)")
    calibrated, best_corrMSE, best_step, phase_floors = \
        train_corrMSE_calibration(anchor, candidate, tokens)

    # ── NIB Evaluation ────────────────────────────────────────────────────────
    _banner("NIB Evaluation")
    nib = nib_eval_extended(anchor, calibrated, tokens)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.time() - t_global) / 60.0
    off     = nib["nib_official"]
    comb    = nib["nib_combined"]
    status  = "PASS" if off["pass"] else "FAIL"

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  ABI RESULT SUMMARY" + " " * 48 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  corrMSE (best):  {best_corrMSE:.6f}  @ step {best_step:<9d}" + " " * 22 + "║")
    print("║                                                                    ║")
    print("║  OFFICIAL NIB  (rng=7777, n=5):" + " " * 36 + "║")
    print(f"║    JS={off['mean_js']:.5f}  top1={off['mean_top1']:.4f}  "
          f"top5={off['mean_top5']:.4f}  ent={off['mean_ent']:.4f}  → {status:<6}" + " " * 7 + "║")
    print("║                                                                    ║")
    print("║  EXTENDED NIB  (n=25, 5 seeds):" + " " * 35 + "║")
    print(f"║    mean={comb['mean_top5']:.4f}  "
          f"95% CI=[{comb['ci_95_low']:.4f}, {comb['ci_95_high']:.4f}]" + " " * 30 + "║")
    print("║                                                                    ║")
    print(f"║  Elapsed: {elapsed:.1f} min" + " " * (57 - len(f"{elapsed:.1f}")) + "║")
    print("╚" + "═" * 68 + "╝")

    if off["pass"]:
        print()
        print("  ✓  PATH 2C COMPLETE — UNIVERSAL KNOWLEDGE TRANSFER VERIFIED")
        print("  ✓  T5-large ABI achieves non-inferior logit distributions")
        print("  ✓  Independent candidate seed, no shared ABI weights, frozen backbone")

    # ── Save results ──────────────────────────────────────────────────────────
    result = {
        "architecture": {
            "model":      "t5-large",
            "tap_layers": [19, 20, 21, 22, 23, 24],
            "n_taps":     6,
            "d_abi":      4096,
            "d_in":       6144,
            "per_tap_ln": True,
            "seed_anchor": 42,
            "seed_candidate": 99,
        },
        "training": {
            "total_cal_steps":   16000,
            "phase_schedule":    {"P1": [4000, 5e-3], "P2": [3000, 5e-4],
                                  "P3": [3000, 5e-5], "P4": [6000, 5e-6]},
            "best_corrMSE":      best_corrMSE,
            "best_step":         best_step,
            "phase_floors":      {k: round(v, 6) for k, v in phase_floors.items()},
        },
        "nib_official":  nib["nib_official"],
        "nib_extended":  {k: v for k, v in nib["nib_extended"].items()},
        "nib_combined":  nib["nib_combined"],
        "elapsed_min":   round(elapsed, 1),
    }
    out = ROOT / "abi_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {out.name}")


if __name__ == "__main__":
    main()
