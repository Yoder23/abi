# ABI Developer Start-Here

This document gets you operational in under 10 minutes. Read it top to bottom — every step matters.

---

## Step 0 — Understand What You Are Working With

The **Adaptive Bridge Interface (ABI)** attaches a trainable correction module to a frozen T5-large backbone. After training, it makes the backbone's output distribution in a new domain non-inferior to its output distribution on the domain it was originally calibrated on — as measured by the **Non-Inferiority Benchmark (NIB)**.

The published result (Experiment 45AS, `cross_arch_t5_nib_v53.py`) achieves:
- **Official NIB top-5: 0.8725** — PASSES the protocol threshold of 0.860
- **Extended NIB (n=25, 5 random seeds): mean top-5 = 0.8549**

The backbone is never modified. Only the ABI module's 163.6 M parameters are trained.

---

## Step 1 — Prerequisites

### Hardware

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | NVIDIA CUDA | Tested on RTX 3080 Laptop (16 GB VRAM) |
| GPU VRAM | ≥ 10 GB | The ABI + T5-large backbone fits in 10–14 GB during training |
| RAM | ≥ 16 GB | T5-large weights alone are ~2.8 GB; ABI adds ~630 MB in fp32 |
| Disk | ≥ 5 GB free | For model cache, logs, and checkpoints |

**CPU-only is supported but extremely slow** (expect 20–30× longer training). Not recommended for reproduction.

### Software

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| Python | 3.10 | Tested exclusively on 3.10 |
| PyTorch | 2.1 | `output_hidden_states=True` behaviour confirmed on 2.1+ |
| transformers | 4.38+ | `T5ForConditionalGeneration`, `T5TokenizerFast` |
| sentencepiece | 0.1.99 | T5 tokenizer backend |
| numpy | 1.24+ | NIB evaluation |

---

## Step 2 — Install Dependencies

```powershell
pip install "torch>=2.1" "transformers>=4.38" "sentencepiece>=0.1.99" numpy tqdm
```

Verify:

```powershell
python -c "
import torch, transformers, sentencepiece, numpy
print('torch    ', torch.__version__, '  CUDA:', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('sentencepiece', sentencepiece.__version__)
print('numpy    ', numpy.__version__)
"
```

Expected (minimum acceptable):
```
torch     2.1.x   CUDA: True
transformers 4.38.x
sentencepiece 0.1.99
numpy     1.24.x
```

If `CUDA: False`, check your CUDA driver and PyTorch CUDA build.

---

## Step 3 — Download T5-large (one-time, requires internet)

The training scripts set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` by default, so the model **must be cached before running the scripts**.

```powershell
python -c "
from transformers import T5ForConditionalGeneration, T5TokenizerFast
print('Downloading T5-large weights and tokenizer...')
T5ForConditionalGeneration.from_pretrained('t5-large')
T5TokenizerFast.from_pretrained('t5-large')
print('Done. t5-large is cached.')
"
```

This downloads approximately 2.9 GB. It only needs to run once; HuggingFace caches to `~/.cache/huggingface/hub/` (or `%USERPROFILE%\.cache\huggingface\hub\` on Windows).

**Verify the cache:**

```powershell
python -c "
import os; from transformers import T5ForConditionalGeneration
m = T5ForConditionalGeneration.from_pretrained('t5-large')
print(f't5-large loaded: {sum(p.numel() for p in m.parameters()):,} parameters')
"
```

Expected output: `t5-large loaded: 737,668,096 parameters`

---

## Step 4 — Navigate to the Working Directory

All scripts expect to be run from:

```
C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready\layercakeogwithdecoder\
```

```powershell
cd "C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready\layercakeogwithdecoder"
```

Verify you are in the right place:

```powershell
Test-Path "cross_arch_t5_nib_v53.py"   # must return True
Test-Path "cross_arch_t5_nib_v48.py"   # must return True
Test-Path "cross_arch_t5_nib_v51.py"   # must return True
```

---

## Step 5 — Read the Existing Published Result (2 seconds)

You do not need to retrain to see the result. The published run's output is already in the repository.

```powershell
python -c "
import json
r = json.load(open('cross_arch_t5_nib_v53_results.json'))
nib = r['nib_official']
ext = r['nib_combined']

print('=== EXPERIMENT 45AS — PUBLISHED RESULT ===')
print()
print('OFFICIAL NIB (rng=7777, n=5 chunks):')
print(f'  top5={nib[\"mean_top5\"]}  top1={nib[\"mean_top1\"]}  JS={nib[\"mean_js\"]}  ent={nib[\"mean_ent\"]}')
print(f'  PASS={nib[\"pass\"]}')
print()
print('EXTENDED NIB (5 rng seeds × 5 chunks = n=25):')
print(f'  mean_top5={ext[\"mean_top5\"]}  std={ext[\"std_top5\"]:.4f}  SE={ext[\"se_top5\"]:.4f}')
print(f'  95% CI: [{ext[\"ci_95_low\"]}, {ext[\"ci_95_high\"]}]')
for seed, val in ext.get('per_rng_mean_top5', {}).items():
    tag = 'PASS' if val >= 0.860 else 'fail'
    print(f'    rng={seed}: {val:.4f}  [{tag}]')
print()
print(f'corrMSE: {r[\"best_corrMSE\"]:.6f} @ step {r[\"best_step\"]}')
print(f'Elapsed: {r[\"elapsed_min\"]:.1f} min')
"
```

Expected output:
```
=== EXPERIMENT 45AS — PUBLISHED RESULT ===

OFFICIAL NIB (rng=7777, n=5 chunks):
  top5=0.8725  top1=0.8508  JS=0.01391  ent=0.2256
  PASS=True

EXTENDED NIB (5 rng seeds × 5 chunks = n=25):
  mean_top5=0.8549  std=0.0316  SE=0.0063
  95% CI: [0.8425, 0.8673]
    rng=7777: 0.8725  [PASS]
    rng=1111: 0.8353  [fail]
    rng=2222: 0.8475  [fail]
    rng=3333: 0.8542  [fail]
    rng=4444: 0.8651  [PASS]

corrMSE: 0.003047 @ step 15466
Elapsed: 237.4 min
```

---

## Step 6 — Understand the Three-Script Dependency Chain

The experiment is split across three scripts for historical reasons. When you run `cross_arch_t5_nib_v53.py`, it imports:

```
cross_arch_t5_nib_v53.py   (Experiment 45AS — main script)
    ├── import cross_arch_t5_nib_v48 as base      (shared utilities + base classes)
    └── import cross_arch_t5_nib_v51 as repro_base (extended NIB evaluation function)
```

**You never call v48 or v51 directly.** Only run `cross_arch_t5_nib_v53.py`.

| Script | Role | Modify? |
|--------|------|---------|
| `cross_arch_t5_nib_v48.py` | Shared base: `SVAT5Large`, `MultiTap6SV`, `DomainModuleSV`, training loop, `make_batch`, `ppl`, `get_lr`, `freeze_backbone` | No |
| `cross_arch_t5_nib_v51.py` | Extended NIB: `nib_eval_with_seed(anchor, cal, tokens, rng_seed, label="")` | No |
| `cross_arch_t5_nib_v53.py` | Best result: `MultiTap6SV_PerTapLN` class + full experiment `main()` | No |

---

## Step 7 — Data Pipeline (No Download Required)

The scripts do **not** require an external dataset. They automatically build a Python corpus by reading all `*.py` files in the working directory tree up to 500,000 T5 tokens:

```python
for p in ROOT.rglob("*.py"):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    py_parts.append(txt)
    py_chars += len(txt)
    if py_chars >= MAX_PY * 4:  # ~2 MB of source
        break
py_raw = "\n".join(py_parts)
py_ids = tokenizer(py_raw)["input_ids"][:500_000]
```

The working directory contains over 300 Python files, providing a rich, consistent domain corpus. This means **training is fully self-contained** — no dataset download step.

---

## Step 8 — Training Phase Reference

When you run the script, output appears in four phases. Here is what to expect at each:

### Domain Pre-training (Steps before calibration)
```
Step A: Anchor (single-tap D_ABI=512, SEED_A=42)
  [A] Anchor ABI (SEED_A=42, single-tap layer 24, 2000 steps)...
      step 500/2000  ppl=22.4
      ...
  [A] 142s  ppl_anchor=19.7

Step C: 6-Tap D_ABI=4096 Native (SEED_C=99)
  [C] 6-tap per-tap-LN D_ABI=4096 native ...
      ABI params: 163,614,721
      ...
  [C] 156s  ppl_c=20.1
```

### Calibration Phases P1 → P4
```
  [D] corrMSE Calibration ...  (16000 steps)
      P1=4000@5e-3  P2=3000@5e-4  P3=3000@5e-5  P4=6000@5e-6
      [P1] step    500/16000  lr=5.00e-03  corrMSE=0.006243  ...
      [P1] step   1000/16000  lr=5.00e-03  corrMSE=0.005481  ...
      ...
      [P4] step  15466/16000  lr=5.00e-06  corrMSE=0.003047  ← best checkpoint
      ...
      Restoring best checkpoint (step=15466, corrMSE=0.003047)
```

### NIB Evaluation
```
  [45AS] chunk 1/5: JS=0.0102  top1=0.853  top5=0.880  ent=0.198
  ...
  OFFICIAL NIB:  top5=0.8725  PASS=True
  EXTENDED NIB:  mean=0.8549  CI=[0.8425, 0.8673]
```

If your corrMSE at the end of P4 is within ±0.0003 of 0.003047, the run is consistent with the published result.

---

## Step 9 — Where to Look If Something Goes Wrong

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` | GPU VRAM < 10 GB | Reduce `BATCH_SV` from 2 to 1 in `cross_arch_t5_nib_v48.py` line ~140 (increases training time ~2×) |
| `OSError: t5-large not found` | Model not cached offline | Run Step 3 again with internet connected |
| `ModuleNotFoundError: cross_arch_t5_nib_v48` | Wrong working directory | `cd` to the `layercakeogwithdecoder/` directory first |
| corrMSE stuck above 0.005 after P2 | Backbone not truly frozen / gradient leak | Verify `freeze_backbone()` is called before calibration |
| NIB top-5 << 0.85 with correct corrMSE | RNG seed drift | Ensure `SEED_C=99`, `SEED_A=42` are unmodified in v48 |
| Script produces no output | PowerShell encoding issue | Set `$env:PYTHONIOENCODING="utf-8"` and `$env:PYTHONUTF8="1"` before running |

---

## Step 10 — What Not to Touch

These parameters are locked. Changing them produces a different experiment and invalidates comparability with 45AS:

| Constant | Location | Value | Why frozen |
|----------|----------|-------|-----------|
| `TAP_LAYERS` | v48.py | `[19,20,21,22,23,24]` | Architecture definition |
| `SEED_A` | v48.py | `42` | Anchor initialization |
| `SEED_C_NEW` | v53.py | `99` | Native initialization |
| `D_ABI_NEW` | v53.py | `4096` | Architecture definition |
| `P4_EXTENDED` | v53.py | `6000` | Schedule definition |
| `ENC_LEN` | v48.py | `64` | Batch shape |
| `PRED_LEN` | v48.py | `64` | Batch shape |
| `ACCUM_STEPS` | v48.py | `4` | Effective batch = 2×4=8 |
| `VOCAB_SIZE` | v48.py | `32128` | T5-large vocabulary |
| `SKIP` | v48.py | `5` | NIB: skip first 5 positions |
| NIB RNG seeds | v53.py | `[7777, 1111, 2222, 3333, 4444]` | Protocol definition |

---

## Continuing Onward

- To understand the full architecture in mathematical detail: [ABI_ARCHITECTURE.md](ABI_ARCHITECTURE.md)
- To see all 7+ experiments and why nothing beats 45AS: [ABI_EXPERIMENTS.md](ABI_EXPERIMENTS.md)
- To run the full replication with verified expected output: [ABI_REPRODUCE.md](ABI_REPRODUCE.md)
