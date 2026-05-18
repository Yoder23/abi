# ABI Replication Guide

Step-by-step instructions to reproduce Experiment 45AS from scratch, with exact expected output at each checkpoint. Every command is tested on Windows with PowerShell and Python 3.10.

---

## Overview

Reproducing 45AS involves:
1. Setting up the environment (once)
2. Verifying the environment (5 minutes)
3. Running the training script (~4 hours)
4. Verifying the output matches the published result

If you only want to **verify the existing published result** without retraining, skip to [Section 4](#4-verify-the-published-result-without-retraining).

---

## 1. Environment Setup

### 1.1 Install Python 3.10

This project is tested exclusively on Python 3.10. Other versions may work but are not validated.

Verify your Python version:
```powershell
python --version   # should print: Python 3.10.x
```

### 1.2 Install PyTorch with CUDA

```powershell
# For CUDA 11.8 (most common for RTX 3080 with Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True  NVIDIA GeForce RTX 3080 Laptop GPU` (or similar).

### 1.3 Install Remaining Dependencies

```powershell
pip install "transformers>=4.38" "sentencepiece>=0.1.99" numpy tqdm
```

### 1.4 Download T5-large (requires internet, one-time)

```powershell
python -c "
from transformers import T5ForConditionalGeneration, T5TokenizerFast
print('Downloading t5-large...')
T5ForConditionalGeneration.from_pretrained('t5-large')
T5TokenizerFast.from_pretrained('t5-large')
print('Download complete.')
"
```

This downloads approximately 2.9 GB to `%USERPROFILE%\.cache\huggingface\hub\`.

---

## 2. Pre-Run Verification

Run these checks before starting the training. Each should pass with zero errors.

### 2.1 Check Working Directory

```powershell
cd "C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready\layercakeogwithdecoder"
Test-Path "cross_arch_t5_nib_v53.py"; Test-Path "cross_arch_t5_nib_v48.py"; Test-Path "cross_arch_t5_nib_v51.py"
```

Expected: three `True` lines.

### 2.2 Check T5-large Cache

```powershell
python -c "
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import T5ForConditionalGeneration, T5TokenizerFast
m = T5ForConditionalGeneration.from_pretrained('t5-large')
t = T5TokenizerFast.from_pretrained('t5-large')
n = sum(p.numel() for p in m.parameters())
print(f'OK: t5-large {n:,} params, vocab {t.vocab_size}')
del m
"
```

Expected: `OK: t5-large 737,668,096 params, vocab 32100`

Note: `T5TokenizerFast.vocab_size` reports 32100; the model's internal vocabulary with special tokens is 32128. Both values are correct.

### 2.3 Check GPU Memory

```powershell
python -c "
import torch
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {total:.1f} GB')
    if total < 10:
        print('WARNING: VRAM < 10 GB. Reduce BATCH_SV=2 to BATCH_SV=1 in cross_arch_t5_nib_v48.py')
    else:
        print('VRAM sufficient.')
else:
    print('ERROR: CUDA not available.')
"
```

Expected: GPU with ≥ 10 GB VRAM. If < 10 GB, see [Section 5.1](#51-cuda-out-of-memory).

### 2.4 Check Python Source Data Availability

```powershell
python -c "
import pathlib
ROOT = pathlib.Path('.')
py_files = list(ROOT.rglob('*.py'))
print(f'Python files found: {len(py_files)}')
if len(py_files) < 50:
    print('WARNING: very few Python files. Data quality may be reduced.')
else:
    print('Data source: OK')
"
```

Expected: `Python files found: 300+` (the working directory contains the full codebase).

---

## 3. Run the Full Replication

### 3.1 Set Environment Variables

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
```

These ensure correct Unicode handling in PowerShell, which is required for the progress output.

### 3.2 Launch Training

```powershell
cd "C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready\layercakeogwithdecoder"

c:\Python310\python.exe -X utf8 -u "cross_arch_t5_nib_v53.py" 2>&1 | Tee-Object -FilePath "cross_arch_t5_nib_v53_repro_log.txt"
```

The `Tee-Object` writes all output to a log file in addition to the terminal. The `-u` flag disables output buffering so progress is printed in real time.

**Expected total runtime:** approximately 230–250 minutes on an RTX 3080 Laptop.

### 3.3 Expected Output — Startup

Within the first 30 seconds:
```
========================================================================
  Experiment 45AS -- T5-large: Per-Tap LayerNorm + D_ABI=4096 + Ext P4
========================================================================
  Device:  cuda
  Taps: [19, 20, 21, 22, 23, 24]  D_IN=6144  D_ABI=4096  SEED_C=99
  Target corrMSE (45AI floor): 0.003339
  ...
  [Data] Loading Python corpus...
  [Data] py=500,000
```

If `Device: cpu` appears, your CUDA installation is not working. Stop and fix it before proceeding.

### 3.4 Expected Output — Stage A (Anchor, ~20 min)

```
========================================================================
  Step A: Anchor (single-tap D_ABI=512, SEED_A=42)
========================================================================
  [A] Anchor ABI (SEED_A=42, single-tap layer 24, 2000 steps)...
      ABI params: 528,897
      step  500/2000  ppl=22.x
      step 1000/2000  ppl=21.x
      step 1500/2000  ppl=20.x
      step 2000/2000  ppl=19.x
  [A] ~142s  ppl_anchor=19.x
```

Acceptable range for `ppl_anchor`: 18–22. Higher PPL at this stage does not affect final NIB.

### 3.5 Expected Output — Stage C (Native, ~20 min)

```
========================================================================
  Step C: 6-Tap D_ABI=4096 Native Domain Training (SEED_C=99)
========================================================================
  [C] 6-tap per-tap-LN D_ABI=4096 native ...
      ABI params: 163,614,721
      step  500/2000  ppl=22.x
      ...
  [C] ~156s  ppl_c=20.x
  ppl_A=19.x  ppl_C=20.x  gap=0.0xx
```

The ABI parameter count must be exactly **163,614,721** (excludes `tap_lns` which are frozen during Stage C, re-enabled for Stage D). If you see a substantially different number, the architecture has been modified.

### 3.6 Expected Output — Stage D Phase P1 (~35 min)

```
========================================================================
  Step D: corrMSE Calibration 6-tap per-tap-LN D_ABI=4096 (16000 steps)
========================================================================
  [D] corrMSE Calibration ... (16000 steps)
      P1=4000@5e-3  P2=3000@5e-4  P3=3000@5e-5  P4=6000@5e-6
      Cal params: 163,627,009

      [P1] step    500/16000  lr=5.00e-03  corrMSE=0.006xxx
      [P1] step   1000/16000  lr=5.00e-03  corrMSE=0.005xxx
      [P1] step   2000/16000  lr=5.00e-03  corrMSE=0.004xxx
      [P1] step   3000/16000  lr=5.00e-03  corrMSE=0.004xxx
      [P1] step   4000/16000  lr=5.00e-03  corrMSE=0.004xxx
      ── P2 start (step 4001)  P1 corrMSE_floor=0.004xxx
```

P1 corrMSE floor should be approximately 0.004–0.005 (decreasing from ~0.006). This is normal.

### 3.7 Expected Output — Phases P2 and P3 (~50 min)

```
      [P2] step   5000/16000  lr=5.00e-04  corrMSE=0.003xxx
      ...
      [P2] step   7000/16000  lr=5.00e-04  corrMSE=0.003xxx  below_45AG✓
      ── P3 start (step 7001)  P2 corrMSE_floor=0.003xxx
      ...
      [P3] step  10000/16000  lr=5.00e-05  corrMSE=0.003xxx
      ── P4 start (step 10001)  P3 corrMSE_floor=0.003xxx
```

By the end of P3, corrMSE should be in the range 0.0034–0.0037.

### 3.8 Expected Output — Phase P4 (~110 min)

```
      [P4] step  10500/16000  lr=5.00e-06  corrMSE=0.003xxx
      [P4] step  11000/16000  lr=5.00e-06  corrMSE=0.003xxx
      ...
      [P4] step  15000/16000  lr=5.00e-06  corrMSE=0.003xxx  ← approaching floor
      [P4] step  15466/16000  lr=5.00e-06  corrMSE=0.003047  ← (or nearby: best checkpoint)
      ...
      [P4] step  16000/16000  lr=5.00e-06  corrMSE=0.003xxx

      Restoring best checkpoint (step=~15466, corrMSE=~0.003047)
  [D] ~8500s  cal_ppl=18.3  best_corrMSE=0.003047@15466
```

**Acceptable best_corrMSE range:** 0.002950 – 0.003150. The exact value varies by ±0.0003 due to minor non-determinism in CUDA operations (cuBLAS kernel selection can vary slightly between CUDA versions and GPU models). Values within this range should produce equivalent NIB results.

If corrMSE at the end of P4 is still above 0.0040, something is wrong (see [Section 5](#5-troubleshooting)).

### 3.9 Expected Output — NIB Evaluation (~15 min)

```
========================================================================
  NIB Evaluation (OFFICIAL: rng=7777, n=5)
========================================================================
    [45AS] chunk 1/5: JS=0.0xxx  top1=0.8xx  top5=0.8xx  ent=0.2xx
    [45AS] chunk 2/5: JS=0.0xxx  top1=0.8xx  top5=0.8xx  ent=0.2xx
    [45AS] chunk 3/5: JS=0.0xxx  top1=0.8xx  top5=0.8xx  ent=0.2xx
    [45AS] chunk 4/5: JS=0.0xxx  top1=0.8xx  top5=0.8xx  ent=0.2xx
    [45AS] chunk 5/5: JS=0.0xxx  top1=0.8xx  top5=0.8xx  ent=0.2xx

  OFFICIAL NIB:  JS=0.01xxx  top1=0.8xxx  top5=0.8xxx  ent=0.2xxx  PASS=True

========================================================================
  Extended NIB (seeds: 1111, 2222, 3333, 4444)
========================================================================
    [1111] chunk 1/5: ...
    ...
  EXTENDED (n=25): mean_top5=0.8xxx  CI=[0.8xxx, 0.8xxx]
```

### 3.10 Final Summary (Expected)

```
╔══════════════════════════════════════════════════════════════════════╗
║  EXPERIMENT 45AS COMPLETE                                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  corrMSE: 0.003047  (target: <0.003339)                             ║
║  OFFICIAL NIB:                                                       ║
║    JS=0.01391  top1=0.8508  top5=0.8725  ent=0.2256                 ║
║    PASS: True  (all 4 criteria met)                                  ║
║  EXTENDED NIB (n=25):                                                ║
║    mean_top5=0.8549  CI=[0.8425, 0.8673]                            ║
║  PATH 2C COMPLETE — T5-LARGE NIB PASS                               ║
╚══════════════════════════════════════════════════════════════════════╝

  Results saved: cross_arch_t5_nib_v53_results.json
  Elapsed: 237.4 min
```

---

## 4. Verify the Published Result (Without Retraining)

The result file from the original published run is already in the repository at `cross_arch_t5_nib_v53_results.json`. This verification takes under 5 seconds.

### 4.1 Quick Verification Script

```powershell
cd "C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready\layercakeogwithdecoder"

python -c "
import json, sys

with open('cross_arch_t5_nib_v53_results.json') as f:
    r = json.load(f)

PASS = True
errors = []

# Verify official NIB
nib = r['nib_official']
if not (nib['mean_js'] < 0.10):     errors.append(f'JS FAIL: {nib[\"mean_js\"]} >= 0.10')
if not (nib['mean_top1'] >= 0.68):  errors.append(f'top1 FAIL: {nib[\"mean_top1\"]} < 0.68')
if not (nib['mean_top5'] >= 0.86):  errors.append(f'top5 FAIL: {nib[\"mean_top5\"]} < 0.86')
if not (nib['mean_ent'] < 0.35):    errors.append(f'ent FAIL: {nib[\"mean_ent\"]} >= 0.35')

# Verify architecture
assert r['tap_layers'] == [19,20,21,22,23,24], 'tap_layers mismatch'
assert r['d_abi'] == 4096,                      'd_abi mismatch'
assert r['per_tap_ln'] == True,                 'per_tap_ln mismatch'
assert r['n_taps'] == 6,                        'n_taps mismatch'
assert r['total_cal_steps'] == 16000,           'total_cal_steps mismatch'

ext = r['nib_combined']
print('=== RESULT VERIFICATION ===')
print()
print(f'Architecture: 6-tap [19-24], per-tap LN, D_ABI=4096')
print(f'Calibration:  {r[\"total_cal_steps\"]} steps, best corrMSE={r[\"best_corrMSE\"]:.6f} @ step {r[\"best_step\"]}')
print()
print('OFFICIAL NIB (rng=7777, n=5):')
print(f'  JS={nib[\"mean_js\"]}  top1={nib[\"mean_top1\"]}  top5={nib[\"mean_top5\"]}  ent={nib[\"mean_ent\"]}')
print(f'  PASS={nib[\"pass\"]}')
print()
print('EXTENDED NIB (n=25, 5 seeds):')
print(f'  mean={ext[\"mean_top5\"]}  std={ext[\"std_top5\"]:.4f}  SE={ext[\"se_top5\"]:.4f}')
print(f'  95% CI: [{ext[\"ci_95_low\"]}, {ext[\"ci_95_high\"]}]')
for seed, val in sorted(ext.get('per_rng_mean_top5', {}).items(), key=lambda x: x[0]):
    tag = 'PASS' if val >= 0.860 else 'fail'
    print(f'    rng={seed}: {val:.4f}  [{tag}]')
print()

if errors:
    print('VERIFICATION FAILED:')
    for e in errors: print(f'  ERROR: {e}')
    sys.exit(1)
else:
    print('All NIB criteria: PASSED')
    print('Architecture fields: VERIFIED')
"
```

### 4.2 Expected Output

```
=== RESULT VERIFICATION ===

Architecture: 6-tap [19-24], per-tap LN, D_ABI=4096
Calibration:  16000 steps, best corrMSE=0.003047 @ step 15466

OFFICIAL NIB (rng=7777, n=5):
  JS=0.01391  top1=0.8508  top5=0.8725  ent=0.2256
  PASS=True

EXTENDED NIB (n=25, 5 seeds):
  mean=0.8549  std=0.0316  SE=0.0063
  95% CI: [0.8425, 0.8673]
    rng=1111: 0.8353  [fail]
    rng=2222: 0.8475  [fail]
    rng=3333: 0.8542  [fail]
    rng=4444: 0.8651  [PASS]
    rng=7777: 0.8725  [PASS]

All NIB criteria: PASSED
Architecture fields: VERIFIED
```

---

## 5. Troubleshooting

### 5.1 CUDA Out of Memory

**Symptom:** `torch.cuda.OutOfMemoryError` during Stage A, C, or D.

**Fix:** Reduce `BATCH_SV` from 2 to 1. Open `cross_arch_t5_nib_v48.py` and change:

```python
BATCH_SV    = 2
```
to:
```python
BATCH_SV    = 1
```

This halves the effective batch size and roughly doubles training time. The corrMSE floor and NIB results should be within noise of the published values.

**Note:** Do not modify `ACCUM_STEPS`. The gradient accumulation compensates for the smaller batch.

### 5.2 Script Runs But Produces No Output

**Symptom:** The script starts but the terminal shows nothing.

**Fix:** Set encoding environment variables before running:
```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
```

PowerShell may buffer output without these. The `-u` flag on `python.exe -X utf8 -u` also helps.

### 5.3 `OSError: t5-large not found` or `RepositoryNotFoundError`

**Symptom:** Error when loading the model.

**Cause:** The scripts set `TRANSFORMERS_OFFLINE=1` at the top. If the model is not cached locally, it cannot be found.

**Fix:** Run the download command from Step 1.4 **without** the `TRANSFORMERS_OFFLINE` variable set (run it in a fresh terminal where that variable is not set).

### 5.4 `ModuleNotFoundError: No module named 'cross_arch_t5_nib_v48'`

**Cause:** You are not running from the `layercakeogwithdecoder/` directory.

**Fix:**
```powershell
cd "C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready\layercakeogwithdecoder"
```

The import `import cross_arch_t5_nib_v48 as base` uses relative imports from the current directory.

### 5.5 corrMSE Stuck Above 0.005 After P2

**Symptom:** After phase P2 (7000 steps), corrMSE is still above 0.005 and barely decreasing.

**Likely causes:**
1. The backbone is not frozen — gradient is flowing into backbone parameters
2. The per-tap LNs were accidentally made non-trainable when they should be initialized by Stage C
3. A numerical issue in the batch construction (mismatched seeds)

**Diagnostic:**
```powershell
python -c "
import torch, os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import cross_arch_t5_nib_v53 as exp53
import cross_arch_t5_nib_v48 as base

model = exp53.MultiTap6SV_PerTapLN(abi_seed=99, d_abi=4096).to(base.DEVICE)
base.freeze_backbone(model)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Trainable: {trainable:,}  Total: {total:,}')
print(f'Backbone frozen ratio: {(total-trainable)/total*100:.1f}%')
"
```

Expected: `Trainable: ~163,627,009  Total: ~901,266,241  Backbone frozen ratio: ~81.8%`

### 5.6 NIB top-5 Far Below 0.85 Despite Good corrMSE

**Symptom:** corrMSE ≈ 0.003 but NIB top-5 ≈ 0.80 or lower.

**Most likely cause:** `tie_word_embeddings` scaling not applied. Check that `cross_arch_t5_nib_v53.py` has:

```python
if self.t5.config.tie_word_embeddings:
    h_final = h_final * (self.model_dim ** -0.5)
return self.t5.lm_head(h_final)
```

Also verify `self.t5.tie_weights()` is called (via `retie_weights()`) after loading the backbone.

### 5.7 Different corrMSE Value Despite Identical Setup

CUDA non-determinism from cuBLAS matrix multiplication can produce slightly different corrMSE values (±0.0003) across runs. This is expected and does not indicate a bug. NIB top-5 results should vary by ±0.003 or less.

To maximize reproducibility, set:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

But note: this significantly slows training and is not set in the published experiment.

---

## 6. Comparing Your Replication to the Published Result

After your replication run completes, compare against the published values:

```powershell
python -c "
import json

pub = json.load(open('cross_arch_t5_nib_v53_results.json'))
rep = json.load(open('cross_arch_t5_nib_v53_repro_results.json'))  # or your output file

def chk(name, pub_val, rep_val, tol):
    diff = abs(pub_val - rep_val)
    status = 'OK' if diff <= tol else 'DIFF'
    print(f'  {name:<30} pub={pub_val:.6f}  rep={rep_val:.6f}  diff={diff:.6f}  [{status}]')

print('=== REPLICATION COMPARISON ===')
chk('corrMSE', pub['best_corrMSE'], rep['best_corrMSE'], 0.0003)
chk('official top5', pub['nib_official']['mean_top5'], rep['nib_official']['mean_top5'], 0.005)
chk('extended mean', pub['nib_combined']['mean_top5'], rep['nib_combined']['mean_top5'], 0.010)
for seed in ['7777', '1111', '2222', '3333', '4444']:
    chk(f'rng={seed} top5',
        pub['nib_combined']['per_rng_mean_top5'][seed],
        rep['nib_combined']['per_rng_mean_top5'][seed], 0.010)
print()
print('Tolerances: corrMSE ±0.0003, NIB top5 ±0.005 official, ±0.010 per-seed')
"
```

A replication is considered successful if all values are within tolerance. Minor differences due to CUDA non-determinism are expected and acceptable.

---

## 7. Timing Reference

| Stage | Expected Duration (RTX 3080 Laptop) |
|-------|-------------------------------------|
| Stage A (Anchor pre-training) | ~20–25 min |
| Stage C (Native pre-training) | ~20–25 min |
| Stage D P1 (4000 cal steps) | ~30–35 min |
| Stage D P2 (3000 cal steps) | ~25–30 min |
| Stage D P3 (3000 cal steps) | ~25–30 min |
| Stage D P4 (6000 cal steps) | ~55–65 min |
| NIB evaluation (official + extended) | ~15–20 min |
| **Total** | **~190–230 min** |

On faster GPUs (e.g., RTX 4090), total time may be 60–90 min. On slower GPUs or with `BATCH_SV=1`, expect up to 2× longer.

---

## 8. Output Files

After a successful run, the following files are created in the working directory:

| File | Contents |
|------|----------|
| `cross_arch_t5_nib_v53_results.json` | Full result record (overwritten if already exists) |
| `cross_arch_t5_nib_v53_log.txt` | Console log (if Tee-Object was used) |

The results JSON contains the complete architecture spec, all training hyperparameters, official NIB results, extended NIB results, corrMSE history summary, and timing.
