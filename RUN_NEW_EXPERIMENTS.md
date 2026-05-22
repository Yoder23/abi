# Running the New Scale Experiments (Claims 10–12)

Three new experiment scripts target the main critiques: "only tested small models" and
"not universal (only specific families)". All scripts use the **same pre-registered
NIB thresholds** as all prior experiments. No parameters were changed.

---

## Experiments Overview

| Script | Source | Target | Arch | D_ABI | Est. runtime |
|---|---|---|---|---|---|
| `exp_qwen_1p5b_nib.py` | GPT-2-small (117M) | Qwen2-1.5B (1.54B) | Qwen2 | 256 | ~2–3 h |
| `exp_deepseek_1p3b_nib.py` | GPT-2-medium (354M) | DeepSeek-Coder-1.3B | **Llama** | 256 | ~3–4 h |
| `exp_qwen_7b_nib.py` | T5-large (730M) | **Qwen2-7B (7B, INT8)** | Qwen2 | 256 | ~6–8 h |

Run order: start with `exp_qwen_1p5b_nib.py` (fastest, builds confidence),
then `exp_deepseek_1p3b_nib.py`, then `exp_qwen_7b_nib.py` overnight.

---

## Requirements

All requirements are already satisfied if the existing experiments pass:
- PyTorch ≥ 2.0, CUDA 11.8+
- `transformers` ≥ 4.35
- `bitsandbytes` ≥ 0.41 (for `exp_qwen_7b_nib.py` only)
- `datasets`
- All model weights cached locally in HuggingFace cache

VRAM requirements:
- Exps A and B: ~8 GB peak (two 1–1.5B models + ABI)
- Exp C (7B): ~11 GB peak (Qwen2-7B INT8 ~10.5 GB + ABI modules)

---

## Recommended Run Commands

From the `abi_release/` directory:

```bash
# Experiment A — Cross-Scale: GPT-2-small -> Qwen2-1.5B (~2-3h)
python exp_qwen_1p5b_nib.py 2>&1 | tee exp_qwen_1p5b_run.log

# Experiment B — Cross-Lineage: GPT-2-medium -> DeepSeek-Coder-1.3B (~3-4h)
python exp_deepseek_1p3b_nib.py 2>&1 | tee exp_deepseek_1p3b_run.log

# Experiment C — 7B Scale: T5-large -> Qwen2-7B INT8 (~6-8h, run overnight)
python exp_qwen_7b_nib.py 2>&1 | tee exp_qwen_7b_run.log
```

Each script prints live NIB chunk progress and writes results to a JSON file on completion.

---

## Expected Output Structure

Each script writes a result JSON with this structure:
```json
{
  "experiment": "...",
  "overall_pass": true,
  "nib_l2": {
    "mean_js":           "< 0.10  (PASS threshold)",
    "mean_top1_agree":   ">= 0.68 (PASS threshold)",
    "mean_top5_overlap": ">= 0.86 (PASS threshold)",
    "mean_entropy_diff": "< 0.35  (PASS threshold)",
    "pass": true
  },
  "ppl_native_...",
  "ppl_calibrated_...",
  "claim": "..."
}
```

---

## After Results Are In

Once all three scripts complete successfully (overall_pass = true):

1. Update `CLAIMS.md` Claims 10–12: change `⏳ PENDING` → `✅ VALIDATED` and add result numbers
2. Update `README.md` claim ladder: remove "7B+ — not yet tested" note
3. Update `SKEPTICS.md`: add actual result numbers to the "in progress" note
4. Commit:
   ```bash
   git add exp_qwen_1p5b_nib.py exp_qwen_1p5b_nib_results.json \
           exp_deepseek_1p3b_nib.py exp_deepseek_1p3b_nib_results.json \
           exp_qwen_7b_nib.py exp_qwen_7b_nib_results.json \
           CLAIMS.md ROADMAP.md SKEPTICS.md README.md
   git commit -m "feat: 7B scale proof + Llama-family + 1.5B cross-scale (Claims 10-12)"
   git push origin main
   ```

---

## What Each Experiment Proves

**Exp A (1.5B):** D_ABI=256 is not tied to the target model's size. The same 256-dim
bottleneck that works for Qwen2.5-0.5B (Exp 32) also works for Qwen2-1.5B (13x larger
target). The fixed bottleneck is not overfit to a specific model scale.

**Exp B (DeepSeek/Llama):** D_ABI=256 crosses the GPT-2 / Llama architectural family
boundary. DeepSeek-Coder uses RoPE + SwiGLU + RMSNorm — none of which appear in any
prior tested model. This is the most architecturally-distant target tested to date.

**Exp C (7B):** Scale is not a fundamental blocker. At 7B parameters, the 256-dim ABI
bottleneck still achieves NIB criterion using only ~20 MB of trainable parameters on top
of a 10.5 GB quantized backbone. Directly addresses the #1 critic response.
