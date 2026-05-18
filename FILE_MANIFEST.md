# ABI Open-Source File Manifest

This document defines exactly which files belong in the published ABI repository. Everything not listed here is either LayerCake-specific infrastructure or development-only archive material.

---

## Repo Identity

**Repository name:** `abi-transfer`  
**Tagline:** *Autonomous Basis Injection — domain behavioral reconstruction across LLM architectures*

---

## INCLUDE — Production Package (`abi/`)

| File | Purpose |
|------|---------|
| `abi/__init__.py` | Package entry point |
| `abi/models.py` | `AnchorABI`, `CandidateABI`, `DomainModule` |
| `abi/training.py` | Stage A (domain), Stage C (KD calibration), Stage D (Procrustes) |
| `abi/evaluation.py` | NIB evaluation logic |

---

## INCLUDE — Entry Points

| File | Purpose |
|------|---------|
| `run_abi.py` | Full Path 2C training + NIB evaluation (same-backbone T5-large) |
| `verify_result.py` | Standalone result verifier (no GPU, < 5 s) |
| `wikitext_cache.py` | WikiText-2 data loader (used by all experiment scripts) |
| `reproduce_abi.py` | Cross-size paste reproduction harness |

---

## INCLUDE — Locked Experiment Scripts (one per validated claim)

| File | Claim | Result file |
|------|-------|-------------|
| `cross_arch_t5_nib_v53.py` | Path 2C — T5-large same-backbone NIB PASS | `cross_arch_t5_nib_v53_results.json` |
| `cross_family_nib.py` | Exp 32 — GPT-2-small → Qwen2.5-0.5B cross-family NIB PASS | `cross_family_nib_results.json` |
| `cross_arch_enc_dec_nib.py` | Exp 39 — T5-large (enc-dec) → GPT-2-medium (dec-only) NIB PASS | `cross_arch_enc_dec_nib_results.json` |
| `cross_arch_t5_succession.py` | Exp 40 — T5-large backbone-update invariance (304% efficacy) | `cross_arch_t5_succession_results.json` |
| `scale_validation_test.py` | GPT-2-medium 354M backbone-update invariance (65% efficacy) | `scale_validation_results.json` |
| `succession_test_v2.py` | 3-round GPT-2 succession stress test | `succession_results_v2.json` |
| `calibration_scaling_law_b.py` | Exp 35b — calibration step scaling law (R² ≈ 0.97) | `calibration_scaling_law_b_results.json` |
| `cross_lineage_transfer_test.py` | Cross-lineage Pythia → GPT-2 transfer | `cross_lineage_results.json` |
| `cross_size_large_nib_v9.py` | Cross-size 117M → 774M decoder NIB PASS | `cross_size_large_nib_v9_results.json` |
| `multi_domain_atlas.py` | Multi-domain atlas (Python + WikiText + SQL) | `multi_domain_atlas_results.json` |

---

## INCLUDE — Locked Result Files

| File | Experiment | Key metric |
|------|-----------|------------|
| `cross_arch_t5_nib_v53_results.json` | Path 2C (45AS) | top-5 = 0.8725 PASS |
| `cross_family_nib_results.json` | Exp 32 | top-5 = 0.8701 PASS |
| `cross_arch_enc_dec_nib_results.json` | Exp 39 | top-5 = 0.8699 PASS |
| `cross_arch_t5_succession_results.json` | Exp 40 | efficacy = 304.3% PASS |
| `scale_validation_results.json` | Scale (GPT-2-medium) | efficacy = 65.3% PASS |
| `calibration_scaling_law_b_results.json` | Exp 35b | R² ≈ 0.97 |
| `cross_lineage_results.json` | Cross-lineage | PASS |
| `cross_size_large_nib_v9_results.json` | Cross-size | PASS |
| `procrustes_nib_results.json` | Procrustes NIB baseline | reference |
| `analytical_calibration_results.json` | Analytical calibration | reference |

---

## INCLUDE — Documentation

| File | Purpose |
|------|---------|
| `README.md` | Primary landing page — core claim, results table, reproduction |
| `PROOF.md` | Formal proof structure — protocol, evidence, what remains open |
| `ABI_EXPERIMENTS.md` | Complete experimental ledger — all 8 validated claims |
| `ABI_ARCHITECTURE.md` | Architecture spec — tap configuration, d_abi rationale, null-space geometry |
| `ABI_REPRODUCE.md` | Step-by-step reproduction guide |
| `ABI_START_HERE.md` | 10-minute onboarding for new contributors |
| `OPEN_SOURCE_README.md` | Developer reference — component table, honest claims |
| `FILE_MANIFEST.md` | This file |
| `.gitignore` | Excludes checkpoints, logs, dev scripts |
| `requirements.txt` | Python dependencies |

---

## INCLUDE — Supporting Experiments (context only)

| File | Purpose |
|------|---------|
| `abi_ablation_test.py` | Objective ablation (corrMSE vs KL vs logit-MSE) |
| `abi_ablation_results.json` | Results |
| `knowledge_non_interference.py` | Domain isolation test |
| `knowledge_non_interference_results.json` | Results |
| `non_inferiority_benchmark.py` | NIB protocol reference implementation |
| `non_inferiority_results.json` | Baseline NIB result |
| `nib_geometry_diagnostic.py` | ABI geometry analysis |
| `nib_geometry_diagnostic_results.json` | Results |
| `generation_equivalence_test.py` | Generation-level equivalence test |
| `generation_equivalence_results.json` | Results |
| `precision_parity.py` | fp32 vs bf16 parity |
| `precision_parity_results.json` | Results |
| `method_robustness_sweep.py` | Robustness across seeds/hyperparams |
| `method_robustness_results.json` | Results |
| `ranking_quality_analysis.py` | Top-k ranking analysis |
| `ranking_quality_results.json` | Results |
| `transition_zone_multiseed.py` | Calibration step transition zone |
| `transition_zone_results.json` | Results |
| `abi_scaling_law.py` | ABI capacity vs model size |
| `abi_scaling_results.json` | Results |
| `analytical_calibration.py` | Analytical step count predictor |
| `calibration_budget_floor.py` | Minimum calibration floor |
| `calibration_budget_floor_results.json` | Results |

---

## DO NOT INCLUDE — Archive (development only)

### Iterative experiment scripts (superseded by locked final versions)
- `cross_arch_t5_nib_v1.py` through `cross_arch_t5_nib_v60.py` (except v53)
- All corresponding `_log.txt` and `_results.json` files
- `cross_arch_path3a_*.py`, `cross_arch_path3a_*.json`, `cross_arch_path3a_*.pt`
- `cross_arch_nib_v2.py`, `cross_arch_nib_v3.py`
- `cross_size_large_nib_v1.py` through `cross_size_large_nib_v8.py`
- `cross_size_nib.py`, `cross_size_nib_v2.py`, `cross_size_nib_v3.py`
- `cross_arch_qwen2_v*.py`
- `cross_model_transfer.py`, `cross_model_transfer_v2.py`

### All run logs and error logs
- `*_run.log`, `*_err.log`, `*_log.txt`, `atlas_run_*.log`, `atlas_err_*.log`

### Intermediate model checkpoints
- All `.pt`, `.ckpt`, `.safetensors` files (except baseline_5k_ckpt.pt if needed for reference)
- `cross_arch_t5_nib_v*_calibrated.pt`

### Development / diagnostic / patch scripts
- `_fix_gen_test.py`, `_fix_gen_test2.py`
- `_patch_*.py` (all patching scripts)
- `check_*.py`, `debug_*.py`, `diagnose_*.py`, `diag_*.py`
- `inspect_*.py`, `compare_*.py` (most comparison scripts)

### LayerCake LM system (separate repo)
- `layercake_model*.py`, `layercake_data.py`, `layercake_decoder.py`
- `train_layercake_*.py`, `train_domains_*.py`
- `layercake_domains/`
- `cortana*.py`, `cortical_swarm/`, `cortex_governor.py`
- `autonomous_moa.py`, `moa_*.py`
- `mcp_*.py`, `auto_router*.py`
- All LAYERCAKE_*.md, CORTANA_*.md, MOA_*.md docs

### Investor / business / pre-publication material
- `INVESTOR_*.md`, `BUSINESS_MODEL.md`, `PATENT_*.md`, `FULL_PLATFORM_*.md`

---

## Validated Claim Ladder (summary)

| Claim | Experiment | Top-5 / Efficacy | Status |
|-------|-----------|-----------------|--------|
| Same-backbone NIB (T5-large) | Path 2C / 45AS | 0.8725 | **LOCKED** |
| Cross-family decoder-only (GPT-2 → Qwen2.5) | Exp 32 | 0.8701 | **LOCKED** |
| Cross-architecture enc-dec → dec-only | Exp 39 | 0.8699 | **LOCKED** |
| Backbone-update invariance (dec-only, GPT-2-medium) | scale_validation_test | 65.3% | PASS |
| Backbone-update invariance (enc-dec, T5-large) | Exp 40 | 304.3% | **LOCKED** |
| Cross-lineage (Pythia → GPT-2) | cross_lineage | PASS | PASS |
| Cross-size (117M → 774M decoder) | cross_size_large_nib_v9 | PASS | PASS |
| Calibration scaling law | Exp 35b | R² ≈ 0.97 | PASS |
