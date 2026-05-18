# ABI — For Skeptics

Direct answers to the hard questions. No spin.

---

## "You cherry-picked the result."

The NIB protocol was fixed before the experiments ran: seed=7777, n=5 chunks, skip first 20 positions, four thresholds. The thresholds (top-5 ≥ 0.860, top-1 ≥ 0.680, JS < 0.100, ent_diff < 0.350) are documented in `PROOF.md` and in the result files. They were not tuned post-hoc.

The extended evaluation (n=25, 5 independent seeds) is reported in full in `PROOF.md`. Three of the five seeds fall below the single-run top-5 threshold (0.860). The mean (0.8549) is above the noise floor. This is disclosed explicitly — including in the README.

---

## "You only tested T5-large on Python code."

Correct. Claim 1 (Path 2C) is specifically T5-large on Python-domain data. It is not a general claim.

The cross-architecture experiments (Claims 2–4) extend the result to:
- GPT-2 → Qwen2.5-0.5B cross-family
- T5-large (enc-dec) → GPT-2-medium (dec-only) cross-architecture
- GPT-2-medium backbone update invariance (WikiText domain)
- T5-large backbone update invariance (WikiText domain)
- Pythia → GPT-2 cross-lineage
- Cross-size: 117M–774M

What has not been tested: 7B+ models, multilingual, medical, legal. These gaps are stated in `CLAIMS.md` and at the top of the README.

---

## "The NIB thresholds are too easy to pass."

Run `python verify_result.py` and look at the corrMSE value (0.003047). See `ABI_ARCHITECTURE.md §5` for the corrMSE floor analysis: the null-space geometry of the `proj_out` projection means that corrMSE = 0.003047 is structurally meaningful, not trivially low.

The seven-objective ablation is in `experiments/abi_ablation_test.py + _results.json`. Raw KL loss (which produces 12.7% lower training loss) gives top-5 agreement **0.030 lower** than corrMSE — demonstrating that NIB top-5 is not trivially maximized by any loss that minimizes perplexity.

---

## "You need a second lab to reproduce this."

Yes. We agree. That's an explicit gap listed in `ROADMAP.md` (v0.3). The code is here so a second lab can try.

What we provide now: every script, every pre-computed result JSON, a standalone verifier (`verify_result.py`) that checks every constant in < 5 seconds on CPU, and a full step-by-step reproduction guide (`ABI_REPRODUCE.md`).

---

## "The ABI module has 163.6M parameters — you're just training a new model."

The ABI module has 163.6M parameters. The frozen T5-large backbone has 730M. ABI does not update any of those 730M. Comparison:

| Approach | Parameters trained |
|---------|-------------------|
| Standard fine-tuning | 730M (backbone) |
| LoRA (r=8) | ~1.7M |
| ABI module | 163.6M |
| ABI calibration only (cross-arch) | ~8M (proj_in + proj_out) |

ABI is not a LoRA-class parameter-efficient method. The claim is not parameter efficiency. The claim is: the backbone is frozen and the domain module is portable. These are different properties.

---

## "Why not just use LoRA?"

LoRA modifies the backbone's effective weights. After LoRA fine-tuning, the base model's behavior changes permanently (or requires the adapter to be applied). ABI's backbone is literally unchanged — you can remove the ABI module and the backbone returns to its exact original behavior.

The portability result (Claim 3: frozen module migrating from T5-large to GPT-2-medium) is not achievable with LoRA because LoRA adapters are tied to the specific weight dimensions of the model they were trained on.

A formal LoRA baseline comparison is planned for v0.3. See `ROADMAP.md`.

---

## "This is just knowledge distillation."

Standard knowledge distillation requires a teacher forward pass at training time and typically distills into a smaller or equal model by minimizing logit-level KL divergence. ABI:

1. Uses corrMSE — a hidden-space objective, not logit-level distillation
2. Does not use a teacher forward pass at inference time
3. Transfers a frozen module to a different architecture (not distilling into the same architecture)
4. Achieves non-inferiority (not just minimizing distillation loss)

The ablation in `experiments/abi_ablation_test.py` tests raw KL loss, weighted corrMSE, logit-MSE, and corrMSE. corrMSE is the only objective that produces NIB PASS. logit-MSE is catastrophic (top-5 ~ 0.6).

---

## "The Procrustes rotation is doing all the work."

The Procrustes alignment experiment is in `experiments/procrustes_full_nib.py + _results.json`. Procrustes alone (without KD calibration) does not achieve NIB PASS. Both components are required for the cross-architecture results. The contribution of each component is separated in the ablation.

---

## "I can't trust result files you pre-computed."

Run everything from scratch:

```bash
# Takes ~4 hours on RTX 3080 Laptop (16 GB VRAM)
python run_abi.py

# Takes ~8 minutes — cross-architecture enc-dec → dec-only
python cross_arch_enc_dec_nib.py

# Takes ~10 minutes — backbone-update invariance
python cross_arch_t5_succession.py
```

See `ABI_REPRODUCE.md` for exact expected output at every stage.

---

## "How do I know verify_result.py isn't just returning True?"

`verify_result.py` contains embedded expected constants. Run it and look at the output — each check prints the actual value, the expected value, and a pass/fail. You can also open the file and read the 25 explicit assert statements directly.

---

*If your concern is not addressed here, open a GitHub issue. Label it `skeptic-question`.*
