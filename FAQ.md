# ABI — FAQ

---

## General

### What is ABI?

Autonomous Basis Injection is a method for domain transfer across LLM architectures. A domain module trained on a frozen backbone can be migrated to a second model — with a different architecture, tokenizer, and vocabulary — by calibrating only interface projections. The backbone weights in both models remain entirely unchanged.

### What is the core result?

Two independently initialized ABI modules on a frozen T5-large backbone pass the Non-Inferiority Benchmark (NIB) with top-5 token agreement = 0.8725. A T5-large-trained frozen domain module also transfers to GPT-2-medium (top-5 = 0.8699), across the encoder-decoder ↔ decoder-only architectural boundary.

### What is the NIB?

The Non-Inferiority Benchmark evaluates whether a candidate model's logit distribution is non-inferior to an anchor's on held-out text. Four criteria must all pass simultaneously:

| Criterion | Threshold |
|-----------|-----------|
| Top-5 token agreement | ≥ 0.860 |
| Top-1 token agreement | ≥ 0.680 |
| Jensen-Shannon divergence | < 0.100 |
| Entropy difference | < 0.350 |

Protocol: seed=7777, n=5 chunks of 512 tokens, skip first 20 positions per chunk.

### What does "non-inferior" mean?

It means the candidate's logit distributions are statistically indistinguishable from the anchor's at the NIB thresholds — not that they are identical. Non-inferiority is the claim, not superiority.

---

## Reproduction

### How do I verify the result in < 5 seconds?

```bash
git clone https://github.com/Yoder23/abi
cd abi
pip install -e .
python verify_result.py
```

No GPU required. No model download required. `verify_result.py` checks the pre-computed result file against embedded expected constants.

### How do I run the full experiment from scratch?

See `ABI_REPRODUCE.md`. It takes ~4 hours on a 16 GB GPU. You need `t5-large` (~2.9 GB).

### Do I need a GPU?

`verify_result.py` — CPU only, < 5 seconds.  
Training / full reproduction — yes, ≥ 10 GB VRAM recommended (tested: RTX 3080 Laptop 16 GB).  
Cross-architecture experiments — yes, ~8–10 minutes on the same GPU.

### What if I get different numbers?

First check: are you using seed=7777, 5 chunks of 512 tokens, skipping the first 20 positions? These are required. If you're using different data, a different model version, or a different random seed, results will differ. Open an issue with your setup details.

---

## Technical

### Why is the ABI module so large (163.6M parameters)?

The module needs to span 6 × 1024-dim = 6144 inputs (6 tap layers × T5 d_model), project to 4096 dims for the domain sub-network, and project back to 1024. The 4096-dim inner size creates a null space that is used explicitly for calibration geometry (see `ABI_ARCHITECTURE.md §5`).

### Why corrMSE and not KL divergence?

Seven objectives were ablated. Raw KL loss produces 12.7% lower training loss than corrMSE — but top-5 NIB agreement is 0.030 lower. The logit-level KL gradient interferes with the top-5 token geometry that NIB measures. corrMSE (hidden-space MSE on the correction vector) avoids this interference. Full ablation results: `experiments/abi_ablation_test.py + _results.json`.

### What does "backbone frozen" mean precisely?

`requires_grad = False` on every parameter of the T5-large (or GPT-2) backbone from the first training step to the last. The backbone's weights are identical before and after ABI training. You can verify this in the training code (`abi/training.py`).

### How does cross-architecture transfer work?

1. Train ABI module on source model (e.g., T5-large), backbone frozen throughout.
2. Extract sentence-level mean-pooled ABI representations from both source and target models on 2000 shared sentences.
3. Compute orthogonal Procrustes rotation matrix mapping source representations to target representation space.
4. Initialize `proj_in` and `proj_out` from Procrustes result; fine-tune them for 1200 steps with KD calibration. The domain module weights are NOT updated.
5. Evaluate with NIB in target model's native vocabulary.

### What is backbone-update invariance?

After training an ABI module on a backbone, the backbone is fine-tuned for 1000 steps on new data (WikiText-2). The pre-trained ABI module is then applied zero-shot to the updated backbone. Transfer efficacy = (raw_backbone_ppl − zero_shot_ppl) / (raw_backbone_ppl − oracle_ppl). For T5-large: 304.3%. For GPT-2-medium: 65.3%. Both exceed the 50% threshold.

---

## Scope

### Does this work on LLaMA, Mistral, or other 7B+ models?

Not yet tested. GPU constraint (RTX 3080 Laptop 16 GB) prevented 7B+ experiments in this release. This is an explicit gap in `CLAIMS.md` and `ROADMAP.md`.

### Does this work for languages other than English?

Not yet tested. The Python-domain experiments use English code/text. Multilingual support is on the roadmap.

### Is this production-ready?

No. Research prototype. No inference optimization, no serving infrastructure, no security hardening. See `CLAIMS.md` for the full list of what is not claimed.

---

## Contributing

### How can I help?

See `CONTRIBUTING.md`. The highest-value contributions right now are:
1. Independent reproduction on different hardware
2. Running the experiments on 7B+ models
3. Testing on non-Python domains

### Can I use this in my paper?

Yes. Apache 2.0. Please cite the repository — see the `CITATION.cff` and the `README.md` citation block.

---

*Not answered here? Open a GitHub issue.*
