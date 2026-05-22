# ABI Roadmap

This is an honest roadmap. Items are ordered by priority. Timelines are approximate.

---

## v0.1.0 — Research Preview ✅ (released 2026-05-18)

**Goal:** Strangers can verify the result in 5 seconds, reproduce it in a few hours, and understand exactly what is and is not claimed.

- [x] 4 locked core results (Path 2C, Exp 32, Exp 39, Exp 40)
- [x] 9 claim scripts + result JSONs
- [x] 13 supporting experiment scripts + result JSONs
- [x] `verify_result.py` standalone verifier (< 5 sec, no GPU)
- [x] `run_abi.py` full reproduction entry point
- [x] `ABI_REPRODUCE.md`, `PROOF.md`, `ABI_ARCHITECTURE.md`, `ABI_EXPERIMENTS.md`
- [x] README with honest "not claimed" table, claim ladder, claim-to-file map
- [x] `CLAIMS.md`, `SKEPTICS.md`, `FAQ.md`, `CONTRIBUTING.md`
- [x] Apache 2.0 LICENSE, CITATION.cff, pyproject.toml
- [x] GitHub Actions CI (verify + lint)
- [x] Issue templates (reproduction report, new model result, bug report)

---

## v0.2.0 — Reproducibility & Baselines (target: ~2 months)

**Goal:** An independent lab can reproduce every validated claim from scratch.

- [ ] Formal LoRA baseline comparison (portability: can a LoRA adapter migrate the same way?)
- [ ] Adapter baseline comparison (prefix tuning, IA³)
- [ ] Full test suite (`tests/` with pytest) covering NIB evaluation, data pipeline, model forward pass
- [ ] Colab notebook: `ABI_Verify.ipynb` — run `verify_result.py` in Colab (no local install)
- [ ] Docker image on Docker Hub (`yoder23/abi:v0.2.0`)
- [ ] MkDocs documentation site
- [ ] WikiText-103 domain experiment (extend beyond Python code)
- [ ] At least one reproduction report from hardware different from RTX 3080 Laptop

---

## v0.3.0 — Scale & Scope Extension (target: ~4 months)

**Goal:** Honest data at 7B+ scale and at least one non-English domain.

- [x] 7B+ experiment — **IN PROGRESS**: `exp_qwen_7b_nib.py` written, running (Qwen2-7B INT8, T5-large → 7B, Claims 10–12)
- [x] Llama-family experiment — **IN PROGRESS**: `exp_deepseek_1p3b_nib.py` written (GPT-2-med → DeepSeek-Coder-1.3B, first Llama-arch test)
- [x] 1.5B scale experiment — **IN PROGRESS**: `exp_qwen_1p5b_nib.py` written (GPT-2-small → Qwen2-1.5B)
- [ ] Non-Python domain (medical / legal / multilingual) NIB evaluation
- [ ] Multi-domain ABI: multiple simultaneous domain modules on one backbone
- [ ] Benchmark harness: automated comparison against LoRA, adapters, fine-tuning at matched compute
- [ ] Result registry: structured JSON ledger of all contributed reproduction results
- [ ] Zenodo DOI for the dataset / result artifacts
- [ ] Hugging Face model hub: upload ABI checkpoints for the 4 core experiments

---

## v0.4.0 — Paper Draft (target: ~6 months)

**Goal:** Enough material for an arXiv preprint.

- [ ] Full ablation covering 10+ objectives (extending current 7-objective ablation)
- [ ] Statistically rigorous multi-seed evaluation for all 9 claims (not just Claim 1)
- [ ] Independent reproduction by at least one external contributor
- [ ] Formal analysis of the null-space geometry (extending `ABI_ARCHITECTURE.md §5`)
- [ ] arXiv preprint draft

---

## v0.5.0 — Production Architecture Sketch (target: ~9 months)

**Goal:** A concrete design for how ABI could be deployed in a multi-tenant serving scenario.

- [ ] Production architecture sketch (not implementation)
- [ ] Latency analysis: ABI module overhead vs. baseline
- [ ] Memory analysis: hot-swap cost at inference time
- [ ] Tiny demo: one frozen backbone, two ABI modules, hot-swapped in a serving loop

---

## Non-Goals (will not be in this repository)

- A general fine-tuning library (use Hugging Face PEFT for that)
- A production serving system
- Support for closed-weight models

---

## How to Influence the Roadmap

Open a GitHub issue with label `roadmap` and describe what you want and why. Items with demonstrated external interest will be prioritized.

---

*v0.1.0 released 2026-05-18. Roadmap subject to change based on results and community feedback.*
