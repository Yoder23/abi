# Contributing to ABI

Thank you for your interest. ABI is a research preview. Contributions that improve reproducibility, extend coverage, or correct errors are welcome.

---

## What We Want

### High priority
- **Independent reproductions** — run `verify_result.py` or the full experiments on different hardware and report your numbers
- **7B+ model experiments** — we lack the hardware; if you have it, the methodology is in `ABI_REPRODUCE.md`
- **Non-Python domains** — code, medical, legal, multilingual; compare against ABI on Python
- **LoRA / adapter baselines** — formal comparison of ABI portability vs. LoRA portability
- **Bug reports** — if a script crashes or produces wrong output, open an issue

### Medium priority
- Documentation improvements (clarity, typos, missing steps)
- Additional ablations using the experimental framework
- Tests (see `tests/` directory)

### Lower priority / out of scope
- Production serving infrastructure
- New architectures not yet supported by the ABI package
- UI / demo polish

---

## What We Don't Want

- Retroactive changes to the NIB protocol or thresholds (the protocol is immutable for existing claims)
- Result files modified post-hoc
- Scripts that overwrite the locked result JSONs (`cross_arch_t5_nib_v53_results.json`, `cross_family_nib_results.json`, `cross_arch_enc_dec_nib_results.json`, `cross_arch_t5_succession_results.json`)
- PRs that expand the scope of existing claims without the full NIB validation pipeline

---

## How to Contribute

### For reproduction reports
1. Run `python verify_result.py` or one of the core experiment scripts
2. Open an issue with label `reproduction-report`
3. Include: hardware, OS, Python version, exact command, full output

Use the reproduction report issue template — it will prompt you for the right fields.

### For bug reports
Use the bug report issue template. Include a minimal reproduction case.

### For new results (new model, new domain)
Use the new model result issue template. Include:
- Which claim you are extending
- Your NIB scores (all four criteria)
- Hardware used
- Exact command to reproduce

### For code changes
1. Fork the repository
2. Create a branch: `git checkout -b my-feature`
3. Make your changes
4. Run `python verify_result.py` to confirm the locked result is unaffected
5. Run `ruff check .` (install with `pip install ruff`) — the CI will enforce this
6. Open a PR with a clear description of what you changed and why

---

## Code Style

- Python 3.10+
- `ruff` for linting (`ruff check .`)
- Line length: 100 characters
- No type annotations required for experimental scripts
- The `abi/` package code should remain clean and readable

---

## PR Requirements

Every PR must:
- Pass the CI lint check (`ruff check .`)
- Pass `python verify_result.py` (the locked Path 2C result must not change)
- Have a clear description of what changed and why
- Not modify locked result files

---

## Governance

This is a single-author research preview. Significant architectural changes require discussion in an issue before a PR. Open an issue first if you plan a major change.

---

## Code of Conduct

Be direct, honest, and kind. Disagreement about scientific claims is welcome; personal attacks are not.

---

*Questions? Open an issue. Label it `question`.*
