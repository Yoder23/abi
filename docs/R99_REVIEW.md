# Reproduce the R99 negative result

This packet is for `abi-research-2026-11-05`, not the R7 scientific release.
Expected outcome: `ABI_ACQUISITION_PILOT_FAILED`. Reproduction must not turn
that result into a fluent-core, minimum-information or superiority certificate.

## Required inputs

1. Clone this repository with Git LFS and check out the exact annotated review
   tag. Pull LFS objects. The three failed final learner checkpoints and their
   parent are required; LFS pointer text is not a model.
2. Clone [LayerCake](https://github.com/Yoder23/layercake) into a sibling named
   `layercake_release`, with Git history available. Use host source commit
   `a24458818dc4489b4b977c49bc6ea5b8d089ef24`. No ABI code is placed in that repo.
3. Use the measured Python/package versions in
   `results/review_2026_11_05/r99_manifest.json` as the reference environment.
   The lightweight public API install alone does not install every research
   dependency; see the `extraction` and `dev` extras in `pyproject.toml`.
4. Prepare the public Qwen2-7B-Instruct snapshot at revision
   `f2826a00ceef68f0f2b946d945ecc0477ce4450c`. The raw-evidence verifier needs
   its exact tokenizer; new acquisition requires its approximately 15.24 GB
   source snapshot. Source weights are not bundled with ABI.

The acquisition code uses `huggingface_hub.snapshot_download` with the pinned
model/revision, then loads only that local snapshot. Do not substitute a newer
teacher or tokenizer. Source-file hashes are in the source accounting records.

## Read-only checks

From the ABI checkout:

```bash
git lfs fsck
python scripts/audit_r98_completion.py
python -m experiments.grounded_english_r99.run verify-source
python -m experiments.grounded_english_r99_fp32.run verify
python scripts/run_review_tests.py
python scripts/check_docs.py
```

The completed FP32 verifier rereads all 192 generation rows (48 per system),
all three 384-step logs, final checkpoint hashes/finite tensors, shared exposure,
raw/normalized source text, exact normalization spans and paired bootstrap
comparisons. Its expected scores are 10 ABI-plan, 20 distillation, 27 LoRA and
0 parent out of 48. Raw teacher is 40/48; normalized teacher is 43/48.
These remain mechanical proxies, not human fluency judgments.

`freeze`, `pilot`, and the numerical-audit execution commands are write-once
research commands. Do not run them into existing evidence paths. A new live
replication requires a new output namespace and a separately recorded lineage;
the commands above recompute retained evidence, not independent hardware.

## Published reconstruction

On 2026-11-05, new public Git clones of this review tag and
`layercake-review-2026-11-05` were reconstructed without copying development
assets. The latter tag has the same bound host Python bytes as `a244588`;
its later changes are documentation-only. Required checkpoint objects were
downloaded separately from GitHub LFS.

All 50 manifest assets matched their exact hashes and sizes (1,419,854,427
bytes); all 738 frozen inputs matched. C1 audit, R99 source verification and
the full FP32 evidence verifier succeeded, reproducing
`ABI_ACQUISITION_PILOT_FAILED`. Both checkouts remained clean, documentation
checks passed, and 45 focused/adversarial tests passed in the public checkout.

- [Machine-readable reconstruction receipt](https://github.com/Yoder23/abi/releases/download/abi-research-2026-11-05/public_reconstruction_receipt.json)
  SHA-256: `e82458a4257c09f9a3e00ec9c9aab2bdf80d235c96d42164884b9492ff42bbb7`.
- [Reconstruction script](https://github.com/Yoder23/abi/releases/download/abi-research-2026-11-05/verify_public_review.py).
- [Focused test receipt](https://github.com/Yoder23/abi/releases/download/abi-research-2026-11-05/public-focused-tests.xml).

Scope: the same Windows machine, installed Python dependencies and cached
pinned public teacher tokenizer were reused. This proves public availability
and raw-evidence recomputation for the bounded negative pilot, not a fresh
training run, live generation replication, a dependency-isolated environment,
independent hardware or full-campaign reproduction. The receipt was added to
the release after publication; the annotated source tag was not changed.

## What to challenge

- Every source span must come from the stored teacher response.
- Training must not consume development targets.
- All learner arms must share the same parent and acquisition information.
- Extra plan tokens and unequal trainable parameter counts must stay visible.
- No scalar from an older LayerCake speed certificate may promote these models.
- Source qualification, numerical repair and failed learner quality are three
  separate findings. The failure must not be concealed by the successful
  source normalizer or a positive comparison against the weak parent.

The next hypothesis is improved supplied-fact retention without serializing a
long generated JSON plan. That successor is not implemented or certified by
this packet. See [the full result](R99_ENGLISH_PILOT.md).
