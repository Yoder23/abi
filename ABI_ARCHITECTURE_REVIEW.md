# ABI Architecture Review

Review date: 2026-06-04

Scope: pause further experiment runs and audit whether the current ABI
structure is architecturally sound for lossless, model-agnostic domain
knowledge transfer.

## Verdict

The current ABI architecture is sound for a scoped claim:

> A frozen source domain operator can be copied, rotated into a target ABI
> basis, and calibrated through a small target-side interface to match a native
> target domain oracle on certified model pairs.

It is not yet architecturally sufficient for the stronger claim:

> Arbitrary GPT5 domain knowledge can be migrated losslessly into a GPT6 target
> without target-side domain learning or full retraining.

The central blocker is the Phase C native target ABI oracle. In the strongest
current recipes, Phase C trains a target-side ABI/domain oracle on the target
domain, then Phase D initializes from and distills against that oracle. This is
a useful transfer certificate, but it proves target-compatible replay of a
copied source ABI payload more than it proves standalone source-knowledge
extraction and lossless migration. In short, the target native oracle remains
the main architectural dependency to reduce or remove.

## Implemented Since Review

- Added `abi.artifacts` with deterministic module hashing, ABI artifact
  construction, compatibility-certificate construction, and cost-ledger
  construction.
- Wired `exp_generic_causal_nib_v2.py` so future generic v2 runs emit
  `abi_artifact`, `compatibility_certificate`, and `cost_ledger` result fields.
- The artifact records source-domain hashes, rotated copied-core hashes before
  and after calibration, frozen-core claims, copied payload parameter counts,
  alignment metadata, and target-side trainable groups.
- The compatibility certificate now explicitly leaves source-preservation and
  oracle-light transfer as open gates instead of hiding them behind NIB.
- The cost ledger now counts measured Phase A, Phase C, Phase D, and NIB
  timings for future runs, plus phase steps, token counts, and parameter counts.
- Added an opt-in `ABI_SOURCE_PRESERVATION_EVAL` probe for future generic v2
  runs. It caches source next-token decoded surfaces before optional source
  release, compares them with the final calibrated target on the same raw
  prompts, and feeds measured source-preservation results into the certificate.
- Added `ABI_ORACLE_MODE` for future generic v2 runs. The default remains
  `full_native_target_oracle`; `target_base_interface` trains only the target
  ABI interface with the domain path disabled; `base_target_reference` skips
  target ABI/reference training. Result certificates now distinguish target
  native-oracle NIB from oracle-light target-reference NIB.

## Implemented Pipeline

The generic v2 runner implements this path:

1. Train a source ABI/domain module on the domain corpus while the source
   backbone stays frozen.
2. Train a native target ABI oracle on the same target-domain corpus while the
   target backbone stays frozen.
3. Fit a cross-tokenizer ABI alignment map from paired sentence embeddings.
4. Rotate the source domain MLP into the target ABI basis.
5. Build the calibrated target from the frozen target backbone/head, the copied
   rotated source domain core, and target-side ABI interface modules.
6. Optionally initialize the target interface from the Phase C native oracle.
7. Calibrate target-side interface parameters with KD, rank/top-set losses, and
   optional post-hoc logit scale or bias.
8. Select checkpoints on validation/post-hoc data and evaluate final NIB on the
   held-out eval split when split controls are enabled.

This is a clean and inspectable architecture. The copied domain MLP boundary is
real, the target backbone is frozen, and the latest proof-layer logic preserves
a stricter no-leakage split discipline.

## Architecturally Strong Pieces

- Frozen target backbone and head create a real compute-saving envelope.
- The copied source domain MLP core gives ABI a portability property that LoRA
  does not attempt to provide.
- Procrustes alignment gives a model/tokenizer-agnostic coordinate bridge.
- Native target-interface initialization is a principled fix for random
  interface search instability.
- EMA and validation/post-hoc split selection make the strongest runs more
  repeatable than the earlier ad hoc probes.
- The generated proof docs correctly avoid claiming lossless arbitrary-model
  transfer.

## Architectural Gaps

1. Native oracle dependency: the current strongest recipe still requires a
   target native ABI/domain oracle. For a GPT5 to GPT6 production story, this
   must become an oracle-light or no-target-domain-oracle path.
2. Evaluation target: NIB compares calibrated transfer to the target native
   oracle, not directly to the source model's domain knowledge. That validates
   target-oracle imitation, not full source-knowledge preservation.
3. Alignment capacity: sentence-level mean-pooled Procrustes is a global linear
   map. It is useful, but it cannot be assumed sufficient for token-level,
   positional, multilingual, factual, or tool/code knowledge transfer.
4. Payload capacity: the copied payload is a single ABI-space MLP plus optional
   lightweight residuals. This is unlikely to be a complete carrier for
   arbitrary domain knowledge without richer composition, routing, or memory.
5. Metric scope: NIB is a strong distributional/logit agreement test, but it is
   not enough for production claims. We still need domain QA, task accuracy,
   source-behavior retention, forgetting, safety, and calibration stress tests.
6. Post-hoc output calibration: validation-only logit scaling is legitimate
   under the split discipline, but it is output-distribution repair, not
   evidence by itself that the ABI payload transferred knowledge.
7. Selective-transfer gap: the current proof stack does not yet require a
   selected-domain pass plus an off-domain noninterference pass. For a
   GPT5-to-GPT6 migration story, targeted domain transfer must prove both that
   the intended payload moved and that unrelated target behavior was not
   overwritten or leaked.
8. Cost accounting: current docs count trainable parameters and local runtime,
   but they do not yet fully account for Phase A, Phase C, calibration FLOPs,
   repeated sweeps, baseline parity, or failed-run search cost.
9. Certificate gap: the formal A1-A5 compatibility theorem exists, but the
   runner does not yet emit a complete measured compatibility certificate for
   every successful result.

## Stop Rule Before More Experiments

Do not start more frontier runs until the next experiment can answer one of
these architectural questions:

- Does the source domain payload preserve source behavior after transfer?
- Does the selected domain transfer without measurable off-domain leakage or
  unrelated target-behavior overwrite?
- Can the target interface work without training a full target native domain
  oracle on the target domain?
- Do compatibility metrics predict held-out transfer quality before NIB?
- Does the result survive a task-level domain benchmark, not only logit NIB?
- Is the measured compute saving still valid after counting all prerequisite
  phases and search cost?

If an experiment does not answer one of those questions, it is lateral tuning.

## Required Architecture Work

1. Add a first-class `ABIArtifact` concept that serializes the copied source
   payload, source metadata, alignment metadata, freeze status, and target
   calibration recipe. This separates what is migrated from what is trained.
   Status: implemented for future generic v2 result artifacts.
2. Add source-preservation evaluation: compare transferred target behavior
   against the source domain teacher on tokenizer-independent prompts or tasks.
   Status: first opt-in decoded next-token surface probe implemented; stronger
   task-level preservation benchmarks remain open.
3. Add oracle-light calibration: distinguish full Phase C native-oracle
   distillation from modes that use only general target ABI initialization,
   unlabeled alignment data, or a limited domain calibration budget.
   Status: mode infrastructure implemented for generic v2; passing
   oracle-light evidence remains open.
4. Emit an ABI compatibility certificate for each run: alignment residuals,
   domain-equivariance probes, interface residuals, top-k margins, JS/entropy
   curves, and pass/fail thresholds.
5. Add task-level domain benchmarks for WikiText plus at least code, SQL, and
   factual/domain QA before treating NIB as adoption evidence.
6. Add selective-transfer evaluation: for each selected domain, run a paired
   off-domain control against the frozen target base/reference and record both
   selected-domain transfer and off-domain no-leakage metrics.
   Status: opt-in runner fields and generated proof gates are implemented;
   repeat-certified strict selective-transfer evidence remains open.
7. Build a cost ledger that includes Phase A, Phase C, Phase D, post-hoc
   selection, repeats, failed sweeps, and matched baselines.
   Status: implemented for per-run measured phases; failed-sweep and matched
   baseline aggregation remain open.
8. Add artifact/provenance tests that verify copied-core weights remain frozen
   where the result claims frozen-core transfer.

## Stronger GPT5-To-GPT6 Gate

A credible next-generation claim needs all of the following:

- Frozen target base model and frozen copied source payload.
- No full target domain oracle, or a clearly bounded oracle-light path.
- Source-preservation pass on held-out domain tasks.
- Selective-transfer pass showing the chosen domain moves while off-domain
  target behavior remains preserved.
- Target-native NIB pass on held-out data.
- External task pass with bounded accuracy loss.
- Cost ledger showing real savings after all phases are counted.
- Matched baselines against LoRA, adapters, partial fine-tuning, full
  fine-tuning, and conventional distillation.
- Repeat certification across seeds, streams, domains, and model families.

Until those gates pass, the correct claim is scoped frozen-core ABI transfer,
not lossless universal domain migration.
