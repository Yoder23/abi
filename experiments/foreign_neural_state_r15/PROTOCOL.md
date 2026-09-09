# R15A foreign neural-state extraction protocol

## Question

R15A asks whether a generic frontend can recover a bounded capability directly
from the parameter change created when a conventional open-weight model learns
that capability. It does not change the sealed R11 package, executor, codecs,
or recipient mechanism.

The registered causal path is:

```text
frozen Qwen before -> ordinary rank-16 output LoRA learning -> Qwen after
                   -> effective 8 x 896 output-weight delta only
                   -> frozen generic frontend
                   -> unchanged R11 package
                   -> frozen Pythia, Qwen2, and T5 recipients
```

R15 is hierarchical. R15B (activation deltas), R15C (after weights only), and
R15D (after activations only) remain closed unless R15A passes.

## Public development history

Every result is retained; only v7 qualifies the frontend.

- v1 and v2 established that underfit source learning events do not support a
  useful frontend.
- v3 made all 160 source atomic events exact but the kernel frontend recovered
  only 11/32 development capabilities.
- v4's six-direction bilinear frontend recovered 31/32.
- v5 recovered 60/64; all four errors were structurally invalid affine pairs.
- v6 constrained output to valid affine pairs and recovered 63/64. Additional
  epochs and three regularization settings reproduced the same single error.
- v7 fixed the measured information-discarding defect: it reads all 24 atomic
  transition constraints from the weight delta and selects one of 32 valid
  affine codes independently per operator. On a completely fresh public split,
  Qwen learned 320/320 atomic events exactly and the frontend recovered 64/64
  development capabilities (384/384 canonical labels).

The v7 frontend has 21,696 learned readout weights plus 192 biases. It was
trained only on 256 public `(weight delta, capability label)` pairs. At
extraction it receives 7,168 floating-point delta values and receives zero
prompts, answers, outputs, logits, activations, oracle calls, or candidate
program searches. The full public frontend is exactly retrainable from the
consolidated 320-event public tensor dataset.

## Secret boundary

Before the secret is generated, the following are frozen in Git and by hash:

- the R11 tag, commit, binding manifest, executor, package schema, host codecs,
  and three recipient revisions;
- Qwen model revision, targeted output-LoRA architecture, initialization,
  optimizer, full-batch 24-row atomic curriculum, learning rate, and steps;
- the public v7 receipt, consolidated delta dataset, frontend tensor artifact,
  frontend architecture, and all frontend parameters;
- eight held-out capabilities, all split seeds and sizes, package and control
  gates, strict verifier, live replay, and physical extraction sandbox;
- the allowed neural input: only the effective before/after output-weight delta.

The secret reveal is not mounted into the extractor. Each extraction runs in a
network-isolated WSL mount/PID/IPC/UTS namespace after `pivot_root` into a tmpfs
root. The old root and Windows drive are unmounted. The allowlisted capsule
contains only a generic pure-stdlib worker, the frozen frontend specification
and tensors, and one anonymous delta. It contains no capability ID, reveal,
operation table, prompt, answer, behavior row, source-success ledger, or ABI
package.

## Held-out data and controls

Eight independently derived capabilities are required. For every capability:

- Qwen is trained on all 24 depth-one operator/state cases from the identical
  before state;
- R15 receives only the 7,168-value effective weight delta;
- R14 receives 256 non-atomic black-box probability observations as the frozen
  behavioral baseline;
- the emitted package is evaluated on 10,000 prompt-disjoint programs at
  depths 13--18 and 1,000 order counterfactual rows;
- each frozen recipient evaluates 512 rows under BASE, AFTER, BEFORE, WRONG,
  ZERO, RANDOM, SHUFFLED, REMOVED, BACKEND_REMOVED, CODEC_REMOVED, and RESTORED;
- Qwen's before, after, wrong, random, shuffled, removed, and restored adapter
  states are executed live and retained as raw probability rows; and
- zero, wrong-capability, random, and shuffled delta interventions are checked
  independently of stored verdict fields.

## Pass gates

Strict recomputation must establish all of the following:

- 8/8 Qwen learning events are 24/24 exact after training;
- the frozen, physically isolated weight-delta frontend recovers 8/8 exact
  latent capabilities with positive per-operator margins;
- all eight packages score 10,000/10,000 on unseen programs and 1,000/1,000 on
  order counterfactuals;
- wrong, random, and shuffled delta controls are at most 0.30 accuracy, while a
  zero delta is rejected as non-information;
- source removal is exactly equal to before and restoration exactly equals
  after;
- Pythia, Qwen2, and T5 each score 1.0 in AFTER and RESTORED for every package,
  all registered negative controls are at most 0.30, and removal conditions are
  byte-for-byte equal to BASE;
- recipient optimizer steps are zero; source model, recipient weights, codecs,
  R11, frontend, deltas, packages, raw rows, and manifests remain hash-bound;
  and
- a separate live verifier reloads the source adapters, reruns every source
  condition, repeats all physical extractions, and reruns all three recipients.

Missing artifacts, rows, hashes, packages, raw evidence, or recomputation are
hard failures. Stored scientific pass booleans are never inputs to the verdict.

## Claim ceiling

A pass establishes only:

> `BOUNDED_FOREIGN_NEURAL_STATE_CAPABILITY_RECOVERY`

It would show that information installed by ordinary gradient learning in a
conventional Qwen parameter delta can be decoded by a previously frozen generic
frontend into a source-independent R11 package and realized exactly by three
heterogeneous frozen recipients with zero recipient training.

It is not teacher-behavior cloning, extraction of knowledge already present in
the pretrained Qwen checkpoint, English or domain extraction, a minimality
result, arbitrary-model support, or superiority to LoRA or distillation. Those
moonshot claims remain open even if R15A passes.
