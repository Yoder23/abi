# ABI technical claims

Status: the original 18/18 certificate is superseded by a repaired validation
candidate. Human and independent-hardware review remain closed until the
repaired public tag survives clean public reconstruction and a fresh blind
red-team.

## Proven technical claims

- The sealed English, Python, chemistry, and civics packages execute through
  the generic canonical ABI runtime in the three tested LayerCake v25,
  Qwen2.5-0.5B, and Pythia-160M codec/conformance environments.
- One zero-parameter frozen adapter per tested environment accepts all four
  packages without capability-specific fitting or calibration. New
  certification runs occur in private mount namespaces containing only the
  generic corpus, canonical ABI/specification, adapter code, and the selected
  host; capability archives and source-success ledgers are physically absent.
- All package and adapter bytes remain unchanged across the 12 tested cells.
- The locked suite records 5,043/5,043 receiver source successes and exact
  source-output bytes, plus 1,681/1,681 cross-environment canonical output
  identities and 300/300 specialist action-sequence identities.
- Fresh post-repair isolation executes 2,100 raw rows: 700 per host, with zero
  target successes and 700/700 cross-host output identities.
- All 12 capability removals fail closed and identical reinstall restores exact
  output; all 24 equal-size random/shuffled corrupt packages are rejected.
- Teacher/source execution, receiver training, and receiver calibration are
  absent from the locked matrix events.
- Recomputed idle adapter overhead is within the registered 10% bound on 20
  paired observations for each tested environment.
- The repaired host-causality audit performs 3,072 new live executions across
  real-host, neutral-host, zero-state, random-state, shuffled-state,
  host-removed, adapter-removed, and capability-removed conditions. The six
  host-state conditions preserve all 128 selected outputs per host; all 128
  adapter removals and 128 capability removals per host fail during live
  execution. This is a bounded **standalone capability-runtime** result;
  Qwen/Pythia state is noncausal under the tested runtime.

The strict verifier is `abi_v2.strict_validation`; repaired evidence is under
`results/abi_final_validation_v2/`. It rejects missing packages, rows, hashes,
mount evidence, adapters, and stale execution code without consuming experiment
status/gate booleans.

## Explicitly not proven

- Not proven to work with every LLM architecture.
- Not base-weight tensor transplantation or hidden-state injection.
- Not proof of a universal semantic coordinate system inside arbitrary models.
- Not evidence that Qwen or Pythia base-model computation generates the answers.
- Not independently reproduced on different hardware yet.
- Not independently human-rated yet; 0/21,000 ratings are complete.
- Not a certified global minimum-information representation. Its status is
  `PENDING_AFTER_EXTERNAL_VALIDATION`.
- Not universal superiority over LoRA, distillation, or fine-tuning.
- Not unseen-task generalization: the exact-retention reference suite is public.

## Component ownership

```text
prompt
  ↓
tested native codec/conformance environment
  ↓
generic frozen adapter (integrity + native units; no learned semantics)
  ↓
canonical ABI (typed context, intent, lifecycle, strict UTF-8)
  ↓
immutable standalone capability (learned semantics and generation/routing)
  ↓
canonical capability output
  ↓
same frozen codec adapter
  ↓
native token-unit realization
```

The capability architecture remains frozen at commit
`acfed2a225a32d36c32b625e35c6ede536cfab01`, tag
`abi-host-independence-technical-proof-2026-08-24`.
The tag `abi-final-technical-validation-ready-2026-08-25` is historical and
must not be promoted; it predates the physical-isolation and live-causality
repairs.
