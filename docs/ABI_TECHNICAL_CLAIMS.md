# ABI technical claims

Status: final internal technical validation passed (18/18); ready for human and
independent-hardware review. Those external reviews remain incomplete.

## Proven technical claims

- The sealed English, Python, chemistry, and civics packages execute through
  the generic canonical ABI runtime in the three tested LayerCake v25,
  Qwen2.5-0.5B, and Pythia-160M codec/conformance environments.
- One capability-blind, zero-parameter frozen adapter per tested environment
  accepts all four packages without capability-specific fitting or calibration.
- All package and adapter bytes remain unchanged across the 12 tested cells.
- The locked suite records 5,043/5,043 receiver source successes and exact
  source-output bytes, plus 1,681/1,681 cross-environment canonical output
  identities and 300/300 specialist action-sequence identities.
- English-only specialist leakage is 0/900; wrong-capability success is 0/1,200.
- All 12 capability removals fail closed and identical reinstall restores exact
  output; all 24 equal-size random/shuffled corrupt packages are rejected.
- Teacher/source execution, receiver training, and receiver calibration are
  absent from the locked matrix events.
- Recomputed idle adapter overhead is within the registered 10% bound on 20
  paired observations for each tested environment.
- The host-causality audit proves the packages plus generic runtime are a
  **standalone capability-runtime**. A neutral UTF-8 stub reproduces all 5,043
  promoted outputs. Qwen/Pythia hidden states are not materially causal to the
  capability answers; their checkpoints provide frozen conformance probes and
  their tokenizers provide native unit representations.

The raw-evidence recomputation is
[`results/abi_final_validation/headline_recomputation.json`](../results/abi_final_validation/headline_recomputation.json).
The causal result is
[`results/abi_final_validation/host_causality.json`](../results/abi_final_validation/host_causality.json).

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

The technical architecture is frozen at commit
`acfed2a225a32d36c32b625e35c6ede536cfab01`, tag
`abi-host-independence-technical-proof-2026-08-24`.
