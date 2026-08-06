# ABI Phase 3 V38 Failure Attribution V40

Status: **COMPLETE — PRIMARY FIT/STATE FAILURE, SECONDARY HEADER SHIFT**

V40 replayed the sealed V38 checkpoint without training or mutation.

- Autonomous replay on 280 acquisition records was 240/280 exact (85.71%),
  below the preregistered 90% threshold. Primary ownership is therefore model
  fit or autonomous causal state.
- On 280 held-out prompts, the original shared validation header produced
  0/280 functional passes and 279/280 fact-free-mode outputs.
- Removing the shared first line produced 46/280 passes (16.43%).
- Replacing it with a capability-matched acquisition header produced 115/280
  passes (41.07%).

The next materially distinct proposal must address both insufficient
autonomous fit/state and header-sensitive capability routing. Header rewriting
alone is not sufficient under the controlling priority rule. Data expansion,
extra steps alone, tokenizer variants, and nearby BPE sweeps are not supported.

Result SHA-256: `850c03b413fa4cf1e188653542fe1966e8ac942d67c79e51a3c0fef71065eef6`

Evidence SHA-256: `eb813c63940355314cf95deb0305007da36434da7c9ba433ed71c268cd523bdf`

Raw rows SHA-256: `ed84ea772dee059da98d01ef6958ce06bb9685465e29651e080a4dc2da4d8697`

This is read-only negative-evidence attribution. It does not certify Phase 3,
identify a LayerCake regression, or support superiority over LoRA or
distillation.
