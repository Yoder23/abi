# R44 — Frozen shared-core prospective replication

Status: `PREREGISTERED` before fixture materialization or execution.

## Frozen candidate

- R43 result SHA-256:
  `72967c1ae40f6e65f49b0a7be74bb196a8b07b972ee180fe0a3de824a2325bed`.
- Candidate package SHA-256:
  `6ccb115de59e71b1c70efcfe2b5a1673b88d8f1478fb0059d8577ef5e2b6e6a4`.
- Zero-state package SHA-256:
  `7b674ea1d818338f962a8fb6df0bfb6b50377764008313cf5bdbdf7ec8bd6391`.
- LayerCake host commit:
  `f8a4569508f9f4a20e4babe3b4a942ac887da147`.
- No package, tensor, tokenizer, adapter, model, training row, seed, or scorer
  may change.

## Prospective fixture

After this protocol and its generator are committed, materialize 144 new rows:
12 per frozen task, base index 1,600,000, using the frozen R31 instruction and
detail generators. The first nine rows per task cover all three instruction
paraphrases crossed with all three detail cycles. No generated prompt or result
may be used for training or package selection.

ABI applies only the frozen R43 role-tagging adapter. Route with the frozen
task inference function. Execute the exact signed R43 package through fresh
CPU and CUDA `DirectCakeHost` registries. Execute the exact signed zero-state
package on the first row of each task on both devices. No teacher/source model,
teacher row, logits, activations, or ABI-local neural loader may be present.

## Gates

1. Candidate functional score at least 132/144 per device, at least 10/12 in
   every task, exactly 12/12 abstention, 144/144 route and contract checks, no
   representation errors, and all 144 outputs non-empty/non-collapsed.
2. CPU and CUDA candidate outputs are byte-identical 144/144.
3. All 24 zero-state task probes fail functional quality and none matches its
   candidate output.
4. Strict signature/package/tensor identity passes before execution.
5. Removal rejects generation; reinstall restores the first output exactly.
6. A one-byte-corrupted package is rejected by strict installation.
7. Active parameters remain exactly 296,554 and receiver training is zero.
8. Raw fixture, observations, lifecycle/corruption records, environment and
   artifact hashes are written once. A separate verifier must recompute raw
   metrics and replay the candidate matrix before certification.

Missing data, packages, hashes, devices, rows, or controls fail closed. A pass
certifies prospective replication only for the bounded twelve-task
supplied-content interface. It is not unrestricted English, arbitrary teacher
or domain transfer, global minimality, or LoRA/distillation superiority. The
full ABI moonshot remains open.

