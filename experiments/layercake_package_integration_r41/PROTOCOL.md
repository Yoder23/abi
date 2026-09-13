# R41 — Real LayerCake Package-Boundary Integration

Status: `PREREGISTERED` before implementation and execution.

## Frozen inputs

- ABI base commit: `48afee5`.
- LayerCake host commit: `f8a4569`.
- R36 result SHA-256:
  `1830a0a1be9b520376ce228fd50514e79f9b3a8224d56f7a1b36e0338d03d5bd`.
- R39 result SHA-256:
  `eee143b1f35038ed518cbd4a9a8aa034da03e0799fea39a9498e76b7bed242f0`.
- R40 result SHA-256:
  `50c6a5aabb8e416ec7e959919591246eca1193b36c0c6b707ad0c217f7f324d6`.
- R40 fixture and its 144 record IDs, prompts, task labels, and scores are
  immutable. No row may be removed or edited.
- All twelve state/tokenizer hashes are read from the frozen R39 result and
  must be rechecked before packaging.

## Intervention

Build twelve signed, non-executable `.cake` archives through LayerCake's
public `build_package` interface. Each archive contains exactly one frozen
R36/R39 task state and its declarative field-addressed tokenizer. Do not train,
fine-tune, quantize, prune, mutate, transpose, rename, or numerically convert a
tensor. The signing key is a deterministic, explicitly non-production
research-fixture key so an exact rerun can reproduce the package bytes.

Install the packages into fresh CPU and CUDA `DirectCakeHost` registries through
the public LayerCake installer. Route only by the already frozen R40 task
inference function. Generate live outputs through the selected installed cake.
The ABI-local model loader is not an accepted execution path.

## Required gates

1. All 12 source state/tokenizer hashes match R39 and all 12 packages load with
   valid Ed25519 signatures.
2. Every loaded tensor is exactly equal to its frozen source tensor; no source
   tensor changes before/after packaging.
3. Fresh CPU host: 144/144 route, contract, and functional checks pass, with
   zero representation failures.
4. Fresh CUDA host: the same 144/144 checks pass and every output byte string is
   identical to CPU and frozen R40.
5. Install/verify succeeds for every package. Remove rejects generation for
   every package. Reinstall restores the exact output on its task's first row.
6. A one-byte-corrupted copy of every archive is rejected by strict install.
7. With all packages installed in a fresh sparse-control host, one request to
   one explicitly selected package loads exactly one module and executes no
   other package.
8. No teacher model, teacher weights, teacher logits, teacher activations,
   source corpus, training rows, or ABI-local generation loader is loaded.
9. Runtime parameters are frozen and receiver training/update calls are zero.
10. Raw per-row observations, package inventory, host lifecycle records,
    component hashes, environment identity, and timings are written once and
    bound into the result by SHA-256.

Any missing prerequisite, file, hash, raw row, device, package, signature,
control, or recomputation is a failure. Stored status booleans are not trusted.

## Promotion rule and claim ceiling

All gates must pass in one immutable execution. A pass proves only that these
twelve bounded teacher-derived neural components cross the real canonical
LayerCake package/install/execute boundary unchanged and reproduce the frozen
144-row prospective supplied-content behavior. It does not prove unrestricted
English, arbitrary teacher/domain extraction, a global information minimum,
teacher-quality parity, or superiority to LoRA/distillation.

