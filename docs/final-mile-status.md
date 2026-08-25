# ABI final-mile status

Status date: 2026-08-25

## Current outcome

```text
ABI REPAIRED VALIDATION: LOCAL STRICT PASS
PUBLIC RECONSTRUCTION AND BLIND RED-TEAM: PENDING
```

The earlier 18/18 readiness declaration and tag
`abi-final-technical-validation-ready-2026-08-25` are superseded. They remain
available as historical evidence, but cannot support a current readiness claim.
Adversarial review identified three material weaknesses: certification excluded
data by name/path instead of physical absence, the causal audit replayed stored
evidence through a neutral stub, and final verification trusted derived status
fields.

The first repaired candidate tag,
`abi-final-validation-v2-repaired-2026-08-25`, is also superseded. Its default
Windows checkout surfaced one CRLF/LF normalization mismatch in an unrelated
historical JSON receipt. The content was normalized without changing its JSON
meaning; `abi-final-validation-v2-repaired-r1-2026-08-25` restarts clean public
reconstruction from a new immutable lineage.

## Repaired local evidence

The repaired certification worker runs inside a private WSL2 mount namespace.
Its exact capsule contains only the generic certification corpus, canonical ABI
implementation/specification, adapter code, and selected host code or snapshot.
The development drive is replaced by a private `tmpfs`; raw `/proc/self/mountinfo`
and a complete capsule byte inventory are preserved. Capability archives and
source-success ledgers are physically absent.

The live causal campaign executes every selected task anew under eight
conditions: real host, neutral host, zero state, random state, shuffled state,
host removed, adapter removed, and capability removed. It does not read the
prior matrix outputs or source-answer references. Adapter removal fails
realization and capability removal fails generation in the live path.

The strict verifier derives its verdict from raw rows, source bytes, immutable
package/adapter hashes, capsule inventories, raw mount tables, and repeated
timings. It does not consume experiment gate or status booleans. Its certificate
binds all 59 required inputs with per-file SHA-256 and an aggregate digest.

| Recomputed item | Local repaired result |
| --- | ---: |
| Physically isolated hosts | 3/3 |
| Certification roundtrip rows | 384/384 |
| Native host forward rows | 32/32 finite |
| Capability archives visible during certification | 0 |
| Source-success ledgers visible during certification | 0 |
| Locked matrix rows | 5,043/5,043 |
| Cross-host output identities | 1,681/1,681 |
| Specialist action identities | 300/300 |
| New live causal rows | 3,072/3,072 |
| New live isolation rows | 2,100/2,100 |
| Isolation target successes | 0/2,100 |
| Supported repository tests | 66/66 |
| Disposable archive hostile mutations | 9/9 rejected; exact restore |
| Trusted scientific booleans consumed | 0 |

Recomputed median idle-adapter overhead from 20 paired observations is 0.89%
for LayerCake, 0.05% for Qwen2, and 7.45% for Pythia, all below the registered
10% ceiling. These are bounded conformance overhead measurements, not general
inference-performance comparisons.

## Exact claim boundary

The evidence supports a bounded standalone capability-runtime result across
LayerCake v25, Qwen2.5-0.5B, and Pythia-160M codec/conformance environments.
The same immutable English, Python, chemistry, and civics packages execute
through one unchanged zero-parameter adapter per environment. Qwen/Pythia host
state is measured but cannot enter the frozen semantic realization API; their
checkpoints and tokenizers participate in conformance/native-unit handling, not
answer generation.

It does not prove base-weight tensor transplantation, host-model generation,
compatibility with arbitrary LLMs, unseen-task generalization, a global minimum
information representation, human-rated quality, independent reproduction, or
universal superiority over LoRA, distillation, or fine-tuning.

## Remaining mandatory order

1. Publish all four immutable capability packages and the definitive archive at
   durable hash-addressed public URLs.
2. Clone the repaired tag into a brand-new directory with no development assets.
3. Download and verify only the published manifests/assets, then reproduce the
   release from them.
4. Run the strict hostile mutation suite in the disposable extracted release.
5. Run a fresh blind Codex red-team against the repaired tag.
6. Only after those pass may human ratings and different-hardware independent
   execution begin.

## Evidence map

- Strict certificate: `results/abi_final_validation_v2/strict_validation.json`
- Physical certification: `results/abi_final_validation_v2/isolated_certification_strict/`
- Live causality: `results/abi_final_validation_v2/live_causality/`
- Live isolation: `results/abi_final_validation_v2/live_isolation/`
- Pre-public strict hostile receipt:
  `results/abi_final_validation_v2/strict_hostile_pre_public.json`
- Historical pre-repair certificate: `results/abi_final_validation/`
- Historical first live run with declarative booleans:
  `results/abi_final_validation_v2/live_causality_untrusted_boolean_history/`
- Historical strict summary before complete input binding:
  `results/abi_final_validation_v2/strict_validation_pre_boolean_cleanup.json`

Human ratings remain 0/21,000. Different-hardware reproduction and the
registered minimum-information frontier remain open.
