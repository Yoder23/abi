# ABI final-mile status

Status date: 2026-08-25

## Current outcome

```text
ABI REPAIRED VALIDATION: TECHNICAL PROOF FROZEN
R3 PUBLIC RECONSTRUCTION: PASS; R3 BLIND RED-TEAM: FAIL
R4 CONTENT-BOUND REPAIR: LOCAL TECHNICAL PASS; PUBLICATION PENDING
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
meaning; `abi-final-validation-v2-repaired-r1-2026-08-25` restarted clean public
reconstruction from a new immutable lineage.

The r1 reconstruction then correctly failed 65/66 tests: clean-tree validation
still compared the repaired certification evaluator to the superseded frozen
candidate receipt. A new additive repaired receipt at
`results/abi_final_validation_v2/frozen_release_candidate.json` closed that
branch split. The r1 tag/release is superseded. A blind review of r2 then found
that isolation was incomplete, host states were recorded rather than applied,
and several claims were not fully recomputable. Those failures are preserved.

The physical/live r3 proof is frozen at commit
`7064c94f2c6137a29b8793d9b0ec95137efb051e`, annotated tag
`abi-final-validation-v2-physical-live-proof-2026-08-25`.

R3 was then published and reconstructed solely from its public manifest. Its
fresh blind audit still returned `FAIL` because the admitted runtime trees were
not content-bound and the restricted reviewer could not query the exact release
endpoint. R4 is an additive repair; r3 remains immutable historical evidence.

## Repaired local evidence

The r4 certification worker runs inside a private WSL2 mount namespace.
Its exact capsule contains only the generic certification corpus, canonical ABI
implementation/specification, adapter code, and selected host code or snapshot.
The development drive is replaced by a private `tmpfs`; raw `/proc/self/mountinfo`
and a complete capsule byte inventory are preserved. The broad `/etc` bind is
replaced by a minimal read-only runtime-configuration tmpfs. Every reachable
non-virtual regular file is fully hashed and content-scanned, readable ZIP
members are expanded, and every symlink is recorded. Capability archives are
detected by internal cake/ABI archive structure even if renamed. Capability
archives and source-success identifiers are physically absent.

The live causal campaign executes every selected task anew under eight
conditions: real host, neutral host, zero state, deterministic-random state,
deterministic-shuffled state, host removed, adapter removed, and capability
removed. Qwen/Pythia state conditions mutate a native parameter and run a new
forward; the resulting state is consumed by the adapter. Host removal receives
no checkpoint or native objects. The campaign does not read prior matrix
outputs or source-answer references. Adapter removal fails realization and
capability removal fails generation in the live path.

The strict verifier derives its verdict from raw rows, source bytes, immutable
package/adapter hashes, capsule inventories, raw mount tables, and repeated
timings. It does not consume experiment gate or status booleans. Its certificate
binds all 98 required inputs with per-file SHA-256 and aggregate digest
`24b29e4a6f48ae48b4dbb3b7185223a36b174d547fd7b0e20dc7cc874d67202e`.

| Recomputed item | Local repaired result |
| --- | ---: |
| Physically isolated hosts | 3/3 |
| Certification roundtrip rows | 384/384 |
| Native host forward rows | 32/32 finite |
| Reachable filesystem inventory rows | 301,543/301,543 |
| Reachable regular-file bytes content-scanned | 11,681,818,650 |
| Capability archives visible during certification | 0 |
| Campaign/success identifiers visible during certification | 0 |
| Locked matrix rows | 5,043/5,043 |
| Cross-host output identities | 1,681/1,681 |
| Specialist action identities | 300/300 |
| New live causal rows | 3,072/3,072 |
| New live isolation rows | 2,100/2,100 |
| Isolation target successes | 0/2,100 |
| Supported r4 release tests | 75/75 |
| Disposable archive hostile mutations | 17/17 rejected; exact restore |
| Trusted scientific booleans consumed | 0 |

Recomputed median idle-adapter overhead from 20 paired observations is 6.87%
for LayerCake, 0.0039% for Qwen2, and 2.45% for Pythia, all below the registered
10% ceiling. These are bounded conformance overhead measurements, not general
inference-performance comparisons.

## Exact claim boundary

The evidence supports a bounded standalone capability-runtime result across
LayerCake v25, Qwen2.5-0.5B, and Pythia-160M codec/conformance environments.
The same immutable English, Python, chemistry, and civics packages execute
through one unchanged zero-parameter adapter per environment. Qwen/Pythia host
state is freshly computed under physical parameter interventions and consumed
by the conformance adapter, but does not change canonical capability output in
the tested runtime. Their checkpoints and tokenizers participate in
conformance/native-unit handling, not answer generation.

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

- Strict certificate:
  `results/abi_final_validation_v2/strict_validation_r4_content_bound.json`
- Physical certification:
  `results/abi_final_validation_v2/isolated_certification_strict_r4_content_bound/`
- Live causality: `results/abi_final_validation_v2/live_causality/`
- Live isolation: `results/abi_final_validation_v2/live_isolation/`
- Pre-public strict hostile receipt:
  `results/abi_final_validation_v2/strict_hostile_pre_public_r4.json`
- Frozen repaired candidate:
  `results/abi_final_validation_v2/frozen_release_candidate_r4.json`
- Preserved r3 blind failure:
  `results/abi_final_validation_v2/blind_redteam_r3_fail.md`
- Historical pre-repair certificate: `results/abi_final_validation/`
- Historical first live run with declarative booleans:
  `results/abi_final_validation_v2/live_causality_untrusted_boolean_history/`
- Historical strict summary before complete input binding:
  `results/abi_final_validation_v2/strict_validation_pre_boolean_cleanup.json`
- Historical r2 live-state and isolation failures:
  `results/abi_final_validation_v2/live_causality_r2_logged_state_history/` and
  `results/abi_final_validation_v2/isolated_certification_strict_r2_stale_source_history/`

Human ratings remain 0/21,000. Different-hardware reproduction and the
registered minimum-information frontier remain open.
