# ABI

ABI is a research codebase for acquiring, labeling, packaging, validating, and
eventually minimizing capabilities extracted from open-weight teacher models.
It is designed to produce immutable capability artifacts that a separate
[LayerCake](https://github.com/Yoder23/layercake) runtime can install, compose,
route, and execute.

The repositories have deliberately separate responsibilities:

- **ABI:** teacher qualification, probing, labeling, segregation, provenance,
  information accounting, artifact construction, and verification.
- **LayerCake:** hosting, installation, transfer between compatible hosts,
  routing, orchestration, and inference performance.

## Current release status

An independent audit dated 2026-09-14 returned
`FULL_MOONSHOT_NOT_PROVEN`. That is the controlling whole-moonshot verdict.
The untouched audit, its machine-readable summary, and its four receipts are
preserved under
[`reviews/independent_2026-09-14/`](reviews/independent_2026-09-14/).
R97 is the strongest bounded local transfer result; R7 remains the strongest
publicly reconstructed release. Neither may be combined with V1089 to claim
one coherent final-artifact proof.

The latest technical validation is R7:

<https://github.com/Yoder23/abi/releases/tag/abi-final-validation-v2-repaired-r7-2026-08-30>

R7 passes a bounded capability-runtime/conformance proof across LayerCake v25,
Qwen2.5-0.5B, and Pythia-160M using four immutable packages: English, Python,
civics, and chemistry.

This is not a claim that ABI already extracts a fluent minimal English core
from any teacher. Human ratings, different-hardware reproduction, registered
minimum-information certification, teacher-extraction quality, and comparison
with LoRA/distillation remain open.

## Later local compiler campaign

The controlling public release is still R7, but it is not the end of the local
research record. The separately sealed V1089 compiler campaign has completed
all currently registered machine work through Phase 7 and a clean same-machine
Phase 8 export rehearsal:

| Phase | Local evidence status |
| --- | --- |
| Phase 0 | Complete |
| Phase 1 | Complete |
| 2 | Machine evidence and blinded handoff complete; 0/21,000 independent-human preferences |
| 3 | All registered machine gates pass; unconditional certification remains dependent on Phase 2 |
| 4 | B40 is the smallest stable passing tested budget among B20/B40 for the exact five-route host |
| 5 | Three-domain selective reconstruction and behavioral exclusion certified |
| 6 | Three-package composition, portability, and provenance certified on three fresh host initializations |
| 7 | Same exact integrated product certified on the registered CPU and CUDA device |
| 8 | Clean same-machine export rehearsal certified; independent operator and different hardware still required |

The B40 English host passed the registered machine-quality contract in all
three 1,400-observation screens (4,200 observations total). The exact
seed-104729 core later used by the Phase 7 product scored 1,381/1,400 across fourteen
capabilities with zero collapse, and its paired advantage over the teacher was
10.29 percentage points (95% bootstrap lower bound 8.64 points). The integrated
Phase 7 product retained the same core and three packages while measuring
13.24x the optimized CPU-transformer throughput and 13.51x the registered GPU
comparator throughput. These are bounded same-machine results, not human,
arbitrary-hardware, global-minimum, deployed-multi-source, or universal
ABI-over-LoRA/distillation claims. The authoritative state is
[`evidence/current/ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V1089.json`](evidence/current/ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V1089.json).

The latest additive local result is R97. A frozen 530,050-parameter
teacher-derived reasoning bridge scored 1,399/1,400 on a separately sealed
prospective catalog, versus 1,088 for the live pinned teacher, 60 for the
unchanged LayerCake parent, and 17 for a randomized bridge. It retained
1,087/1,088 teacher-correct rows, executed the registered sparse path on every
row, showed zero collapse, passed strict raw recomputation and 12/12 hostile
mutations, and reproduced its raw evaluation byte-for-byte in a second live
run. See
[`experiments/prospective_length_span_r97/CERTIFICATE.md`](experiments/prospective_length_span_r97/CERTIFICATE.md).
R97 is a bounded two-hop reasoning transfer result; it is not evidence of a
complete fluent English substrate, arbitrary-domain extraction, or universal
superiority over LoRA/distillation.

The independent-audit response restores exact root-relative compatibility
copies of historical certificates and the V6 catalog so the frozen Phase 6-8
verifiers replay without relying on test order. It also puts the exact local
R97 parent and V1089 Phase 7 handoff payloads on the Git-LFS publication
surface. Phase 5 clean replay remains blocked: six baseline tensors totaling
1,191,079,704 bytes were deleted during the recorded 2026-08-29 storage
cleanup and were not recoverable from local LFS or other working copies. This
gap is explicit in the review manifest; no verifier is weakened around it.
See [`docs/INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md`](docs/INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md).

The additive R8 native-neural-transfer campaign is a Level 0 negative result.
Its strongest package-only recipient interface produced AFTER−BASE +0.001953
across 1,024 paired Pythia rows (95% bootstrap CI -0.019531 to +0.024414).
Canonical extraction was exact, but recipient behavior did not transfer. The
held-out secret was never revealed, and R8 does not change the R7 release.
The final public verifier rejected 7/7 hostile mutations and ignored one forged
scientific boolean without changing the `NO` verdict (8/8 expected outcomes).

R9 then tested the backend/compiler hypothesis directly with an intentionally
capability-specific Pythia neural backend. V1 failed to fit its own training
rows. The preregistered v2 repair exposed both embedding and final recipient
states and trained a 1,039,880-parameter backend for 10,000 steps. It still
recomputed at only 12.85% training accuracy and 12.5% unseen-depth AFTER
accuracy, versus 7.32% BASE; ZERO also reached 12.5%, failing causality. All
8,192 evaluation and 288 training observations replayed live, and 5/5 hostile
controls were rejected. R9 therefore closes this recipient-state GRU branch
without opening the universal-backend experiment or changing R7.

R10 then connected the synthetic R8 extraction to a clean package slot and
three frozen recipients. Its runtime-owned copy/paste component passed: four
2,040-byte AFTER packages produced 1.0 accuracy after paste and restore across
61,440 Pythia, Qwen2, and T5 rows; removal exactly matched BASE, all negative
controls were at most 0.15625, recipient parameters were unchanged, and a fresh
live run reproduced every row byte-for-byte. The overall R10 contract is still
negative because the source model's native decoder reached only 0.59375 to
0.65234375 against the registered 0.99 source gate. See
[`docs/R10_COPY_PASTE_RESULT.md`](docs/R10_COPY_PASTE_RESULT.md). This proves a
bounded canonical-runtime component, not LayerCake integration, lossless source
behavior, English/domain extraction, or native neural transplantation.

R11 now supplies the missing bounded neural-ABI construction proof. Four
held-out synthetic capabilities were learned as compact neural transition
states, packaged into four 2,053-byte content-addressed artifacts, and executed
through frozen DistilGPT2, Pythia, Qwen2, and T5 native output heads. AFTER and
RESTORED were exactly 1.0 on all 90,112 registered source-plus-recipient rows;
all teacher/recipient canonical UTF-8 outputs matched, negative controls were
at most 0.19140625, all removal interventions exactly restored BASE, recipient
optimizer steps were zero, and model/codec hashes did not change. A fresh live
replay of the exact revision-001 package bytes reproduced both raw JSONL files
byte-for-byte, and 7/7 hostile mutations failed closed. See
[`docs/R11_NATIVE_NEURAL_CONSTRUCTION_RESULT.md`](docs/R11_NATIVE_NEURAL_CONSTRUCTION_RESULT.md).

R11 is deliberately narrow. Its teacher substrate is ABI-native and synthetic;
it does not prove extraction from a conventional pretrained LLM, English or
natural-domain transfer, internalization into recipient transformer blocks, or
LoRA/distillation superiority. R7 remains the controlling public release.

R12-A has now exercised the missing frontend on one bounded public synthetic
capability using conventionally trained Qwen2.5-0.5B. Its best native source
checkpoint reached `510/512 = 0.99609375`, two rows short of the registered
exact source gate. Its 24 atomic probes were exact, and the fixed
zero-parameter extractor emitted an unchanged 2,053-byte R11 package whose
executor was exact on `512/512` unseen, prompt-disjoint compositions without
the teacher at execution time. This is a bounded public frontend construction
pass and a strict teacher-gate failure, not held-out R12 certification. See
[`docs/R12_FOREIGN_FRONTEND_PUBLIC_RESULT.md`](docs/R12_FOREIGN_FRONTEND_PUBLIC_RESULT.md).

R13-B now passes a separately preregistered held-out Track B control. Four
conventionally LoRA-trained Qwen sources each exposed 24/24 atomic transitions;
the fixed zero-parameter compiler emitted four 2,053-byte packages that scored
2,048/2,048 against the hidden capability oracle and 6,144/6,144 across frozen
Pythia, Qwen2, and T5 recipients. Seven fresh-process replays were byte-exact,
and the repaired verifier passed 9/9 hostile controls. Native Qwen matched only
372/2,048 long compositions, so this proves bounded enumerable capability
canonicalization—not behavioral transplantation. Bulk replay assets remain
unpublished. See
[`docs/R13_BOUNDED_CAPABILITY_EXTRACTION_RESULT.md`](docs/R13_BOUNDED_CAPABILITY_EXTRACTION_RESULT.md).

R14 then removed the exhaustive-table shortcut. Its fixed frontend received
only 256 mixed, non-atomic Qwen probability observations per capability and was
tested on 10,000 unseen-depth cases plus 1,000 order counterfactuals drawn from
a 4.64-billion-case space. It recovered one of three fresh latent programs
exactly; the other two packages reached only 0.2449/0.2520 and 0.2657/0.2340
on unseen/order-counterfactual oracle evaluation. Three fresh source replays
were byte-exact and 7/7 hostile controls passed. R14 is therefore a bounded
partial result and failed certification—not a non-exhaustive extraction pass.
See
[`docs/R14_NON_EXHAUSTIVE_CAPABILITY_RESULT.md`](docs/R14_NON_EXHAUSTIVE_CAPABILITY_RESULT.md).

R15A materially changed the frontend from behavior probing to controlled
foreign neural-state access. A generic decoder qualified on 320 public Qwen
learning events received only anonymous 7,168-element before/after effective
output-weight deltas. On eight fresh committed capabilities it recovered 8/8,
emitted eight 2,053-byte R11 packages, and scored 1.0 over 80,000 unseen plus
8,000 counterfactual cases. The packages produced 1.0 AFTER/RESTORED behavior
across unchanged Pythia, Qwen2, and T5 recipients; full-state controls were at
most 0.265. Strict verification passed, fresh live execution reproduced all
135,168 recipient rows byte-for-byte, and 9/9 hostile mutations were rejected.
See [`docs/R15_FOREIGN_NEURAL_STATE_RESULT.md`](docs/R15_FOREIGN_NEURAL_STATE_RESULT.md).

R15A is the first bounded foreign learned-state recovery pass, but it is not
the full ABI moonshot. The source capabilities were deliberately taught with
complete atomic supervision, and extraction used a before/after delta. The
result does not establish extraction or labeling of pretrained English/domain
knowledge, teacher-quality natural generation, after-only extraction,
LayerCake English-core ingestion, minimality, or LoRA/distillation superiority.

R15B removes the source-training requirement for one deliberately narrow
representation prerequisite. An unchanged Qwen2-7B-Instruct generated 24/24
registered atomic answers under four secret slot mappings. A frozen,
zero-parameter decoder operating in a Linux pivot-root/no-network capsule
recovered all four transitions from pre-answer residuals and frozen digit-head
rows. The resulting four 2,053-byte R11 packages scored 40,000/40,000 on the
registered deep oracle and reproduced 33,792 recipient rows across Pythia,
Qwen2, and T5; strict and live verification passed and 10/10 hostile mutations
were rejected. See
[`docs/R15B_PREEXISTING_REPRESENTATION_RESULT.md`](docs/R15B_PREEXISTING_REPRESENTATION_RESULT.md).

R15B is not English or open-world domain extraction. Its prompts explicitly
state the arithmetic operation, source reasoning exposes the answer before the
captured terminal digit, semantic labels are externally registered, and the
package's deep generalization comes from the registered affine inductive bias.
The full ABI moonshot therefore remains open, and R7 remains the controlling
published release.

R16 now passes a preregistered bounded local factual-acquisition and semantic-
segregation test. From an unchanged Qwen2-7B-Instruct source it recovered 16/16
held-out facts across chemistry and geography, emitted two immutable packages,
and matched 48/48 disjoint evaluation questions with the teacher absent.
Target-only execution was 48/48, other-domain and removed-package conditions
abstained 48/48, a rotated-score control scored 0/48, live evidence replayed
byte-exactly, and the expanded verifier rejected 21/21 hostile mutations. A
blind review passed with no Critical or High finding. See
[`docs/R16_FACTUAL_SEMANTIC_RESULT.md`](docs/R16_FACTUAL_SEMANTIC_RESULT.md).

This is a structured closed-candidate memory result with a registered ontology
and candidate vocabulary. It is not autonomous open-world discovery, fluent
English extraction, teacher-quality free generation, native neural
transplantation, production LayerCake ingestion, minimality, or superiority to
LoRA/distillation. A second preregistered 16-fact selection, bound to the
repaired public protocol before reveal, independently repeated every gate and
expanded combined held-out coverage to 27 distinct facts. Blind review of the
exact replicated seal passed with zero findings; durable publication and clean
public reconstruction remain open.
The full ABI moonshot therefore remains open, and R7 remains the controlling
published release.

R17 now tests the next, explicitly separate step: teacher-derived
compositional English surface realization rather than factual lookup. Its
first frozen public source interface failed at 38/72 extraction and 28/48
evaluation outputs. Its sole bounded repair improved this to 67/72 and 47/48
but still missed the locked 120/120 source prerequisite, before ABI compilation
or LayerCake was invoked. The exact-template branch is closed; its successor
must separate functional language quality from raw string identity. See
[`docs/R17_LINGUISTIC_REALIZATION_RESULT.md`](docs/R17_LINGUISTIC_REALIZATION_RESULT.md).

R18 replaces that exact-string boundary with a frozen feature-factorized
grammar package. Reusing the same R17-v2 teacher outputs, its 3,881-byte
package scored 48/48 on lexically disjoint functional realizations, versus
47/48 for the teacher and 46/48 for independent per-signature consensus. It
had zero regressions on teacher-correct rows, a 0/48 mood-permutation control,
and 48/48 removal abstentions. Strict recomputation, 12/12 hostile mutations,
and a fresh byte-exact physical replay passed. The preregistered hidden lexical
replication then failed at 46/48, with two source-correct plural questions
regressed by unsafe cross-number auxiliary pooling. This remains a bounded
public prerequisite, not a certified realization mechanism, unrestricted
English, or the ABI moonshot. See
[`docs/R18_FACTORIZED_REALIZATION_RESULT.md`](docs/R18_FACTORIZED_REALIZATION_RESULT.md).

R19 is a materially different successor: it learns same-number structural
contrasts between positive and negative forms instead of pooling across number.
On the disclosed R17 and failed-R18 evidence sets it scored 96/96 total
functional rows with zero source-correct regressions, 0/96 controls, and 96/96
removal abstentions. Its separately committed 20-lexeme hidden replication then
scored 48/48, versus 43/48 for the unchanged teacher and 48/48 for independent
modal consensus, with zero teacher-correct regressions, 0/48 control behavior,
and 48/48 removal abstention. The 4,263-byte package contains no source or host
parameters; 22/22 hostile mutations were rejected and a fresh physical run
reproduced the package and evaluation byte-for-byte. R19 therefore passes only
the registered bounded surface-realization claim. It is not unrestricted
English, prompt understanding, conversation, autonomous labeling, arbitrary-
domain extraction, LayerCake ingestion, minimality, or LoRA/distillation
superiority. See
[`docs/R19_CONTRASTIVE_REALIZATION_RESULT.md`](docs/R19_CONTRASTIVE_REALIZATION_RESULT.md).

R20 then tested whether that structural approach could scale to raw natural
instructions for prose, summary, email, bullets, clarification, and abstention.
The frozen unchanged teacher produced all 192 rows, but reached only 10/72
registered-exact extraction outputs and 19/120 evaluation outputs. Output
program support was insufficient for six classes, so compilation, packaging,
and LayerCake were not invoked. The evidence shows that exact delexicalized
templates are not a viable bridge to broader free-form English; that branch is
closed rather than prompt-tuned. See
[`docs/R20_INSTRUCTIONAL_REALIZATION_RESULT.md`](docs/R20_INSTRUCTIONAL_REALIZATION_RESULT.md).

R21 then replaced the failed template compiler with signed generative
LayerCake packages and matched sequence-distillation controls. Its factorized
system passed the public suite at 116/120 on every seed, versus 96/120 for the
teacher, and passed a fresh 1,080-row GPU replay, 54-row CPU replay, and all 24
package lifecycle/corruption controls. Its separately committed hidden split
failed: every factor seed reached 113/120 and exceeded the 107/120 teacher, but
summary was 13/20 against the frozen 15/20 minimum. Teacher-side semantic
labeling also failed; the successful registered-ontology classifier is not
autonomous discovery. See
[`docs/R21_LABEL_SEPARATED_GENERATIVE_TRANSFER_RESULT.md`](docs/R21_LABEL_SEPARATED_GENERATIVE_TRANSFER_RESULT.md).

R22 then tested one explicit semantic-plan normalization call for each of the
600 preserved R21 training responses. It failed before student training:
520/600 targets passed functional scoring and 255/600 preserved every field
verbatim once, although all 600 remained grounded and non-collapsed. LayerCake
was not invoked. This closes unconstrained teacher rewriting as the immediate
repair path and preserves the ABI-versus-LayerCake failure boundary. See
[`docs/R22_SEMANTIC_PLAN_NORMALIZATION_RESULT.md`](docs/R22_SEMANTIC_PLAN_NORMALIZATION_RESULT.md).

R23 then evaluated the unchanged R21 package bytes on a new committed hidden
split with a public-corpus-supported semantic contract. All three factorized
seeds scored 120/120 versus 96/120 teacher, 99-110 labeled monolith, and 27-32
raw sequence. A fresh replay reproduced 1,080 GPU and 54 CPU outputs, passed all
24 package removal/restoration controls, and rejected 24/24 targeted tensor
corruptions. This certifies a bounded six-task supplied-content mechanism, not
general English or the full ABI moonshot. See
[`docs/R23_SEMANTIC_REPLICATION_RESULT.md`](docs/R23_SEMANTIC_REPLICATION_RESULT.md).

## R7 at a glance

| Evidence | Result |
|---|---:|
| Public archive | 844,018,841 bytes |
| Archive SHA-256 | `fc50f423986149b5d4670ec9e28698540f64be96034efa26e5704c4469921e88` |
| Archive members | 1,193, all exact |
| Physical certification environments | 3/3 |
| Reachable inventory rows | 301,543 |
| Locked matrix | 5,043/5,043 rows |
| Live causality | 3,072 rows, 24 distinct processes |
| Live isolation | 2,100 rows, zero target successes |
| Fail-closed controls | 19/19 pre-public and 19/19 blind replay |
| Public reconstruction tests | 17/17 |
| Blind tar-prefix controls | 12/12 |
| Blind verdict | PASS, bounded scope |

The blind reconstruction succeeded with a 368-character extracted path while
Windows long paths were disabled.

## Install

ABI targets Python 3.10. For the lightweight package and CLI:

```bash
python -m pip install -e .
abi status
abi self-check
```

Teacher extraction and research workflows require the dependencies in
`requirements.txt`. Human-rating helpers use the optional human extra:

```bash
python -m pip install -e ".[human]"
```

## Public R7 verification

Download `public_release_assets_r7.json` from the R7 GitHub Release. It binds
the definitive archive and four capability packages by byte length, SHA-256,
content address, release tag, and commit.

From a clean checkout of the public tag:

```bash
python -m abi_v2.public_reconstruction \
  --manifest /path/to/public_release_assets_r7.json \
  --tag-clone /path/to/clean/tag-clone \
  --workspace /path/to/new/workspace \
  --output /path/to/public-reconstruction-receipt.json
```

The external different-hardware workflow is under `external_reproduction/`.
Its command sequence is:

```text
abi-reproduce verify
abi-reproduce certify-hosts
abi-reproduce capability-matrix
abi-reproduce causality
abi-reproduce isolation
abi-reproduce performance
abi-reproduce hostile-audit
abi-reproduce report
```

Executing this sequence on the development laptop is a rehearsal, not an
independent reproduction.

## Human review

The frozen packet contains 7,000 judgments for each of three independent
raters. The gate remains `0/21,000` until real raters complete and attest their
forms:

```bash
abi human-rate --rater R1
abi human-rate --rater R2
abi human-rate --rater R3
```

See `docs/PHASE2_HUMAN_RATING_HANDOFF_V1.md` for the exact handoff.

## Published artifacts

| Capability | Bytes | SHA-256 |
|---|---:|---|
| English | 253,216,208 | `acb787b3ffa0153c57d88cd37ba81c3f00b370d4ca4937e659cd4c775851f25d` |
| Python | 448,404 | `f1defaef2771ced336a332572a2d2f0e1e542399c877d182c48a6cd2e199231d` |
| Civics | 495,919 | `634ce66958859ec36dc1fbdf5ef34d6d2a9949d10cf2348a68c245d8c325d604` |
| Chemistry | 510,981 | `f9c9b2668fda5ef6b92844c1b7097fbdf8ff0daaae51f5b86f72d4a49000abeb` |

## Claim boundary

R7 supports only the bounded capability-runtime/conformance result described
above. It does not establish:

- arbitrary or universal model compatibility;
- tensor transplantation between unrelated architectures;
- complete diagnosis or extraction of teacher knowledge;
- fluent teacher-quality English generation;
- superiority over LoRA, distillation, or fine-tuning;
- human-rated quality;
- independent hardware reproducibility;
- global information minimality; or
- completion of the full ABI moonshot.

See `docs/ABI_TECHNICAL_CLAIMS.md`, `docs/ABI_FINAL_RESULTS.md`, and the
`review_packet/` directory before citing results.

## Repository map

- `abi/` — acquisition, labeling, artifact, and public CLI code.
- `abi_v2/` — canonical runtime/conformance, certification, strict verification,
  public reconstruction, and external reproduction tooling.
- `tests/` — supported automated checks.
- `results/abi_final_validation_v2/` — immutable R3-R7 validation lineage.
- `results/abi_moonshot/packages/` — published specialist packages.
- `experiments/native_isa_r11/` — bounded native-neural ABI construction code.
- `results/native_isa_r11/` — R11 preregistered raw evidence and live replay.
- `experiments/foreign_teacher_r12/` — conventional-teacher frontend protocol
  and fail-closed public verifier.
- `results/foreign_teacher_r12/` — additive R12-A receipts and exact packages;
  large source adapters remain excluded.
- `experiments/foreign_capability_r13/` — separated Track B held-out protocol,
  replay, sealing, and hostile-verification code.
- `results/foreign_capability_r13/` — local R13-B evidence and content-addressed
  artifact manifest; bulk replay assets are not yet public.
- `experiments/foreign_capability_r14/` — non-exhaustive capability protocol,
  source replay, diagnosis, and hostile-verification code.
- `results/foreign_capability_r14/` — compact R14 negative certificate and
  local artifact manifest; bulk adapters and raw replay rows are not public.
- `experiments/foreign_neural_state_r15/` — R15A preregistration, generic
  neural-state frontend, physical extraction, strict/live verification, and
  hostile audit.
- `results/foreign_neural_state_r15/` — compact R15A certificate and local
  immutable run lineage; bulk raw observations are not yet public.
- `experiments/preexisting_representation_r15b/` — R15B public qualification,
  preregistered held-out protocol, isolated representation decoder, and
  fail-closed verification.
- `results/preexisting_representation_r15b/` — compact R15B certificate and
  local evidence lineage; bulk recipient replay rows are intentionally
  excluded from Git.
- `experiments/factual_semantic_r16/` — R16 public qualification,
  preregistered held-out protocol, isolated structured-fact compiler, and
  fail-closed verifiers.
- `results/factual_semantic_r16/` — R16 raw held-out/live evidence, immutable
  chemistry/geography packages, accounting, and local certificates.
- `experiments/prospective_length_span_r97/` — frozen prospective reasoning
  protocol, strict verifier, hostile audit, and bounded certificate.
- `results/prospective_length_span_r97/` — source capture, bindings, raw rows,
  verification receipts, and byte-identical live replay.
- `external_reproduction/` — independent-operator workflow and environment
  lock.
- `review_packet/` — ordered technical and external-review handoff.
- `docs/` — architecture, claims, results, and review instructions.

Historical experimental evidence remains in Git for auditability. Generated
caches, model weights, temporary public reconstructions, and bulk reproducible
intermediates are intentionally excluded from the production tree.

## License and research status

This repository is a research release. Consult `LICENSE` and `NOTICE` before
redistribution. Scientific claims are governed by immutable evidence and the
claim ceiling above, not by roadmap language.
