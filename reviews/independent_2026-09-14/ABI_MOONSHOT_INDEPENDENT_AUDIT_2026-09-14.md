# ABI moonshot independent scientific audit

Audit date: 2026-09-14  
Auditor role: independent code, evidence, reproducibility, and falsification review  
Decision rule: every mandatory claim must pass in one coherent final-artifact lineage; a missing, conditional, bounded-only, unpublished, or external gate blocks the full claim.

## 1. Executive verdict

`FULL_MOONSHOT_NOT_PROVEN`

This is not a close call under the frozen decision rule. The repository contains substantial, unusually careful bounded evidence, but it does not contain one coherent evidence chain proving the complete capability in the mandate. The decisive failures are:

1. required human evidence is `0/21,000`;
2. independent-operator, different-CPU/CUDA Phase 8 execution has not occurred;
3. R7, V1089, and R97 are different artifacts and lineages and cannot be combined into one final proof;
4. R97's strongest scientific result is not publicly reconstructible because its unchanged LayerCake-parent checkpoint and associated payloads are ignored/local-only and absent from a durable release;
5. V1089's Phase 8 manifest contains three `tracked: false` payloads, two of which are absent from the public R7 release;
6. broad, teacher-competitive, human-validated open-ended English generation; automatic discovery; semantic labeling; latent core purity; arbitrary-teacher support; global minimality; and universal LoRA/distillation superiority are not demonstrated;
7. the current-master Phase 5-8 strict verifiers fail their frozen implementation/certificate bindings, and selected tests are order-dependent.

Novelty or “groundbreaking” status is not established. No serious prior-art review was part of this audit, and the technical moonshot itself did not pass.

## 2. Strongest exact claims currently supported

### Strongest local scientific claim

R97 supports this claim and no broader one:

> A frozen 530,050-parameter, task-specific neural bridge, prepared before the R97 catalog was evaluated and installed over one frozen compatible LayerCake parent, solved 1,399/1,400 prospective two-hop nonce-reasoning prompts across four registered families. The live source teacher solved 1,088/1,400, the unchanged parent 60/1,400, and a random bridge 17/1,400. It retained 1,087/1,088 teacher-correct rows, produced zero registered collapses, physically invoked the sparse path on 1,400/1,400 rows, ran without the teacher at LayerCake inference, and reproduced a byte-identical 1,400-row replay.

Evidence: `ABI_ROOT/results/prospective_length_span_r97/screen_v1/evaluation.jsonl`, 1,400 rows, 1,338,609 bytes, SHA-256 `b4a4999228af61a509a7d22f8f6906b3075ce649a52e531b959646fbbdb40b8c`; candidate checkpoint `ABI_ROOT/results/length_invariant_span_r96/candidate_v1/bridge.safetensors`, 2,123,440 bytes, SHA-256 `9d3d6b38b1d437b5cbbed50c1470b4df1aa0371ffa51b1abd2adefbb645806e4`; strict receipt `audit_outputs/r97-strict-independent.json`, SHA-256 `59438fbff7e07a2d81f78b9c9905d473b09678892026b33cbd30cc0a02dfe30c`. Command: `python -m experiments.prospective_length_span_r97.verify_evidence_v1 --root . --output <audit-output>`.

This does **not** prove broad English, open-ended generation, automatic discovery, latent semantic purity, arbitrary-teacher or arbitrary-host transfer, human quality, systems superiority, global information minimality, or the complete ABI moonshot. The bridge consumed 8,129 earlier teacher-derived records and 96,000 synthetic structural examples; the prospective prompts are disjoint, but the capability family and structural task were deliberately constructed.

### Strongest publicly reproducible claim

The exact R7 public release supports a bounded capability-runtime/conformance claim for four immutable packages and three named runtime environments. From a clean tag clone and only the published manifest/assets, strict raw recomputation passed with zero trusted scientific booleans, its stdout exactly matched the frozen certificate hash, and 17 focused tests passed.

Evidence: tag `abi-final-validation-v2-repaired-r7-2026-08-30`, ABI commit `3f82a9f4d67dda5c8ea13bd59b2d8f1bbd3dd128`, manifest SHA-256 `3691f73624f88c6bfadfaf0d9571e1eaa15fbada834d42af7dda5e4b4279d559`, archive 844,018,841 bytes and SHA-256 `fc50f423986149b5d4670ec9e28698540f64be96034efa26e5704c4469921e88`; independent receipt `audit_outputs/r7-public-reconstruction.json`, SHA-256 `3f0689159a0ec91a63f8080d6dda481af6f202e4b7e26c4213ba06aef1f8a514`. Command: `python -B -m abi_v2.public_reconstruction --manifest <public_release_assets_r7.json> --tag-clone <clean-r7-clone> --workspace <fresh-workspace> --output <receipt>`; result `PASS_PUBLIC_MANIFEST_ONLY_RECONSTRUCTION`, strict certificate identity exact, `17 passed in 22.39s`.

R7 does not publish or prove R97, does not complete V1089, and does not prove foreign-teacher capability preservation.

### “Lossless” boundary

| Meaning | Verdict | Reason |
|---|---|---|
| Artifact losslessness | `BOUNDED_PASS` | Exact hashing, package lifecycle, byte identity, composition, and restoration are supported for the named compatible packages/hosts in R7 and V1089. |
| Foreign-teacher capability preservation | `FAIL` as a general claim | R97 is a strong bounded prospective result, but there is no broad English/domain/open-ended, human-validated, public, multi-host proof that preserves a foreign teacher's functional capability after removal. |

## 3. Repository identity and completeness

Path abbreviations used below:

- `ABI_ROOT` = `C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready\abi_release`
- `LC_ROOT` = `C:\Python310\layercake_merged_nextgen_perfectA_option1_full_ready\layercake_release`
- `AUDIT_ROOT` = `C:\Python310\FLASHLM_V26_COMPOSITIONAL_COMPILER_LATEST_WORKING_COPY\FLASHLM_V26_COMPOSITIONAL_COMPILER_WORKING_COPY`

| Item | Independently observed state |
|---|---|
| ABI current branch/commit | `master`, `a0ea58b613f6f51b7d382298698a42d536cf84ae`; clean; `0/0` versus `origin/master`. |
| ABI remote | `https://github.com/Yoder23/abi.git`; remote `master` resolved to the same commit. |
| Controlling public R7 | annotated tag `abi-final-validation-v2-repaired-r7-2026-08-30`; commit `3f82a9f4d67dda5c8ea13bd59b2d8f1bbd3dd128`; tag object `94738bf...`. |
| Local LayerCake | `master`, `f8a4569508f9f4a20e4babe3b4a942ac887da147`; clean; ahead of public `origin/master` by three commits (`2170c78`, `cb7ce8f`, `f8a4569`). |
| V1089 LayerCake binding | `662c5a9b7264a1a5478c9dfb656f35c450e2504f`, recorded in `evidence/current/ABI_CAPABILITY_COMPILER_PHASE7_CERTIFICATE_V1.json`. |
| R7 LayerCake binding | `a87a653dbdb1a4e5f713baf7bc508d508277e00d`. |
| R97 LayerCake identity | the parent checkpoint is hash-bound (`b6977f...`) and architecture-named, but no exact LayerCake repository commit is bound in the R97 candidate metadata; this limits source-level host reconstruction. |
| ABI file inventory | 4,040 tracked, 0 untracked, 24,077 ignored in the complete local tree. |
| Git object/LFS state | local LFS `fsck` passed and LFS status was clear; public clone fetched 76 LFS objects (~113.93 MiB). `git fsck --full` found no repository corruption. No submodule dependency was declared. |
| Public R7 assets | exactly six release assets: four cakes, one 844,018,841-byte archive, and one 3,321-byte manifest; all independently hashed to their manifest values. |
| Windows checkout qualification | a normal long-path clone omitted three long filenames; a short-path clone with `core.longpaths=true` and `core.autocrlf=false` was clean and was used for the R7 audit. This is an instruction/portability hazard, not a scientific failure of R7. |

Commands included `git status --short --branch`, `git rev-parse HEAD`, `git remote -v`, `git rev-list --left-right --count HEAD...origin/master`, `git fsck --full`, `git lfs fsck`, `git lfs status`, `git submodule status`, `git ls-files`, `git ls-remote`, `gh release view`, and independent SHA-256 calculation.

## 4. Phase 0-8 reconstruction from the frozen contract

The controlling contract is `ABI_ROOT/ABI_CAPABILITY_COMPILER_CAMPAIGN_CONTRACT_V1.json`, SHA-256 `22ca4da553a5289ec4df870c23569daee345882dd7e559d1e558241dee3df5ab`. Its embedded initial status is historical/stale; later state is recorded in `evidence/current/ABI_CAPABILITY_COMPILER_CAMPAIGN_STATE_V1089.json`, SHA-256 `b95eabb47cd737014701b71aedaa14d218da54b6f3d1c9ca3370ad835e3c6d23`.

For each dependent phase, the entrance criterion is completion of the immediately preceding phase. Evidence requirements are the contract's immutable raw artifacts/verifier plus all mandatory accounting fields and three registered fairness views where applicable.

| Phase | Frozen entrance and exact exit criterion | Required evidence | Audit verdict | Current scientific status |
|---:|---|---|---|---|
| 0 | No training authorized. Exit: machine-readable metrics, thresholds, splits, baselines, seeds, stop rules, and artifact identities frozen before results. | Frozen contract/protocol and certificate. | `PASS` | Certificate `evidence/current/ABI_CAPABILITY_COMPILER_PHASE0_CERTIFICATE_V1.json`, SHA-256 `3d95779cdaaa9f71dc0434df5942474fe635e7bf42a95f300924a411e7473884`. |
| 1 | Enter after 0. Exit: immutable provenance-bound artifact passes adequacy, segregation, diversity, correctness, and adversarial verification. | Source/model provenance, normalized IR/raw records, verification and hostile evidence. | `PASS` | Bounded normalized acquisition artifact certified; certificate SHA-256 `da0c69d53b9b3cf7604ba3e56812f3e9468dcbf293bb14a75576c02d4dc87735`. This is not proof of broad capability extraction. |
| 2 | Enter after 1. Exit: credible optimized baselines reproduced across three paired seeds with raw evidence and verifier. | T0, L0, L1, D0, D1, D2; paired seeds; machine rows; sealed 21,000 human preferences under later locked handoff. | `CONDITIONAL` | Machine packet exists, but human evidence is `0/21,000`; later integrated comparison does not carry every mandatory baseline through all views. Human-readiness audit SHA-256 `159c3348e3f8a26349d9482061001c1e78c98a9ac9c35b42acffa9b02561737d`. |
| 3 | Enter after 2. Exit: A0 beats parent, shuffled, bridge-only, label-free, and monolithic controls while passing autonomous quality and repetition gates. | All A0-A4/parent controls, raw quality/repetition rows, verifier, three seeds. | `CONDITIONAL` | Stored machine gates pass, but the Phase 2 prerequisite is incomplete. Current isolated resilience tests also fail because an expected root certificate binding is absent/stale. Certificate-audit SHA-256 `0197bfde51833a09b701f1777478c9e03b3de77947ca9fc65e6c865fb658d3a3`. |
| 4 | Enter after 3. Exit: smallest passing tested budget and adjacent-lower failure reproduced across three seeds under all fairness views. | B20/B40 raw rows, three paired seeds, information accounting, L0/L1/D0 and required fairness views, strict/hostile verifier. | `BOUNDED_PASS` | B40 passes the registered five-route machine suite; B20 fails all three seeds. This is the smallest passing **tested** budget, not a global minimum; equal-compute and achieved-quality views are not fully demonstrated. Independent verifier receipt SHA-256 `07975fe6f09e053184a9d0bbae35831e1c583ba35d8338738817e1dbf36e1a73`. |
| 5 | Enter after 4. Exit: English isolation, domain install/remove/restore, immutable core, and adversarial exclusion pass. | Frozen chemistry/civics/Python rows, exact lifecycle/identity, controls, hostile tests. | `BOUNDED_PASS` | Three-domain behavior and routing pass on the named machine; latent semantic purity is explicitly unproved. Certificate SHA-256 `93265064268f697cf4bfc348551186dc53c7e780be5420c41a74844888c723b6`. Current-master strict verifier fails a changed catalog binding. |
| 6 | Enter after 5. Exit: exact independent-host transfer, selected-only physical execution, non-collapsing composition, and deletion lineage pass. | Fresh compatible initializations, package/provenance/deletion records, raw composition rows, verifier and hostile tests. | `BOUNDED_PASS` | Three fresh initializations on one machine, 900/900 selected outputs and 900/900 composed components; all deployed packages came from Phi-3. Not arbitrary-host or deployed multi-source evidence. Certificate SHA-256 `6ae934c164a46800a1f7224ad5c32433854eb572b89899e135a0272035d20d70`. Current verifier fails a stale/missing Phase 5 root binding. |
| 7 | Enter after 6. Exit: one final teacher-free LayerCake passes quality, isolation, CPU/GPU, genuine-cold-TTFT, memory, sparse execution, and reproducibility gates. | Same artifact for quality/systems, CPU/CUDA raw timing, memory, identities, sparse execution, teacher/receiver-training absence. | `BOUNDED_PASS` | Same-machine seed-104729 integrated product passes stored machine checks, but “cold” does not purge OS filesystem cache and memory fields are asymmetrically named (ABI delta versus comparator total). Certificate SHA-256 `c0b07a8748d7ee84b0e99c6482585a95d7d9483b49d49248071b397b8b47f77b`. Current verifier fails a Phase 4 root binding. |
| 8 | Enter after 7. Exit: clean content-addressed release reproduces claims and survives adversarial verification on independent hardware. | Complete public payload, independent operator attestation, different CPU/CUDA, first preserved run, hostile verifier. | `NOT_EXECUTED` | Same-machine clean rehearsal is not the phase. No independent operator/hardware run exists; `phase8_certified=false`, `release_certified=false`. Local rehearsal SHA-256 `d8b48eba6a1f849e96d0838278618c3d913fc88780212879fbdd1c3ec921d640`. |

## 5. Moonshot claim/evidence matrix

| Mandatory gate | Verdict | Evidence and boundary |
|---|---|---|
| Pre-existing teacher-knowledge extraction | `BOUNDED_PASS` | R97 causally links earlier teacher-derived evidence to a frozen two-hop nonce bridge; it is not broad teacher knowledge. Synthetic-source R8-style results do not count as pre-existing knowledge. |
| Non-enumerable capability extraction | `FAIL` | Prospective prompts are disjoint, but task families and structural augmentation are deliberately specified; no automatic extraction of an unenumerated capability is shown. |
| Broad English fluency | `FAIL` | Registered machine suites are bounded; historical R8-R23 results contain broad/open-generation failures; no human completion. |
| Open-ended generation | `FAIL` | Closed scoring, constrained spans, or supplied-slot generation dominate the positive evidence. |
| Prompt grounding | `BOUNDED_PASS` | B40 registered machine suite only; 1,400 rows/seed, three seeds. |
| Instruction following | `BOUNDED_PASS` | Same B40 boundary; no external human confirmation. |
| Conversation | `CONDITIONAL` | Machine proxy evidence only; no human gate. |
| Supplied-text summarization | `BOUNDED_PASS` | Supplied-text bounded task; not autonomous broad English. |
| Rewriting | `BOUNDED_PASS` | Registered supplied-text machine task only. |
| Email drafting | `BOUNDED_PASS` | Registered machine task only. |
| Tone and format control | `BOUNDED_PASS` | Registered machine task only. |
| Clarification | `BOUNDED_PASS` | B40 route-specific replication only. |
| Abstention | `BOUNDED_PASS` | ABI passes registered abstention gates; L0/L1 fail some all-seed abstention/collapse conditions. |
| Domain-independent reasoning | `BOUNDED_PASS` | R97 proves four-family two-hop nonce reasoning only. |
| Automatic capability discovery | `FAIL` | Negative R14/R16-era evidence is preserved; no successful autonomous discovery result. |
| Semantic labeling | `FAIL` as a general claim | Bounded label artifacts exist, but automatic general labeling/discovery did not pass. |
| English/domain segregation | `BOUNDED_PASS` | Phase 5 proves authoritative-label behavioral exclusion across chemistry/civics/Python. |
| Latent core purity | `FAIL` | The project correctly states that behavioral routing does not prove absence from weights/activations. |
| Selectable domain packaging | `BOUNDED_PASS` | Phase 5: ABI selected `[300,300,300]`, missing-package abstentions `[300,300,300]`, zero unauthorized English-core calls. |
| Package composition | `BOUNDED_PASS` | Phase 6: 900/900 selected specialist observations and 900/900 structured components on three same-machine host initializations. |
| Package removal/restoration | `BOUNDED_PASS` | Exact bounded lifecycle/hash evidence; not arbitrary-host portability. |
| Package provenance/immutability | `BOUNDED_PASS` | 175 records, 3,401 teacher tokens, deletion index complete; deployed packages all select one source. |
| Arbitrary teacher support | `FAIL` | Only named/bounded teachers and artifacts are evidenced. |
| LayerCake ingestion | `BOUNDED_PASS` | Exact named host/package paths; R97 one parent checkpoint, V1089 three compatible initializations. |
| Teacher-free deployment | `BOUNDED_PASS` | R97 and V1089 record teacher absent at inference and zero receiver-training steps in V1089. |
| Host independence | `BOUNDED_PASS` | Fresh initializations and R7 named environments; not arbitrary independent machines. |
| Different-hardware portability | `NOT_EXECUTED` | The required Phase 8 experiment has not run. |
| Global information minimality | `FAIL` | Only B20/B40 in one five-route architecture is tested. |
| Bounded information frontier | `BOUNDED_PASS` | B20 fails all seeds; B40 passes all seeds, subject to mixed-cause B20 failure and incomplete fairness views. |
| Teacher-relative quality | `BOUNDED_PASS` | R97 candidate-teacher paired difference `0.22214`, 95% bootstrap CI `[0.20071,0.24429]`; this is one synthetic reasoning suite, not general generation quality. |
| CPU throughput/TTFT/memory | `BOUNDED_PASS` | Phase 7 stored same-machine medians: 274.319 B/s versus 20.721 B/s; TTFT 3.121s versus 5.064s; memory methodology caveat below. |
| GPU throughput/TTFT/memory | `BOUNDED_PASS` | Phase 7: 881.358 B/s versus L1 65.219 B/s; TTFT 3.349s versus 11.415s; 505,427,456 versus 7,812,537,856 peak CUDA bytes. |
| Human-perceived quality | `NOT_EXECUTED` | `0/21,000`; an AI reviewer cannot substitute. |
| Public clean-clone reproducibility | `BOUNDED_PASS` for R7; `UNVERIFIABLE` for R97/V1089 | R7 reconstructs exactly. The later claims lack all required published payloads. |

## 6. Teacher -> ABI -> LayerCake causal-chain verdict

| # | Causal requirement | Verdict | Exact boundary |
|---:|---|---|---|
| 1 | Capability in independently trained open-weight teacher | `BOUNDED_PASS` | R97 live source score 1,088/1,400; named open-weight teacher lineage. |
| 2 | ABI accessed information causally originating in teacher | `BOUNDED_PASS` | Candidate initialized/trained from 8,129 teacher-derived earlier records; ablations strongly separate trained, parent, and random. |
| 3 | No hidden answers / no complete enumeration | `CONDITIONAL` | R97 exact prompts were unseen, but the capability family, generators, and 96,000 structural augmentations were designed. Non-enumerability is not proved. |
| 4 | Labels without post-hoc leakage | `FAIL` for the general chain | R97 avoids a semantic-label claim; the general discovery/labeling chain remains unproved. |
| 5 | English/specialist information segregation | `BOUNDED_PASS` behaviorally; `FAIL` latently | V1089 Phase 5 is a different artifact lineage from R97. |
| 6 | Package before prospective evaluation | `PASS` within R97 | R96 frozen before R97; zero candidate observations before binding and no retraining/calibration after the catalog. |
| 7 | LayerCake consumed that exact package | `PASS` within R97 | Checkpoint hash `9d3d6b...`; 1,400 bridge invocations and 8,400 deep-adapter invocations. |
| 8 | Unchanged parent could not already perform task | `PASS` within R97 | Parent 60/1,400 versus candidate 1,399/1,400. |
| 9 | Full random/zero/shuffled/removed causal suite | `CONDITIONAL` | R97 includes random (17/1,400) and unchanged parent, but not every listed removal/shuffle control on the same artifact. |
| 10 | Teacher absent at inference | `PASS` within R97/V1089 | R97 metadata records zero source parameters/blocks/logits/activations retained and teacher absent. |
| 11 | No recipient relearning substituted for transfer | `BOUNDED_PASS` | The parent was frozen and install/inference used no receiver training; the capability-specific bridge itself was trained during package production for this host family. Cross-host no-relearning transfer is not shown for R97. |
| 12 | Same final artifact supplies quality and performance evidence | `FAIL` for the moonshot | V1089 supplies its own same-artifact quality/systems evidence, while R97 supplies the strongest causal reasoning evidence. They are not the same final artifact. |
| 13 | Prospective prompt-disjoint generalization | `PASS` within R97 | 1,400 newly bound prompts; checkpoint frozen before catalog. |
| 14 | Live replay | `PASS` within R97 | primary and replay JSONL are byte-identical, both SHA-256 `b4a499...`. |
| 15 | Hostile mutations fail closed | `PASS` within R97; bounded R7 evidence | R97 independently rejected 12/12; R7's public package passed exact strict reconstruction and contains its frozen pre-public hostile receipt. |

The full causal chain first terminates at item 3 for non-enumerable extraction, terminates independently at item 4 for general semantic labeling, and decisively terminates at item 12 because the strongest causal and systems results use different final artifacts.

## 7. R97 independent recomputation

### Recomputed results

| Family | Rows | Candidate | Live teacher | Parent | Random | Teacher-correct retained |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 350 | 350 | 344 | 18 | 6 | 344 |
| 1 | 350 | 350 | 131 | 19 | 8 | 131 |
| 2 | 350 | 349 | 337 | 15 | 3 | 336 |
| 3 | 350 | 350 | 276 | 8 | 0 | 276 |
| **Total** | **1,400** | **1,399** | **1,088** | **60** | **17** | **1,087/1,088 (0.9990809)** |

- Candidate minus live teacher: point `0.2221428571`; paired prompt bootstrap 95% CI `[0.2007142857, 0.2442857143]`.
- Candidate minus parent: `0.9564285714`, 95% CI `[0.9457142857, 0.9664285714]`.
- Candidate minus random: `0.9871428571`, 95% CI `[0.9807142857, 0.9928571429]`.
- Registered collapses: `0/1,400`.
- Physical sparse rows: `1,400/1,400`.
- Per row: one model invocation, one task-cake invocation, one bridge invocation, six deep-adapter invocations; totals `1,400`, `1,400`, `1,400`, `8,400`.
- Live replay: 1,338,609 bytes, byte-identical to primary, SHA-256 `b4a4999228af61a509a7d22f8f6906b3075ce649a52e531b959646fbbdb40b8c`.
- Hostile verifier: `12/12` mutations rejected, zero accepted; stored receipt SHA-256 `cc34388e3b5ebcc11935da2a6f147df2247ebb61724b84957874b5354f1fb12e`.

### Chronology and information boundary

Git chronology is consistent with preregistration: R96 candidate freeze `f4e2dc1`; R97 preregistration `ad726`; catalog/hash seal `60449`; source capture `10d5b5`; candidate/source binding `2923bf`; certification/replay `b92f`; documentation `688361`. The binding reports zero candidate observations before binding and no post-catalog retraining/calibration.

The checkpoint has 530,050 active parameters; training used 8,129 unique imported records, 3,000 optimizer steps, and 96,000 synthetic structural examples. It stores zero source logits, hidden activations, copied source parameters, or transformer blocks. Candidate metadata: `ABI_ROOT/results/length_invariant_span_r96/candidate_v1/metadata.json`, SHA-256 `1cfd15fe838b903ede7e59417f837a1bf00b3664e6deb29bd02876750bcabcb9`.

### R97 verdict

`BOUNDED_PASS_LOCAL` / `UNVERIFIABLE_PUBLIC`

It proves a strong bounded prospective transfer result for a deliberately specified two-hop nonce-reasoning family on one frozen LayerCake parent. It does not prove broad or open-ended capability extraction, automatic discovery, arbitrary teachers/hosts, semantic segregation, human quality, systems advantage, or the full moonshot. A public stranger cannot strictly reproduce it because the unchanged parent checkpoint `results/role_invariant_choice_r76/candidate_v1/model.safetensors` (350,891,472 bytes, SHA-256 `b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e`) and related metadata/tokenizer payloads are ignored and have no identified public release asset.

## 8. V1089 independent recomputation

### B20/B40 frontier

Independent Phase 4 command: `python -m abi.capability_compiler_phase4_b40_frontier_verify --protocol ABI_CAPABILITY_COMPILER_PHASE4_B40_FRONTIER_VERIFY_PROTOCOL_V1013.json --output <audit-output>`; receipt `AUDIT_ROOT/audit_outputs/phase4-frontier-independent.json`, 51,262 bytes, SHA-256 `07975fe6f09e053184a9d0bbae35831e1c583ba35d8338738817e1dbf36e1a73`, status pass.

| Budget/seed | Result |
|---|---|
| B20 / 104729 | Oracle 1,356/1,400; fluent 88, instruction 94, tone 90; raw 1,251; collapses 7; fails. |
| B20 / 130363 | Oracle 1,366/1,400; email 91, fluent 91; raw 1,361; collapse 1; fails. |
| B20 / 155921 | Functional 1,364/1,400 but one preservation regression (`rewriting-0032`); fails. |
| B40 / 104729 | 1,381/1,400; candidate-teacher point `0.102857`; CI `[0.086429,0.118571]`; zero collapse. |
| B40 / 130363 | 1,383/1,400; point `0.104286`; CI `[0.087857,0.120000]`; zero collapse. |
| B40 / 155921 | 1,377/1,400; point `0.100000`; CI `[0.083571,0.116429]`; zero collapse. |

The documented “4,200 observations” are three 1,400-row passing screens, not 4,200 exact answers.

### Information accounting

- 4,112 record memberships; 4,005 unique source attempts.
- 348,372 authoritative teacher-input tokens; 123,167 teacher-output tokens.
- 1,100,756 unique raw prompt UTF-8 bytes; 379,416 unique teacher-output bytes.
- Packed inputs 469,742 tokens and responses 130,885 tokens because membership/packing differs from unique-source accounting.
- Stored logits `0`; stored hidden activations `0`; copied source parameters `0`.
- L0/L1/D0 each received 2,772 optimizer steps, four target-token exposures, and 523,540 response-token exposures (`4 x 130,885`).
- The compact fairness summary does not put ABI and every control on a complete, authoritative equal-training-compute/parameter-seconds/development-trials ledger. This blocks that fairness view.

### L0/L1/D0 quality

| Control | v1 passing by seed | v2 passing by seed | Other gate failures |
|---|---|---|---|
| D0 | `[22,62,0]` | `[41,90,20]` | collapses `[37,275,93]` |
| L0 | `[1367,1362,1352]` | `[1392,1393,1394]` | collapses `[5,0,5]`; zero-collapse gate fails |
| L1 | `[1395,1375,1366]` | `[1395,1400,1396]` | collapses 0; abstention `[100,75,70]`; all-seed abstention gate fails |

D0 has 11,060,800 parameters and ~50-56s training. L0 retains a 3.821B-parameter base plus 25.16M adapters and trained ~2,638-2,748s. L1 retains the base plus 176.16M adapters and trained ~2,695-2,840s. These support a bounded equal-information/deployment result; they do not establish universal method superiority.

### Phase 5-8 integrated facts

- Phase 5: ABI selected `[300,300,300]`; missing-domain abstentions `[300,300,300]`; L1 selected `[252,198,192]` with zero abstentions; D0 selected `[0,0,0]` with `[0,23,0]` abstentions; 13 hostile tests. This is behavioral control-plane exclusion, not latent purity.
- Phase 6: three fresh compatible host initializations on the same machine; 900/900 selected outputs, 900/900 composed components, 300/300 conflict quarantines, 300/300 English identities, zero collapses. All deployed packages select Phi-3 despite a three-source extraction archive.
- Phase 7: one exact English core SHA-256 `acb787...` plus three exact domain cakes; 244 cross-device identities; 120 core and 120 domain timing observations/device; 100 distinct prompts/category/device; 123 selected executions/device; 255,916,588 integrated active tensor bytes; zero receiver-training steps; teacher absent.
- Phase 7 CPU: ABI 274.319 B/s versus 20.721 B/s, ratio 13.2387; TTFT 3.12095s versus 5.06410s; ABI process RSS **delta** 545,308,672 versus comparator peak process RSS 602,959,872.
- Phase 7 CUDA: ABI 881.358 B/s versus L1 65.219 B/s, ratio 13.5139; TTFT 3.34895s versus 11.41497s; ABI process RSS **delta** 518,238,208 versus L1 peak process RSS 5,031,067,648; CUDA allocation 505,427,456 versus 7,812,537,856.
- “Cold” is model-residency cold, but the OS filesystem cache was not purged. It is not an unconditional genuinely cold system-start claim.
- Phase 8 local rehearsal used detached ABI `4b0004b...` and LayerCake `662c5a9...` on the same physical machine, restored three untracked files, ran CPU/CUDA once, and rejected 15 hostile mutations. It is not independent evidence.

V1089 verdict: Phases 4-7 are defensible only as bounded same-machine machine results. Phase 8 is `NOT_EXECUTED`.

## 9. LoRA/distillation fairness assessment

| View | Verdict | Assessment |
|---|---|---|
| Equal imported information | `BOUNDED_PASS` | B40 ABI, L0, L1, and D0 use the exact B40 sequence-information pack and matched four target-token exposures. No source logits/activations/copied weights entered this pack. The claim remains limited to this campaign and excludes complete integrated D1/D2 evidence. |
| Equal training compute | `FAIL` / insufficient accounting | Optimizer steps and target exposure are matched for named controls, but architecture size and wall time differ greatly; ABI's complete comparable compute/parameter-seconds/development-trials ledger is absent from the headline fairness artifact. |
| Matched achieved quality | `NOT_EXECUTED` | L0/L1/D0 did not all reach the registered quality contract; therefore there is no comparison at matched achieved quality. “Controls did not reach gate within this registered search” is a bounded result, not a matched-quality superiority proof. |

Additional fairness defects:

- The frozen contract requires T0, L0, L1, D0, D1, D2, A0, A1, A2, A3, and A4. The final B40 systems comparison promotes L0/L1/D0 but does not carry D1/D2 and every A-control through the same final integrated quality/systems comparison.
- Parameter budgets are not equal: retained LoRA bases are counted in deployment, appropriately for deployment footprint, but this does not become an equal-parameter training comparison.
- Stopping rules, seeds, and exposures are substantially documented; complete matched development-trial accounting is not.
- D0 receipts contain `system: null`, a low-severity schema/provenance defect.

Conclusion: ABI has a bounded advantage over the registered L0/L1/D0 attempts under equal imported sequence information and the chosen deployment definition. Universal superiority over LoRA or distillation is not proved.

## 10. Statistics and benchmark integrity

Positive findings:

- R97 has 1,400 distinct bound rows, paired prompt-level bootstrap intervals, complete raw preservation, no removed failures, and byte-identical replay.
- V1089 headline quality has 1,400 prompts/seed and three seeds.
- Phase 7 runtime has 120 observations and 100 distinct prompts per named category/device, enough for the frozen median and p95 requirements.
- No p99 claim is authorized: the contract requires 1,000 observations and only 120 are present.
- Cross-model throughput is reported in bytes/second, avoiding tokenizer-dependent token-rate promotion.
- Negative B20/control/history evidence is preserved.

Qualifications:

- OS filesystem cache was not purged for the one-request TTFT measurements.
- ABI memory uses process-RSS delta while comparators are reported as peak total RSS; this is not a perfectly symmetric memory endpoint.
- B20 seed 155921 fails because of one preservation regression, so the adjacent-lower result is operational failure, not purely a demonstrated information-capacity threshold.
- The evidence reviewed does not establish that all development trials across ABI and controls were completely costed.

## 11. Public reproducibility and missing-artifact inventory

### Audit A: public clean-clone

R7: `PASS_PUBLIC_MANIFEST_ONLY_RECONSTRUCTION`.

The clean short-path tag clone was at exact commit `3f82a9f...`, clean, with the annotated tag resolving to it. Only public URLs in `public_release_assets_r7.json` were used. All assets matched size/hash, extraction contained no forbidden development directories, strict raw recomputation exited 0, stdout SHA-256 exactly matched `17973df22cb31eddb3b47c6137aa71b68f32cd9d3010cfcddd232b2a1afce488`, and 17 focused tests passed.

### Audit B: complete local scientific evidence

R97 strict and hostile replay pass locally. V1089 contains coherent local Phase 4-7 evidence and a same-machine Phase 8 rehearsal. Local availability is not public availability.

### Inventory

| Class | State | Consequence |
|---|---|---|
| Tracked | 4,040 ABI files at current master; most protocols/raw results are tracked, including R97 raw JSONL and bridge. | Public Git can supply these, subject to LFS and Windows long-path handling. |
| Untracked | 0 in the local ABI working tree. | No ordinary untracked dirt, but ignored files remain scientifically material. |
| Ignored | 24,077 paths, including generated models/results. | Ignored is not public. Critical R97 parent payload is in this class. |
| Git LFS | Local `fsck` passes; public clone retrieved 76 objects. | LFS itself is healthy for tracked objects; it cannot supply never-tracked payloads. |
| Phase 8 manifest-only | `results/abi_capability_compiler_phase8/readiness_v1073/manifest.json`, 52 files, 281,108,851 total bytes, SHA-256 `b0e0c8f1ffcc52fc7464d52840ab7a086313fa90747158b0b9cff29d0abc8d86`. | A hash in a manifest is not publication. |
| Manifest `tracked:false` #1 | English cake, 253,216,208 bytes, `acb787...`. | Published in R7 and hash-available, but R7 is a different frozen release lineage. |
| Manifest `tracked:false` #2 | Phase 7 materialization result, 1,419 bytes, `d22684e34aabe39c4b590e3d22a547709a75cb8d17386700d410ca407ca6ee8c`. | Absent from R7 assets/public clean master checkout; V1089 clean reconstruction is incomplete publicly. |
| Manifest `tracked:false` #3 | Phase 7 verifier result, 5,524 bytes, `e34c01e02552866893e6f6c9b07be389675d63debb1ce5f85968cc98954ddf92`. | Same missing-public-artifact blocker. |
| R97 missing public payload | unchanged parent model, 350,891,472 bytes, `b6977f...`, plus associated metadata/tokenizer material. | Public R97 strict replay is `UNVERIFIABLE`. |
| Other acknowledged local-only corpus | R13 documentation reports 243,426,540 bytes of bulk adapters/raw rows unpublished. | Claims depending on it are not clean-clone reproducible. |

## 12. Verification execution and defects

| Command | Result |
|---|---|
| `python -m pytest -q` | `84 passed, 4 skipped, 1 warning in 96.23s`. All four skips are exact historical/current-tree qualifications, not current-tree passes. |
| `python -m abi status --json` | Ready for human/independent review; `0/21000`; Phase 8 false; release false. |
| `python -m abi self-check` | Passes deterministic package/self-check with empty English domain labels; this is not scientific moonshot validation. |
| R97 strict verifier | Pass; independent receipt hash `59438f...`. |
| R97 hostile verifier | Pass; 12/12 rejected; receipt hash `cc3438...`. |
| Phase 4 B40 frontier verifier | Pass; receipt hash `07975f...`. |
| R7 clean public reconstruction | Pass; strict identity exact; 17 tests pass. |
| Selected current-master tests in isolation | `2 failed, 16 passed, 9 skipped`; Phase 3 resilience expects an absent/stale root Phase 1 certificate; Phase 0 tests skip absent root certificate; R7 tests skip exact-source mismatch. |
| Current Phase 5 verifier | Fails implementation binding after `catalogs/english_and_first_domains_certification_v6.json` changed. |
| Current Phase 6 verifier | Fails absent/stale root Phase 5 certificate binding. |
| Current Phase 7 verifier | Fails absent/stale root Phase 4 certificate binding. |
| Current Phase 8 readiness/rehearsal verifiers | Fail absent/stale root Phase 7/6 certificate bindings. |

The difference between the full-suite pass and isolated-test failures demonstrates test-order/materialization dependence. No code or evidence was repaired during this audit.

## 13. Findings by severity

### Critical

1. **Failed scientific gate / missing evidence:** no one coherent final artifact establishes the complete teacher -> ABI -> LayerCake quality, segregation, portability, systems, human, and public chain. R7, V1089, and R97 are not combinable.
2. **Failed scientific gate / overbroad claim:** broad teacher-competitive open-ended English and foreign-teacher capability preservation are not demonstrated. Artifact losslessness cannot substitute.

### High

1. **External human requirement:** `0/21,000` sealed preferences.
2. **Independent-hardware requirement:** no independent operator on genuinely different CPU and CUDA hardware; Phase 8 is not executed.
3. **Missing public artifact:** R97's 350,891,472-byte parent checkpoint and associated payloads are absent from public release assets.
4. **Missing public artifact:** V1089 requires two unpublished `tracked:false` Phase 7 result JSONs; its public reconstruction is incomplete.
5. **Implementation/reproducibility defect:** current Phase 5-8 verifiers do not replay from current master because root/binding identities are stale or absent; selected tests depend on suite order.
6. **Failed fairness gate:** equal training compute and matched achieved quality are not demonstrated; D1/D2 and all mandatory A-controls do not appear in one final integrated comparison.
7. **Measurement defect:** “cold” TTFT is model-residency cold without OS-cache purge, and RSS comparisons mix ABI delta with comparator total peak.

### Medium

1. **Evidence-index defect:** `docs/abi_proof_ledger.json` (SHA-256 `6c47d01089c4e8c0f1cea63978a5e14400d46319f192683e03754ef13287d38c`) is stale relative to `docs/ABI_PROOF_LEDGER.md` (SHA-256 `d20778a1d0bee08970759cac0bc9005d6768adbb24bb41d5654740846f63ec48`): it omits R97 and disagrees on at least C3's bounded lineage.
2. **Overbroad interpretation risk:** B40 is the smallest passing tested B20/B40 setting, not a global information minimum; B20's third-seed failure is a single preservation regression rather than clean information insufficiency.
3. **Provenance scope:** all deployed Phase 6 domain packages select Phi-3; a three-source extraction archive does not prove deployed multi-source quality.
4. **Host identity gap:** R97 binds a parent checkpoint hash but does not bind an exact LayerCake repository commit in its candidate metadata.
5. **Public portability:** Windows needs short paths/`core.longpaths=true`; ordinary long-path checkout can appear dirty/missing.

### Low

1. **Schema defect:** D0 receipts contain `system: null`.
2. **Terminology risk:** `abi self-check` can pass while all external scientific gates remain open; it should not be presented as scientific certification.

## 14. Claims that must be removed or narrowed

Remove unless new evidence is produced:

- `FULL_MOONSHOT_PASS`, “full moonshot proven,” or any equivalent;
- “lossless foreign-teacher capability transfer”;
- “general/broad English extraction,” “open-ended fluent English preserved,” or “general knowledge extraction”;
- “automatic capability discovery” or general semantic auto-labeling;
- “latent English-core purity”;
- “arbitrary teacher,” “arbitrary host,” or “hardware-independent” support;
- “global minimum information”;
- “universal superiority over LoRA/distillation”;
- “human validated,” “independently reproduced,” “release certified,” “novel,” or “groundbreaking.”

Narrow positive claims to:

- R7: exact published bounded package/runtime/conformance reconstruction on its named environments;
- V1089: bounded same-machine five-route B40 machine quality, three-domain behavioral exclusion/composition, and CPU/CUDA results for its exact product;
- R97: bounded local prospective two-hop nonce-reasoning transfer on one frozen compatible LayerCake parent;
- Phase 5: behavioral authoritative-label exclusion, not latent purity;
- Phase 4: smallest passing **tested** B20/B40 configuration, not global minimality;
- comparisons: advantage over the registered L0/L1/D0 attempts under equal imported sequence information, not universal method superiority.

Even the phrase `ABI TECHNICAL MOONSHOT: PROVEN` should be replaced by the less ambiguous `R7 BOUNDED CAPABILITY-RUNTIME/CONFORMANCE RESULT: PUBLICLY RECONSTRUCTED`.

## 15. Exact unresolved gates

1. Freeze one coherent final artifact that simultaneously carries the causal teacher-transfer, English/domain quality, package lifecycle, and systems claims.
2. Publish every required payload and raw result by durable content address, including R97's parent checkpoint/metadata/tokenizer and all V1089 `tracked:false` files.
3. Make the exact public clean-clone replay pass without suite-order materialization and with every current strict verifier live.
4. Complete the sealed `21,000` independent human preferences with three real raters.
5. Execute the first preserved Phase 8 run by an independent operator on genuinely different CPU and CUDA hardware.
6. Demonstrate broad prompt-grounded, teacher-relative open-ended generation on prospective hidden tests with no supplied-slot shortcut.
7. Run every contract-mandatory baseline/control (T0, L0/L1, D0/D1/D2, A0-A4 and parent/removal controls) on the same final artifact and frozen suite.
8. Complete equal-information, equal-training-compute, equal-deployment, and matched-achieved-quality views with full cost/development-trial accounting.
9. Measure genuinely cold single-request TTFT and symmetric memory endpoints; retain at least 20 repeats and do not promote p99 without 1,000 observations.
10. Demonstrate automatic discovery/labeling and distinguish behavioral exclusion from latent purity.
11. Demonstrate no-relearning transfer to separately built compatible LayerCake hosts and bind exact host commits.
12. Perform a serious prior-art review before any novelty claim.

## 16. Minimal evidence-supported next experiments

1. **Publication/replay first:** create an additive, immutable release for one chosen final lineage containing every checkpoint, tokenizer, raw row, result JSON, protocol, and exact ABI/LayerCake source commit. Have a fresh reviewer reconstruct it from public assets only. Do not rerun favorable science merely to fill the release.
2. **Coherent-artifact causal matrix:** preregister one hidden prospective suite spanning open-ended English and reasoning; run unchanged parent, random, zero, shuffled, label-free, monolith, host/package/adapter/capability removal, LoRA, and D0/D1/D2 controls against the same final product.
3. **Fairness completion:** report equal imported information, equal compute/parameter-seconds and trials, equal deployment, and matched achieved quality separately. If a comparator wins, preserve it.
4. **External gates:** after public replay passes, send that immutable product to the three human raters and an independent different-hardware operator. Preserve the first results.
5. **Systems correction:** preregister OS-cache-purged cold starts and symmetric process/CUDA memory definitions.
6. **Discovery/purity only if claimed:** use a held-out, unannounced capability and independent semantic probes; otherwise delete those claims.

Additional same-machine repetitions of the existing Phase 8 rehearsal are not the next scientific experiment.

## 17. Could another clean reviewer reproduce each result?

| Result/claim | Public clean reviewer? | Local complete-tree reviewer? |
|---|---|---|
| R7 bounded runtime/conformance | `YES` | `YES`; independently reconstructed here. |
| R97 numerical 1,400-row replay | `NO — UNVERIFIABLE` | `YES`, with the ignored parent payload present; independently recomputed here. |
| V1089 Phase 4 frontier | `PARTIAL` | `YES`; independently recomputed here. |
| V1089 Phase 5-7 stored result | `NO — UNVERIFIABLE AS A CLEAN CURRENT REPLAY` | Historical artifacts are locally inspectable, but current-master strict verifiers fail frozen bindings. |
| V1089 Phase 8 clean reconstruction | `NO` | Same-machine rehearsal only; not the required phase. |
| Human quality | `NO` | Not executed. |
| Independent-hardware portability | `NO` | Not executed. |
| Full moonshot | `NO` | Not proved. |

## 18. Audit receipts

The audit-created receipts are additive and outside the ABI/LayerCake repositories:

- `AUDIT_ROOT/audit_outputs/r97-strict-independent.json` — SHA-256 `59438fbff7e07a2d81f78b9c9905d473b09678892026b33cbd30cc0a02dfe30c`
- `AUDIT_ROOT/audit_outputs/r97-hostile-independent.json` — SHA-256 `cc34388e3b5ebcc11935da2a6f147df2247ebb61724b84957874b5354f1fb12e`
- `AUDIT_ROOT/audit_outputs/phase4-frontier-independent.json` — SHA-256 `07975fe6f09e053184a9d0bbae35831e1c583ba35d8338738817e1dbf36e1a73`
- `AUDIT_ROOT/audit_outputs/r7-public-reconstruction.json` — SHA-256 `3f0689159a0ec91a63f8080d6dda481af6f202e4b7e26c4213ba06aef1f8a514`

No ABI or LayerCake source, protocol, certificate, or evidence file was modified.
