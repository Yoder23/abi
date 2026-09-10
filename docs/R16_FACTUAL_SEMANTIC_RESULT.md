# R16 bounded factual acquisition and semantic segregation

Status: `BOUNDED_LOCAL_REPLICATED_BLIND_PASS; PUBLIC RECONSTRUCTION PENDING`

R16 is the first ABI campaign to extract factual content already present in an
unchanged open-weight teacher, assign that content to registered semantic
namespaces, and execute the resulting immutable packages with the teacher
absent. It is a bounded structured-memory result. It is not fluent English
transfer, autonomous open-world discovery, neural-weight transplantation, or
LayerCake product acceptance.

## Frozen protocol and chronology

The implementation was frozen before the held-out selection was revealed:

- implementation freeze: `b1a3904`;
- preregistration and secret commitment: `b8e97ba`;
- secret commitment:
  `cfb3f707f7ac4eb6a9a4a94246fdcc9b520ddcf338bc944cdf580a296737dba7`;
- reveal: `bd5e571`;
- reveal-file SHA-256:
  `226d74875f6b2a31b50a90d1ee88b5707de61930f45b726434401143f699d9ce`;
- evidence seal: `32b0595`.

The hidden selection contained 16 facts: eight chemistry atomic-number facts
and eight geography national-capital facts. Extraction and evaluation used
three disjoint paraphrases per fact. Free-generation prompts did not include
the answer. The compiler did receive a registered 12-value candidate set with
the correct answer present exactly once, plus the teacher score for each
candidate; no field labeled the correct value. The registered ontology and
candidate vocabulary mean the experiment does not test autonomous ontology or
candidate discovery.

## Result

| Gate | Result |
| --- | ---: |
| Answer-free source generations | 48/48 exact |
| Candidate-sequence selections | 48/48 exact |
| Semantic labels | 48/48 exact |
| Held-out evaluation source answers | 48/48 exact |
| Package answers | 48/48 exact |
| Package/source agreement | 48/48 exact |
| Target namespace only | 48/48 exact |
| Other namespace only | 48/48 abstained |
| Capability removed | 48/48 abstained |
| Rotated-score causal control | 0/48 exact |
| Live source rows replayed | 96/96 byte-exact |
| Live evaluation rows replayed | 48/48 byte-exact |
| Source residual rows verified | 48/48 |
| Declared artifacts verified | 13/13 |
| Hostile mutations rejected | 21/21 |
| Source training | 0 steps |
| Teacher present at package execution | No |

The physical compiler ran as a pure-standard-library worker in a Linux
pivot-root, no-network capsule. It received question/subject strings,
candidate strings, and teacher sequence scores. Capability archives, labeled
answer/oracle fields, fact IDs, secret/reveal files, success IDs, the teacher,
and the development tree were absent. The correct value was nevertheless one
of the candidates, as required by this closed-candidate protocol. The compiler
emitted two canonical packages:

| Namespace | Facts | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `chemistry/periodic-table` | 8 | 584 | `b053fc982bb44c230ef8f809cc6d195c0fe2c1d5d6b5680628749c5af9f5f575` |
| `geography/national-capitals` | 8 | 638 | `228e537e6f50f30e4dbb9d20a8b4c23be3992d95367cd9d249fdd51a14065136` |

Removing a package removed its answers, installing only the other namespace
caused abstention, and rotating the score vectors destroyed performance. These
controls establish causal use and domain segregation for this registered
closed-candidate workload.

## Imported-information accounting

| Item | Measured value |
| --- | ---: |
| Raw / unique source prompts | 96 / 96 |
| Rendered source-prompt UTF-8 bytes | 18,334 |
| Rendered source-prompt token instances | 3,206 |
| Question-only UTF-8 bytes | 4,126 |
| Teacher-generated tokens | 276 |
| Teacher output bytes | 372 |
| Candidate string / token instances scored | 576 / 1,146 |
| Candidate score values imported | 576 |
| Source bundle | 33,971 bytes |
| Final packages | 1,222 bytes |
| Final structured fact records | 16 |
| Frozen source parameters in packages | 0 |
| Bridge parameters trained | 0 |
| End-to-end acquisition wall time | 68.273 seconds |
| Peak GPU memory | 15,327,108,096 bytes |

The run also stored 172,032 hidden-activation values (688,128 tensor bytes;
688,216-byte file) solely for audit evidence. The isolated compiler consumed
zero hidden activations. Source-inference-only wall time and peak CPU RAM were
not separately measured and must not be inferred from the aggregate.

## Verification and evidence

The complete local evidence is under
`results/factual_semantic_r16/heldout_v1/`. A fresh live rerun is under
`results/factual_semantic_r16/heldout_v1_live/` and reproduced every declared
source, evaluation, package, and bundle artifact byte-for-byte. The expanded
strict verifier binds all declared artifacts, the complete pinned Qwen source
snapshot, the live replay, and every residual row. It passed. The expanded
hostile audit rejected all 21 missing, corrupt, stale, and hash-consistent
forgery cases. Strict-v3 opens the seven live files, verifies all four physical
extraction trees, and directly rehashes the 10-file source snapshot.

The controlling repaired local certificate is
`results/factual_semantic_r16/heldout_v1_certificate_v4_revision_002.json`,
evidence SHA-256
`b089b538ca444d3b3ae62995cc87064e98fc76e70c72f05f3df422ecac8d203c`.
Earlier certificates and verifier receipts remain preserved as historical
evidence. The blind review of commit `b63bd55` passed with zero Critical or
High findings; its three Medium and three Low findings are preserved in
`results/factual_semantic_r16/blind_redteam_b63bd55.md`. The additive repairs
are described in [R16_POST_REVEAL_ASSURANCE_AMENDMENT.md](R16_POST_REVEAL_ASSURANCE_AMENDMENT.md).

The historical public-v2 receipt refers to a protocol digest whose file was
not committed. A post-reveal requalification under the available committed
protocol reproduced every public artifact and gate byte-for-byte. That is
assurance evidence, not a retroactive chronology repair. A new hidden
replication bound to the corrected public receipt is therefore required before
R16 is promoted beyond its current local claim.

## Claim boundary

R16 supports only this claim:

> On the preregistered 16-fact, two-domain, closed-candidate workload, ABI
> extracted facts from an unchanged Qwen2-7B-Instruct teacher, assigned them to
> the registered chemistry/geography namespaces, emitted two immutable
> structured packages, and reproduced teacher answers exactly with causal
> removal and namespace-isolation behavior while the teacher was absent.

R16 does not establish:

- autonomous discovery of arbitrary facts, candidate vocabularies, or domains;
- a fluent or domain-pure English neural substrate;
- free-form teacher-quality generation;
- production LayerCake ingestion;
- global or registered minimum-information optimality;
- native neural transplantation; or
- superiority to LoRA, distillation, or fine-tuning.

## Clean replication

The post-review repair and corrected public prerequisite were frozen in
`a3255e0`. A new secret commitment was preregistered without the reveal in
`ec765cf`; the reveal followed in `4eb64ba`. Nineteen code/protocol hashes and
the corrected public receipt were bound before reveal.

The clean replication independently passed the same gates on a new 16-fact
selection. It shared five facts with v1 and expanded the combined held-out
coverage to 27 distinct facts. Its source/package metrics were again exact:

| Replication gate | Result |
| --- | ---: |
| Selected facts | 16 |
| Extraction generation / candidate / label | 48/48 each |
| Evaluation source / package / agreement | 48/48 each |
| Target-only / other-only / removed | 48/48 each |
| Rotated-score control | 0/48 |
| Live files byte-exact and reopened | 7/7 |
| Physical extraction trees / package files | 4/4 / 8/8 |
| Hostile mutations rejected | 21/21 |
| Source snapshot directly rehashed | 10 files / 15,242,778,262 bytes |

The replication emitted 1,240 package bytes, consumed 96 rendered source
prompts (18,358 UTF-8 bytes; 3,212 input tokens), 306 generated tokens, 462
output bytes, and 576 candidate scores. It stored 172,032 residual values only
for audit and supplied none to the compiler.

The replication certificate is
`results/factual_semantic_r16/heldout_v2_certificate.json`, evidence SHA-256
`068388facd7c9793656a98acdad1ff4cb949ee6aea055a624f2f4c1e04867320`.
It remains bounded to the same registered closed-candidate claim.

A fresh blind review of exact evidence commit `26fb029` returned `PASS` with
zero Critical, High, Medium, or Low findings. It independently recomputed the
19/19 preregistered bindings, source inventory, raw rows, packages, strict and
hostile evidence, accounting, and certificate. Permissioned reconstruction of
both physical compiler capsules regenerated the primary and rotated-control
packages exactly. The report is
`results/factual_semantic_r16/blind_redteam_26fb029.md`.

The full ABI moonshot remains open. R7 remains the controlling public release
until the replicated R16 seal receives durable publication and clean external
reconstruction within this bounded claim.
