# R16 bounded factual acquisition and semantic segregation

Status: `BOUNDED_LOCAL_PASS; BLIND REVIEW PENDING`

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
three disjoint paraphrases per fact. Source prompts did not include the
answers. The registered ontology and a 12-answer candidate vocabulary per fact
were supplied by the protocol, so the experiment does not test autonomous
ontology or candidate discovery.

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
| Hostile mutations rejected | 15/15 |
| Source training | 0 steps |
| Teacher present at package execution | No |

The physical compiler ran as a pure-standard-library worker in a Linux
pivot-root, no-network capsule. It received question/subject strings,
candidate strings, and teacher sequence scores. Capability archives, answers,
fact IDs, secret/reveal files, oracle fields, success IDs, the teacher, and the
development tree were absent. The compiler emitted two canonical packages:

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
| Unique prompt UTF-8 bytes | 4,126 |
| Teacher-generated tokens | 276 |
| Teacher output bytes | 372 |
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
hostile audit rejected all 15 missing, corrupt, stale, and hash-consistent
forgery cases.

The final local certificate is
`results/factual_semantic_r16/heldout_v1_certificate_v3.json`, evidence SHA-256
`147a0a204e93916e14a98875d16b9e3fc387147c32386a9516850b1fc3f65703`.
Earlier certificates and verifier receipts remain preserved as historical
evidence; the v3 certificate adds accounting and stronger verification rather
than rewriting them.

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

The full ABI moonshot remains open. R7 remains the controlling public release
until R16 receives blind review, durable publication, and clean external
reconstruction within this bounded claim.
