# R16 blind red-team report

- Reviewed commit: `26fb02908831cab05079ee1ba9821e401ce222d7`
- Bounded claim: `BOUNDED_FACTUAL_EXTRACTION_AND_SEMANTIC_SEGREGATION`
- Verdict: `PASS`
- Findings: Critical 0; High 0; Medium 0; Low 0

## Chronology and identity

The review independently confirmed the linear chronology:

- implementation freeze: `a3255e0dfdb4b61b6dc344e5aa3c6c421662873b`;
- preregistration: `ec765cf35a4652cb8dac1bb48d5904ca9dee19b3`;
- reveal: `4eb64ba432661bbf349be8c8bc8388d6e8d812f5`;
- evidence seal: `26fb02908831cab05079ee1ba9821e401ce222d7`.

Preregistration added only `heldout_v2.json`; reveal added only
`reveals/heldout_v2.json`. All 19/19 preregistered code and protocol hashes
matched the implementation freeze and reviewed tree. The reveal SHA-256 was
`b2a0cd0c70aebeade130607697aec614bbfbbb53f67d3b847cdbad2f94f14e89` and
its secret matched commitment
`df49d8a75597a77dcbba309bb7c318b85cb560dd7d441d5c7301c1b9a7581a7e`.
The corrected public prerequisite receipt SHA-256 was
`10e7cd6a014d6dd6e283225934c83c2bf64bd8c3ff583c86d39331dfd1d881d4`.

The complete pinned source inventory rehashed exactly: 10 files,
15,242,778,262 bytes, evidence
`bdb164548ca29e580e465f520df01fce77524eeec01e807c413443751b4f004b`.

## Independent scientific recomputation

- 16 selected facts, eight per namespace; five v1/v2 overlaps and 27 distinct
  facts across both selections.
- 96 raw source rows: 48 extraction and 48 evaluation.
- 0/96 question-answer leakage instances.
- 96/96 open-answer exactness.
- 48/48 candidate selections and exactly one correct value in every
  12-candidate set.
- 48/48 semantic namespace classifications.
- Compiler bundle: 48/48 raw-row bindings, exact allowed schema, and zero
  answer, oracle, namespace, fact ID, secret, reveal, or success-ID fields.
- Residual tensor shape `(48, 3584)` and 48/48 row hashes.
- Chemistry package: eight facts, 583 bytes,
  `c3358ddb31386602ede787628c6e6b3e791e53ed647fe73d64db25f2ba60e9c4`.
- Geography package: eight facts, 657 bytes,
  `12a590b908aa2e82ca6344a613f8843a2d69e9ca5d2d6162d94448c8a4b5286c`.
- Combined, target-only, and source-agreement evaluation: 48/48 each.
- Other-domain and removal abstention: 48/48 each.
- Rotated-score control: 0/48.
- Stored live files reopened and byte-identical: 7/7.
- v1 run/live substitution into the v2 evidence chain failed closed.

Strict-v4, hostile-v4, the replication certificate, and accounting-v3 all
recomputed exactly. Strict evidence was `4741e36b...4f0d6`; hostile evidence
was `fa998884...34ea2` with 21/21 rejected mutations; certificate evidence was
`068388fa...67320`; and accounting evidence was `963d05b9...33cc0`.

Accounting independently confirmed 96 rendered prompts, 18,358 rendered
UTF-8 bytes, 3,212 source prompt tokens, 576 candidate strings and scores,
1,173 candidate tokens, 306 generated tokens, 462 output bytes, 1,240 final
package bytes, and zero compiler-consumed hidden activations, bridge training,
or source training.

## Physical replay and tests

A fresh 96-prompt GPU source phase loaded the exact pinned four-shard source.
The first end-to-end wrapper encountered WSL sandbox denial
`Wsl/Service/E_ACCESSDENIED`, not a scientific mismatch. With the required WSL
permission, both physical compiler reconstructions completed with worker exit
0, pivot-root/no-network isolation, no old root or Windows mount, 48 records
consumed, and exact primary and rotated-control package bytes.

Focused tests passed 12/12 and the supported suite passed 86/86 using
`C:\Python310\python.exe`.

## Claim ceiling

The evidence proves only extraction of a secret 16-fact selection from the
unchanged pinned source within a frozen 48-fact chemistry/geography universe,
using answer-free generation, registered 12-candidate sequence scoring, and a
registered two-domain ontology. It proves immutable teacher-absent structured
lookup packages and exact namespace/removal controls.

It does not prove autonomous labeling or discovery, open-world or arbitrary-
domain extraction, fluent English extraction, teacher-behavior cloning,
free-form linguistic realization, production LayerCake ingestion, global
minimality, independent-hardware/public reproduction, or superiority to LoRA
or distillation. The full ABI moonshot remains open.
