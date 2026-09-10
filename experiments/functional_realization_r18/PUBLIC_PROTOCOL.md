# R18 public factorized functional-realization prerequisite

## Motivation

R17 showed that exact teacher-string qualification is the wrong boundary for
natural language. Its sole repair produced 114/120 exact strings, including
several fluent variants and one semantic failure. Independent modal templates
derived from its extraction rows would score only 46/48 on the lexically
disjoint evaluation.

R18 tests a materially different representation: a feature-factorized grammar
package. It preserves every raw teacher output, evaluates teacher agreement
separately, and promotes only functional English realization relative to the
registered semantic frames.

## Frozen compiler

The physical compiler receives the 72 R17-v2 extraction records containing
only anonymous IDs, 24 feature signatures, supplied lexical slots, and free
teacher outputs. It receives no prompts, expected outputs, evaluators,
evaluation records, source model, success IDs, or hidden result.

It converts each parseable teacher output into a slot template. For each
signature it chooses the template with maximum empirical support after pooling
only with the signature that differs in subject number. This registered
factorization contains no English surface token or expected answer. Ties are
resolved canonically. At least two parseable observations per signature are
required. The worker runs in a Linux pivot-root, no-network capsule.

The package contains only the selected templates and their evidence counts.
The generic runtime fills supplied lexical slots. A mood-permutation control
keeps identical teacher outputs and slots while assigning declarative evidence
to question signatures and vice versa.

## Public gates

- strict recomputation of the complete R17-v2 source evidence;
- 72 source extraction records and 48 lexically disjoint evaluation records;
- at least two parseable source observations for every signature;
- exactly 24 factorized templates in one physical English package;
- 48/48 package functional exactness on evaluation;
- no package regression on a row the source answered exactly;
- package functional exactness strictly greater than independent modal
  consensus;
- teacher/package exact string agreement reported but not used as the Track B
  quality verdict;
- removal abstention on 48/48 rows;
- mood-permutation control at or below 10% exact;
- teacher absent at package execution; zero source and host training;
- complete accounting for inherited prompts/tokens/bytes and package bytes;
  and
- fail-closed verification of required files, hashes, rows, package contents,
  isolation receipts, and recomputed claims.

## Decision rule and ceiling

A public pass authorizes only a preregistered hidden lexical replication of
the frozen compiler. A failure closes the factorized-template mechanism.

Even a held-out pass would prove bounded teacher-derived compositional English
surface realization for this registered 24-signature family. It would not
prove unrestricted fluency, autonomous capability discovery, arbitrary
domains, production LayerCake acceptance, a globally minimal English core, or
superiority to LoRA or distillation.
