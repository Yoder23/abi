# ABI foreign-transfer research tracks

Updated: 2026-09-09

ABI now separates two scientific targets that must never share a verdict.

## Track A: behavioral transplantation

Target:

`F_package(x) = F_frozen_teacher(x)`

The package must reproduce the frozen teacher's committed outputs, including
teacher mistakes, on public and subsequently committed held-out distributions.
Oracle capability labels may score quality separately, but they may not alter
the transplantation equality verdict.

A Track A pass requires:

- a frozen conventional source model;
- source outputs committed before package evaluation;
- no oracle answers, evaluation rows, or success identifiers available to the
  extractor;
- exact package/source output equality on the registered public prerequisite;
- a separately committed held-out evaluation after the public prerequisite;
- no teacher during package execution;
- zero recipient optimization; and
- fail-closed live replay from immutable artifacts.

Status: `OPEN`. R12-A did not qualify because the canonical package corrected
source errors rather than reproducing them.

## Track B: capability extraction and canonicalization

Target:

`F_package(x) = F_registered_capability(x)`

The package may outperform the source's native execution if the preregistered
evidence interface contains enough information to identify the capability.
Teacher-relative accuracy and oracle-relative accuracy must both be reported.

A Track B pass requires:

- a preregistered capability family, source interface, extraction budget, and
  oracle;
- a held-out capability commitment created before reveal;
- a frozen conventional source at extraction time;
- no oracle answers, evaluation rows, or rule seed available to the extractor;
- exact package/oracle behavior on the registered held-out matrix;
- explicit measurement of package/source disagreements;
- no teacher during package execution;
- zero recipient optimization; and
- fail-closed verification and live replay.

For an enumerable family, the report must state whether probes exhaust the
capability specification. Such a result is a bounded behavioral-compilation
proof, not evidence of general English or domain extraction.

Status: `R13-B ENUMERABLE PASS; R14 OUTPUT-ONLY FAILURE; R15A LEARNED-DELTA PASS`.

R13-B passed this track for four held-out finite transition tables. The result
does not advance Track A: package/source agreement on long compositions was
only 13.09% to 21.68%. It also does not advance the non-exhaustive synthetic,
factual, linguistic, or cross-family stages below.

R14 removed atomic probes and queried 256 mixed behaviors from a space with
4,642,668,576 registered evaluation cases per capability. It recovered one of
three fresh capabilities exactly on 10,000 unseen-depth and 1,000
order-counterfactual cases; two latent selections were wrong. R14 therefore
establishes a bounded one-capability partial success but does not pass stage 2.
The next valid frontend must use materially different controlled foreign
neural-state evidence rather than a nearby output-query or source-LoRA sweep.

R15A executed that neural-state pivot. Its generic frontend recovered 8/8
fresh capabilities from anonymous before/after Qwen output-weight deltas, and
the packages were exact on 88,000 registered oracle cases. Strict, live, and
hostile verification passed. This advances Track B only for a deliberately
learned synthetic capability and a before/after delta; it does not advance
Track A or the pre-existing factual/linguistic stages.

## Progression beyond the finite-table control

Track B must advance through separately registered stages:

1. exhaustively enumerable finite capability;
2. non-exhaustive synthetic capability with genuine generalization (bounded
   learned-delta prerequisite passed in R15A);
3. factual or specialist knowledge with contamination-resistant evaluation;
4. linguistic and reasoning behavior; and
5. cross-family replication and LayerCake product acceptance.

No stage inherits the claims of a later stage.
