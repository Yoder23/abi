# R117 / work package 1: grounded normalization, not another teacher qualification

Register before scoring or creating the prospective panel, 2026-11-23. Preserve
R116 at c2ba1e686e55eb2ab007f59ad913252c90139c3c, decision SHA-256
d42f8c8b71bf8fde4891641ea2374a30289073e046ff15f6eb0a4e2fffdee279.
R116 failed and its recipient gate stays closed. WP2_DESIGN.md is frozen in
this same protocol commit, not authorized to run. No earlier scientific,
product, human, hardware, throughput or release gates change.

**Question.** Can a new *experimental*, subject-bound bidirectional ABI filter
quarantine unsupported English without discarding valid teacher paraphrases?
Not a claim about the shipped normalizer, general semantics or neural transfer.
Content authority is an authored fact table; the teacher supplies wording only.
No teacher-generated semantic labels, recipients, training or host changes.

**Population.** Eight held-out office milestone constructions, six fresh entity
pairs per construction. Withhold all eight from normalizer implementation/tests:
freeze the method first, then author the evaluation phrase bank (no subsequent
method edits). Public investigator-designed challenges, not secret external gold.
Within each cluster cross one/two named projects with prospective/actual state.
In two-project records the foil has the other state. Each user requests only the
named target milestone. Prospective facts describe permission or a plan, NOT
actual performance; actual facts explicitly attest occurrence. Use deterministic
names/order seeded by the method-freeze commit. Publish all facts, challenge
texts, goals and generator before model inference. No label, family name,
variant index or expected output provided to the filter. All arms receive the
same notes, target, and source candidate. No reconstruction, correction,
substitution, rewriting, or dropping difficult evaluation families.

Per 192 contexts: (1) independently authored valid paraphrase, (2) opposite
status, (3) valid statement plus an unsupported assertion, (4) omitted
milestone with no new milestone claim, (5) off-target entity, and (6) one live
Qwen free paraphrase. In addition two no-notes challenges per cluster (96).
Thus 1,248 judgments/arm, including 192 live texts. Report wrong subject
separately from wrong status; an off-target fact may be true of the foil but
is never a usable target. No-notes candidates are never usable. Raw proposals,
refusals, truncations, omissions and unknowns are all retained in evidence.

**Live stream.** R116's pinned Qwen3-4B-Instruct-2507, BF16, native chat,
greedy, seed 117001, no output constraint or reranking, max 80 new tokens,
max 1024 total context (fail, never truncate input), no teacher self-scoring.
Two fresh processes; exactly 192 initial + 192 replay calls, byte-identical
scientific records. Independent of method decisions, transcribe each live
output into V = supported, on-target and complete; U = new unsupported assertion
(including wrong modality or wrong entity attribution); O = no unsupported
assertion but omission/off-target; Q = genuinely unclear, empty, truncated or
unusable. Require the requested project ID and milestone; extra entailed words
are fine; unsupported dates/timing, completion, reasons, approvals etc. are U.
Scope statements about what the notes say are permissible if true. Label the
entire response, not just a selected span. Annotate every row **before any
method predictions** using notes, task, and text alone. Publish the hash-bound
decisions before method execution. Same investigator, NOT human-independent
gold. Uncertain labels never create a pass; publish raw material for rerating.

**Arms (same rows, same target tags, no tuning).** (a) historical R99 raw
summary pass-through; (b) lexical screen, nonempty <=80 words, exact target ID
and no other project IDs; (c) ordinary one-way entailment: lexical screen AND
strict argmax entailment of candidate from all notes; (d) experimental ABI:
same lexical screen, bind the requested source record, require strict argmax
entailment in both directions between that single fact and the whole candidate.
If no bound fact, quarantine. Any tie or non-entailment fails closed. No method
may use panel gold, names as keys to answers, or the teacher for judging. Use
R105's pinned local DeBERTa NLI checkpoint as the *same fallible component*
for c/d, GPU FP32, batch <=16, max512 input, no truncation. R105's external NLI
failure remains in force. Accepted outputs are original UTF-8 source text, not
model-generated verdicts. Emit per-row reason, byte digest and source identity.

**Decision (new WP1 thresholds; previous gates not redefined).** On the authored
panel a cluster must retain all four valid paraphrases and reject every invalid
and omitted candidate in those cells. Require >=44/48 complete clusters and
>=5/6 in EACH family; zero accepted opposite-status/mixed/off-target/omission
rows, and 0/96 accepted on no-notes controls. On the live stream require >=144
of 192 usable V targets retained overall; retain >=90% of V overall, >=80% of
V within each family, with at least 3 V proposals per six in every family x
entity-count x status cell. Zero accepted live U/O/Q; an empty V denominator
cannot pass. Report actual contamination, coverage, rejection of V, withholding,
source omission rates, per-family and factorial counts, and all denominators.
Do not infer a live corruption rejection rate if the teacher made no errors of
that type. Report 95% stratified percentile intervals and paired differences
vs controls for valid retention and invalid acceptance (10,000 cluster draws,
seed 117001, six clusters/family); replay is not new independent data. Absolute
pass is distinct from relative advantage: claim superiority over the one-way
control only if WP1 passes, lower CI for retention delta >= -0.05 and upper CI
for invalid acceptance delta <0 on the authored set. No threshold/label repair.
On failure no training corpus is qualified, even if some rows look good.

**Custody and bounds.** Publish in order: registration+WP2 design, tested
method freeze, fixed inputs/driver, frozen live annotation, decision. Record
every logit triplet and raw candidate/token IDs, both live executions, source
hashes, costs; repeat classification in a separate process with exact replay.
Two teacher workers <=1800s each; two NLI workers <=900s each; <=3*1248 ordered
NLI pairs per worker; no new downloads, 10 GiB free, <=1 GiB evidence, no GPU
settings changes. One attempt per registered worker. Stop an interrupted
stage, preserve failures, no automatic retries. Verify actual data, mutate
evidence to test rejection, publish recomputable packet. No WP2 execution.
