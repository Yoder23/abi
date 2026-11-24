# R121 — Can the teacher supply enough faithful final English?

Register before generating the packet or any model output. Preserve R120
(d368216; decision d81387e9a726e4ee93b7192cf89dd1b4d562aa1895554875c81cf4868136d427).
Only source supply is tested. No filter, classifier, training, recipient,
LayerCake changes, WP2 opening, downloads, GPU setting changes or retries.
A ceiling pass is necessary, not sufficient for WP1 or an English corpus.

**One paired contrast.** A = the R120 direct free restatement prompt on new
notes; B = identical notes/request plus an instruction to first emit a short
record of project, directional participants, event and status, then a line
starting "FINAL:" and one free English restatement. No oracle fields, examples,
gold prefix, extra model call, external answers, reranking or answer repair.
A is scored whole, B only the final English after exactly one leading FINAL:
line marker following nonempty workspace. Do not require correct intermediate
fields for a valid final answer. A missing/duplicate marker, empty final,
truncation, or missing workspace in B is unusable (Q), never silently dropped.
Report interface failures and token costs separately. This tests the combined
prompt/output interface at equal budget, not private reasoning or fluency.

**Population and freeze.** Fixed before launch: eight related but fresh
office-event constructions (four directed rentals and four non-rental anchors);
six fresh project pairs per construction; one or two notes crossed with
authorized/planned/occurred. Foil always another ID, other state, reversed
direction. 48 clusters, 288 prompts per arm; not 576 independent cases.
Use new IDs/names/assets; check no overlapping prompts/IDs with R120. Freeze
actual stimuli, parser, generation driver, blind rubric and scorer in Git
before inference. Counterbalance arm order across batches. Both arms see the
same notes, same frozen offline Qwen, BF16, greedy native chat, 80 output
tokens (including B's record), context 1024, batch <=8, one call/case.
Two fresh GPU workers each reproduce all 576 sequences. Hard limits: 1800s
per worker including startup, 10 GiB free, 256 MiB evidence. Retain all rows,
tokens, exact prompts, stop reasons, hashes, resources; require byte-identical
raw replay and distinct processes. Stop incomplete on failure, no rescue.

**Semantic rubric.** Judge the English against the requested note only: V
means the event, target ID, both participants, direction and status are complete
and no unsupported or off-target claim appears. U means a wrong/unsupported
assertion (including inferring permission from occurrence); O a true other
project or omitted required information without a false assertion; Q means
unscorable/ambiguous including protocol failures. Semantically neutral ID
wrappers are fine. Under either arm a final exactly equal to the complete
target note (optionally preceded by "Project " or "Project ID ") qualifies
as V. This is a transparent lower bar: report verbatim/prefix-copy counts;
do not call a high copy score paraphrasing. Group other cases by fully
anonymized notes and final text, not by system, with equivalent ID/name
substitution only. Review final text without arm, intermediate fields,
predictions or arm scores; require explicit judgments for every remaining
group. Seal labels before aggregate scoring. Not human or independent gold.

**Stop/go.** The hypothetical perfect keep-only selector keeps every V and
nothing else. For each arm require >=216/288 usable, >=3/6 V in each of 48
family/note-count/status cells, and >=144/192 V in BOTH original
authorized+occurred and planned+occurred windows. Original 90%/80% valid
retention and zero contamination follow from this selector only with nonempty
denominators (also require every family in both windows). Do not claim the
44/48 authored-cluster/semantic-filter gates were tested. Report actual
counts by cell, family and status; format/truncation; cost; paired flips;
percentile 95% CI for B-minus-A valid yield using 10,000 paired
within-family cluster bootstrap draws (seed 121001); separately report
occurred effect. Call an intervention benefit only if B passes and primary
lower bound >0. If neither arm passes, STOP this teacher/prompt recipe;
no further prompt or filter variants on this result. If a ceiling passes,
eligible only to propose a fresh semantic selection test, not training.
No modeled best-of-two pooling. WP2 remains CLOSED in every outcome.
