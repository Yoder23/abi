# R118 registration: distinguish permission, intention and occurrence without losing who acted

2026-11-23. Preserve R117's sealed decision
`e9751c1ba679be18733c4a32bd67dadde4c0ed1e680629d80bf82990af6063eb`,
WP2 design and WP1_FAIL. **WP2 CLOSED for every R118 outcome**. No recipient,
training, new weights, GPU settings or product-gate changes.

**Question.** Does adding explicit equality of modality and ordered participant
roles to frozen R117 entailment reject single-field substitutions on new
wording while retaining valid changes of voice and phrasing? Content authority:
one note for a requested project, not teacher self-rating. Permission is not
a plan; a plan is not permission; neither asserts occurrence. Occurrence
does not itself attest authorization or an explicit plan.

**Order and masking.** Publish this registration first. Freeze and publish the
tested method *before authoring the eight test constructions*. Then freeze
and publish complete stimuli and driver before model calls. Nonce IDs, name
assignments and order seeded by the method commit. All filters see the same
raw notes, project ID, untyped alphabetical two-name roster and candidate.
No family, gold modality, role assignment, challenge ID, answer key or teacher
rating is passed to any filter. No rewriting or selecting learned examples.
Keep every row, including unparsed cases. Single investigator, public synthetic
panel; not a blind external study. No tuning or retry on outcomes.

**Factorial.** Eight new milestone events, six project-pair clusters each.
For each cluster cross 1/2 projects with authorized/planned/occurred (288
contexts). For two projects the foil reverses actor/other participant and
has a different status. Both names occur even in a single-project record;
alternate their roles between cluster instances. Use genuinely different
source and positive candidate wording, including active/passive and
non-actor-first formulations. Per context: one valid paraphrase, each of the
two other states with otherwise identical roles, a pure actor swap with
the same name multiset, one joint state/actor swap, supported+unsupported
mixed claim, target-only omission, and wrong project; plus one live free
response. Add 96 no-notes challenges (two/cluster). 2,688 rows per arm.
Report directed state confusions, actor swaps by state, and valid retention
by construction. Construction gold is not a semantic label for open text.

**Arms.** R99 pass-through, R117 lexical and ordinary one-way NLI controls,
exact frozen R117 bound bidirectional filter; plus state-only, roles-only and
both guards stacked on R117. Each experimental guard independently parses the
bound sentence and candidate. Require equal known state and/or equal known
ordered participants; unknown or ambiguous fails closed. No action-family
dictionary tuned on held-out examples. Retain untouched UTF-8 only. NLI is a
fallible component, not ground truth. Publish every prediction, reason,
parsed slots, ordered model input and logit triplet.

**Live stream.** Same offline pinned Qwen BF16 native chat, greedy, max 80 new
tokens/1024 context; same pinned DeBERTa FP32, strict argmax entailment, max
512, no truncation. Teacher free text without constraints or answer choices.
Annotate V/U/O/Q under the full R117 rubric, including both participants,
event, direction, project and status; ambiguity/truncation cannot qualify.
Publish all labels and their response hashes before *any* filter prediction.
Allow grouping for review only by exact text after role-consistent substitution
of the two names and project IDs, within the same source fact and status.
Blinded to predictions, not independent human gold. Report nonverbatim V too.

**Unchanged gates.** >=44/48 complete clusters and >=5/6 in every family;
now a complete cluster must solve all six contexts and every authored negative.
Zero admitted authored invalids and 0/96 no-notes. Live >=216/288 V retained
(same 75% minimum), >=90% V retention overall, >=80% per family, >=3/6 V in
every family x project-count x status cell, zero admitted live U/O/Q. Undefined
denominators fail. Also enforce the original 144/192 count and retention gates
separately on both authorized+occurred and planned+occurred projections. No
historical gate reopened. Report all arm counts and 95% percentile intervals
with 10,000 paired, within-family cluster bootstrap draws (same R117 seed).
Superiority vs one-way or R117 only if absolute pass, lower bound of authored
valid-retention difference >= -0.05 AND upper bound invalid-admission difference
<0. Ablations descriptive. A smaller invalid count cannot compensate for
missing valid language; report source scarcity separately. No qualified
training data or product claim on failure.

**Budget/custody.** One initial and one fresh-process replay each: 288 teacher
sequences/worker <=1,800 s and at most 8,064 classifier pairs/worker <=900 s
(<=16 per forward). Same unchanged laptop GPU, no downloads. Disk >=10 GiB,
R118 evidence <=1 GiB, no automatic retries; if incomplete preserve prefix.
All workers write once. Raw tokens, scientific replay bytes, model identities,
CPU/GPU resource measurements, hashes, recomputation, hostile tests and
public evidence required. No R117 rescoring as new data. WP2 stays closed.
