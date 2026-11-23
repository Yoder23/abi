# R118: permission, plans, events and the people involved

**R118_NOT_QUALIFIED. WP2 remains CLOSED.** The added status guard stopped
all 164 wrong-status claims that frozen R117 admitted on the new panel.
But with both new guards, ABI also discarded half the valid paraphrases.
It did not distinguish the three statuses *and* preserve valid language
reliably enough for a training corpus. No recipient, LayerCake runtime,
LoRA, distillation, or training experiment ran.

## What was frozen before scoring

[Registration](../../experiments/modality_roles_r118/PROTOCOL.md): `de7250c`.
[Text-only guards](../../experiments/modality_roles_r118/method.py): `ca6022e`,
tested and published **before** writing the eight evaluation constructions.
The [actual inputs](../../results/modality_roles_r118/packet.json), evaluation
bank and driver were published at `69b6311` before inference. Qwen supplied
free responses, not labels. All 288 responses were annotated before any filter
predictions and [published](../../results/modality_roles_r118/gold.json) at
`4078a8d`. Review used 164 distinct texts after consistent name/ID substitution
within the same target fact and status. This was investigator review without
model predictions, not independent human gold.

48 clusters (six nonce project pairs in each of eight office scenarios) each
crossed one/two projects and authorized/planned/occurred. Each context had a
new valid paraphrase, two role-preserving status changes, an actor-only swap,
a joint swap, mixed claim, omission, wrong project and a free response.
There were also 96 no-notes controls: **2,688 rows per approach**. The pure
actor swaps preserve the word multiset of the valid paraphrase. The roster
was alphabetical, not labeled with agent/recipient. No method saw expected
roles or status. All additions only quarantine; they cannot repair a sentence.

| Same-input filter | Valid authored retained / 288 | Wrong status admitted / 576 | Pure actor swaps admitted / 288 | Complete clusters / 48 | Live valid retained / 153 | Live unusable admitted / 135 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R99 pass-through | 288 | 576 | 288 | 0 | 153 | 135 |
| ID and length | 288 | 576 | 288 | 0 | 153 | 70 |
| Ordinary one-way entailment | 274 | 269 | 55 | 0 | 95 | 7 |
| Frozen R117 | 276 | 164 | 0 | 6 | 89 | 1 |
| R117 plus status only | 228 | 0 | 0 | 18 | 89 | 1 |
| R117 plus ordered roles only | 180 | 114 | 0 | 0 | 89 | 1 |
| R118: R117 plus both | **144** | **0** | **0** | **6** | **89** | **1** |

R118 admitted none of the 2,112 authored negatives, including all 288 joint
swaps and 96 no-notes controls. R117's six directed status errors were:
authorized → planned **42/96**, authorized → occurred **38/96**;
planned → authorized **12/96**, planned → occurred **36/96**;
occurred → authorized **14/96**, occurred → planned **22/96**.
R118 admitted **0/96 in each direction**. Requiring the *same context* to
retain its valid wording and reject both status foils and the actor swap
gives **48/96 in each true status** for R118, versus 36/96 authorized,
48/96 planned and 62/96 occurred for R117. Zero false admissions alone
would obscure this tradeoff.

Against R117, authored invalid admissions fell 7.77 percentage points
(paired 95% interval −8.24 to −7.29 points); valid retention fell **45.83
points** (interval −45.83 to −45.83). Intervals use 10,000 within-family
cluster draws. Reused constructions can give constant within-family results;
these intervals do not generalize to unseen English. No registered superiority.

## Where coverage was lost

For a valid example, the notes said *Cleo lent the projector to Wade*. The
candidate said *Wade borrowed the projector from Cleo*. The event and direction
were unchanged. R117 accepted it, but the new role parser returned unknown.
For another valid paraphrase, *Wade is the intended borrower under Cleo’s plan
to lend the projector*, the parser reversed the participants.

Of 144 valid authored rejections, the first recorded reason was unknown roles
in 60, unknown status in 48, mismatched roles in 24, and the underlying
entailment check in 12. Ordered roles removed **84** valid cases beyond the
status-only filter without preventing an additional authored or live invalid
acceptance. All six escort clusters passed; no other family had a complete
cluster. The gates required 44/48 overall and at least 5/6 per family. This
tests named roles, not general reference resolution or discovery of semantic roles.

## Live responses: separate source and filter failures

The source supplied **153/288** valid and complete answers, 62 unsupported,
54 omissions and 19 unclear. All 54 omissions lacked the project ID; other
outputs added an ungrounded pending, ongoing or unchanged status. Even a
perfect reject-only filter could not deliver the required **216/288** valid
answers. Seventeen of the 48 source cells also missed the minimum three of six.

Frozen R117 and R118 made **identical acceptance decisions on all 288** live
responses. Both kept 89/153 valid (58.2%; required 90%) and admitted one
ambiguous output: *Gwen introduced Iris at the assembly. Status: P142B2E2F.*
The project identifier was presented as the status value. Neither filter
admitted a live U or O, but an admitted Q also fails the zero-invalid gate.
R118 kept 50 valid in the authorized+occurred projection and 61 in
planned+occurred (144 required in each). Of 109 valid nonverbatim outputs it
kept 45; the other 44 valid outputs were exact note copies and all were kept.
Even nonverbatim is a weak proxy for novelty: 143/153 valid outputs contained
the note's entire milestone sentence verbatim after removing its project prefix.

**Next test worth proposing, not run:** keep status-only as the comparator.
First test whether an ordered-role representation accepts equivalent active,
passive and converse descriptions while rejecting true swaps, at matched
coverage on new events. Separately diagnose unsupported status additions
and missing IDs in the source. More recipient training cannot repair either
input problem. R117 and its WP2 design remain immutable; no gates were lowered.

## Recompute

The [decision](../../results/modality_roles_r118/decision.json) includes every
arm, status direction, family, source cell, paired interval and gate. Per
initial/replay teacher worker: 288 greedy sequences, 24,405 input tokens,
5,829 output tokens including stops, 18,596 UTF-8 output bytes, peak CUDA
8,322,427,392 bytes. Per NLI worker: 8,064 pairs, 307,716 input tokens,
peak CUDA 884,879,872 bytes. Wall seconds including launch: teacher
70.84/72.30; NLI 30.19/31.95. Both fresh-process scientific replays were
byte-identical. Same laptop GPU and settings; no new weights or training.

From the tag or its standalone evidence ZIP, with Python and NumPy (reference
2.2.6), run `python scripts/review_r118.py`. This recomputes from raw logits
and sealed labels; it does not independently validate the semantic judgments.
Optional `--tokenizers` also checks every model input and decoded output using
the original local caches. Do not run the write-once workers over sealed data.
