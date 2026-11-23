# R119: does a change of perspective preserve who did what?

**R119_NOT_QUALIFIED.** On the prespecified 48-cluster test, R119 passed
18 clusters (44 required), the same as R118 status-only. It admitted no
authored actor swaps, but complete retention plus swap rejection in *all
three forms on the same note* fell from 206/288 to 202/288. WP2 remains
closed. No training corpus, LayerCake test, or superiority claim.

| Form (288 matched pairs each) | R118 status-only: valid / swaps admitted / pairs correct | R119: valid / swaps admitted / pairs correct |
| --- | ---: | ---: |
| Active | 252 / 0 / 252 | 250 / 0 / 250 |
| Passive | 276 / 0 / 276 | 276 / 0 / 276 |
| Converse | 208 / 8 / 206 | 204 / 0 / 204 |

All eight extra swap rejections came from the rental family. R119 rejected
**all 36 valid rental converses** as well. It did not demonstrate that it
could tell which person was the renter in that family. The 84 valid converse
rejections split into 36 rental cases rejected by the alias-based entailment
check and 48 cases with unknown status (12 each in sale, succession, duel
and repayment). For example, *Owen bought the easel from Ada* gives the
correct direction for *Ada sold the easel to Owen*, but the inherited status
parser did not recognize the former as occurred. In the repayment example
it was the **note's** “repaid”, not the paraphrase's “received”, that went unrecognized.

| Same-input arm | Valid authored / 864 | Authored invalid admitted / 2,688 | All six voice judgments correct / 288 | Complete clusters / 48 | Live valid retained / 209 | Live unusable admitted / 79 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary one-way entailment | 706 | 608 | 148 | 0 | 70 | 11 |
| Frozen R117 | 820 | 435 | 242 | 0 | 59 | 7 |
| Frozen R118 status-only (primary control) | 736 | 8 | 206 | 18 | 56 | 0 |
| Frozen R118 full role guard | 204 | 0 | 12 | 0 | 18 | 0 |
| Alias-only (no swap preference) | 730 | 0 | 202 | 18 | 40 | 0 |
| R119 (aliases plus swap preference) | 730 | 0 | 202 | 18 | 40 | 0 |

The explicit swap preference changed **no acceptance decision** on this
panel. The simpler alias-only control produced the same result. Comparing
consistent aliasing and a two-assignment consensus with R118 is a compound
intervention; this study cannot attribute the difference to names alone.
R119 passed all six clusters only in loan, order and tutoring. In the other
five families it passed none; the floor was 5/6 in every family. The 96
no-notes controls were all withheld.

Against status-only, valid authored retention fell 0.69 percentage points
(paired 95% interval −1.39 to 0), and invalid admission fell 0.30 points
(−0.74 to 0). Both intervals include zero. Bootstrap draws resample six
clusters within each of eight families, not individual sentences or GPU
replays. See the [machine-readable decision](../../results/voice_converse_r119/decision.json)
for denominators, by-family, per-status and live results. These are bounded
comparisons within a public template panel, not broad English or host claims.

## The matched test

Each of 288 contexts has the **same** fact expressed in three ways: a
subject-first active sentence, a passive sentence, and a converse using a
different predicate and the other participant's perspective. Each has its
own false twin obtained by exchanging just the names; the word multiset is
identical. A matched success means accepting the valid sentence **and**
rejecting its false twin. The all-six score requires doing that in all three
forms in the same context. Merely withholding answers cannot earn either score.

In the lending family, for instance, the atlas being lent by A to B and B
borrowing the atlas from A describe the same loan. B lending it to A does
not. Nonactual notes authorize or plan the specific event; they do not imply
it happened or that a different person has independent permission. The other
families are sale, rental, parade order, stage duel, direct succession, full
debt repayment and tutoring. The task assumes ordinary senses, named people
and a single specified event. These are new stimuli, not eight newly
discovered capabilities; some predicates, notably lending, occurred in
earlier development. Names, surface wrappers and finite constructions recur.

[Preregistration](../../experiments/voice_converse_r119/PROTOCOL.md) at
`ebb071a`; [tested method](../../experiments/voice_converse_r119/method.py)
at `7f5a6f8` before writing the evaluation bank; [actual prospective packet](../../results/voice_converse_r119/packet.json)
and driver at `7563663` before generation. The primary control is the **exact
R118 status-only function**, rerun on the R119 rows, not its old score from
different prompts. R119 retains its lexical and modality requirements. In
place of the original-name equivalence check, it compares the bound note to
the candidate in both directions under two name assignments. The primary
method additionally requires both aligned readings to outrank their swapped
counterparts under each assignment. The alias-only control omits that last
step. Internal aliases are never returned as accepted source text. The
fallible NLI model supplies scores, not ground truth.

There are 48 clusters, not 288 independent language constructions. The
2,688 authored invalids include 864 pure actor swaps, 576 single status
changes, 288 joint changes, 288 mixed assertions, 288 omissions, 288 wrong
projects and 96 no-notes controls. There are 864 authored positives and 288
live responses, for 3,840 decisions per arm.

## Free generation: two distinct limits

The [live annotations](../../results/voice_converse_r119/gold.json) were sealed
at `58ecc5d` before classification. An investigator reviewed 133 distinct
name/ID-normalized text groups covering all 288 full answers. The teacher
supplied wording, not labels. There were **209 valid**, **52 unsupported**,
**26 off-target** (true but unrequested second-project material), and **one
truncated** answer. The valid supply by source status was 90/96 authorized,
89/96 planned and only 30/96 occurred. Eleven family/count/status cells
had fewer than three usable responses; every one was an occurred cell.
Occurrence was frequently inflated into authorization. Even perfect
reject-only filtering could not deliver the required 216 usable answers.
Only two valid answers repeated the entire note byte-for-byte, but 183/209
contained the full milestone sentence verbatim with the project-prefix
formatting discounted. Nonverbatim is not a measure of linguistic novelty.

Of the 209 valid responses R119 retained **40** (19.1%; gate 90%), with zero
invalid responses admitted. The original 192-row projections retained just
9 (authorized+occurred) and 35 (planned+occurred), versus 144 required in each.
Of 169 valid answers withheld, the first failing check was aligned entailment
for 163 and unknown status for six. As a descriptive audit, **135** valid
answers were literally the source note prefixed with the single word
“Project ”. Only **30** passed: 104 failed entailment and one failed status.
This points to a narrower next test than more role training: compare checking
the project ID separately from the milestone text against the unchanged
control, using fresh prefix-only positives, wrong-ID negatives, and hard
rental-direction cases. This is a proposal, not a new run or relaxed gate.

## Recompute from evidence

Both fresh-process replays reproduced the scientific records byte-for-byte.
Per teacher execution: 288 sequences, 26,766 input and 7,943 generated tokens
(including stop tokens where present), 27,246 decoded UTF-8 bytes and
8,363,919,360 peak allocated CUDA bytes. Per classifier execution: 42,240
ordered pairs, 1,732,612 input tokens, 960,398,848 peak allocated CUDA bytes.
Wall times including launch were 97.36/98.25 s for generation and
121.45/124.26 s for classification. The classifier run also overlapped
ordinary CPU regression checks; these times are costs, not throughput claims.
No weights trained or exported, new model download, GPU-setting change or
LayerCake operation. R118's evidence and the frozen WP2 design are unchanged.

Use `python scripts/review_r119.py` from the research tag or its small evidence
ZIP (Python and NumPy; reference NumPy 2.2.6). It binds the raw logs and
receipts, re-derives decisions and intervals, and independently tallies the
positive/swap pair counts. No weights, Git history, GPU or host are needed.
With the original offline tokenizers, `--tokenizers` additionally checks every
model input and the decoded teacher text. Recomputing is not independent
model execution or validation of the investigator's English judgments.
