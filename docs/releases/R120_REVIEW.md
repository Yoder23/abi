# R120: check the project identifier separately from the claim

**The preregistered prefix test passed; overall verdict: `R120_NOT_QUALIFIED`.**
With the same ID check, scoring the event without its header recovered 396
more valid prefix-only restatements. It retained 552/576 rather than 156/576:
+68.75 percentage points (paired 95% interval +64.93 to +72.57). Both arms
rejected all 576 wrong-ID twins. This is evidence for removing ID text from
the semantic comparison, not evidence that adding the ID check caught new
errors. Only 6/48 complete clusters passed (44 required; only loan cleared
5/6 for its family). WP2 remains **CLOSED**. No data released for training,
no LayerCake recipient run, no weights updated, no superiority claim.

| On the same new inputs | “Project” positives / 288 | “Project ID:” positives / 288 | Wrong-ID twins admitted / 576 | Rentals: valid / reversed admitted / both correct (144 pairs) | Valid live retained / 228 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R118 status-only | 82 | 136 | 0 | 54 / 20 / 50 | 71 |
| R119 alias-only | 76 | 80 | 0 | 24 / 0 / 24 | 69 |
| R119 full | 76 | 80 | 0 | 24 / 0 / 24 | 69 |
| Strict ID + raw text | 76 | 80 | 0 | 24 / 0 / 24 | 69 |
| Strict ID + event text (R120) | 276 | 276 | 0 | 36 / 0 / 36 | 204 |

For R120, each header had 276/288 **both-correct** positive/negative pairs.
Every arm kept 276/288 exact bare notes. The 24 valid prefix losses were
the two header variants of 12 reported subleases whose event status the
unchanged guard could not classify. R120 admitted 0/288 missing IDs, 0/288
conflicting IDs and 0/96 no-notes controls; across all authored cases it
kept 1,464/1,728 valid and 0/3,840 invalid.

**Rentals are not solved.** R120 kept 24/36 valid sublease and 12/36 keyboard
converses, but 0/36 marina and 0/36 tiller converses. Of 108 valid rental
converses withheld, 96 failed alias-based entailment and 12 failed status
recognition. The paired gain over strict-ID/raw was +8.33 points, but even
status-only got more rental pairs correct (50), while admitting 20 reversals.
The registered within-construction bootstrap gives [+8.33, +8.33] points:
the gain did not vary across the six project pairs within each construction.
That interval cannot measure uncertainty on new constructions. Across all
eight families, taking off the header gained 12 converse pairs, lost 14
active pairs, and left passive pairs unchanged versus strict-ID/raw. Contexts
correct on all three voice pairs fell from 122/288 to 120/288. There is no
general semantic gain independent of the prefix result.

## What the comparison isolates

R119 admitted only 30 of 135 valid live responses that differed from the
source solely by adding “Project ” before its identifier. R120 does not
rescore those examples as fresh evidence. It compares three filters on new
inputs: frozen R119; R119 plus an explicit header check but still scoring
the raw sentence; and that same check followed by scoring the two milestone
bodies. The latter two arms differ **only** in which text the entailment and
role-swap model sees. The delimiter and binding are checked on original
bytes; the accepted payload, if any, is still the untouched full candidate.
No equality shortcut: even a byte-identical body must pass the frozen
modality, participant, entailment and role-contrast checks.

[Registration](../../experiments/project_binding_r120/PROTOCOL.md) at
`c9255d3`; [tested method](../../experiments/project_binding_r120/method.py)
at `3a3e46d` before writing the new evaluation bank. The [actual input
packet](../../results/project_binding_r120/packet.json) and driver at
`875017b` were published before inference. The finite parser recognizes
explicit ASCII identifiers and supported leading labels, including
“Project ID:” and “For project …,”. It does not infer missing IDs or support
arbitrary English references. Repeated/conflicting canonical identifiers
in the body fail closed. This is not a general parser-security claim.

Each of 288 contexts has an exact-note control and two prefix-only positives.
Their wrong-ID twins change **only** the identifier: to the other project's
ID for the “Project ” prefix, or by one hex digit for “Project ID:”.
There are also 288 missing-ID and 288 conflicting-ID negatives. All original
active/passive/converse pairs, wrong-status, joint-swap, mixed, omission,
off-target and no-notes challenges are retained on fresh stimuli. Four
families concern a keyboard rental, marina lease, studio sublease and
tiller hire. Four non-rental anchors concern sale, loan, gift and an
exhibition match. There are six project pairs per construction, not eight
independent corpora. Each uses both note counts and all three event states.
A rental pair is correct only if the filter accepts the valid change of
perspective **and** rejects the otherwise identical name reversal.

One actual rental item gives the source “P66878366: Theo leased the marina
slip to Ana.” The valid converse is “For P66878366, Ana leased the marina
slip from Theo.” Its negative is “For P66878366, Theo leased the marina
slip from Ana.” The ID and the words *leased … from* are identical in the
two alternatives; the person who got the slip is not. Rejecting both earns
zero on this pair.

## Source quality before filtering

The teacher and fresh replay produced the same 288 outputs, byte for byte.
Labels for all 120 distinct normalized response groups were sealed at
`1c88422` before classifier execution (gold SHA-256
`6e2e07a446cce903c2c80c33b62cd36b34306643703c7e0656b68f2151c716f6`).
The prediction-blind investigator classified 228 as valid, 41 as unsupported
and 19 as off-target; none were unresolved. These are not human ratings.
The source alone missed the minimum of three valid answers out of six in
six cells, all for reported events: marina/two notes (2), match/two (1),
sale/two (0), sublease/two (1), and tiller/one and two (1 each). A filter
that only keeps or withholds the original text cannot remedy those deficits.

**After filtering**, R120 kept 204/228 valid live answers (89.47%, below
90%; 204/288 usable versus 216 required) and admitted 0/60 unusable answers.
Of the 24 valid answers withheld, 15 failed entailment, seven had an
unrecognized status, and two used an unrecognized header. The sublease family
retained 24/31 (77.42%, below 80%). The authorized/occurred and
planned/occurred projections retained 112/192 and 116/192 usable answers
respectively (144 required in each). All 288 live outputs differed from the
bare note; a prefix alone qualifies as a difference, not a novel paraphrase.
The [decision](../../results/project_binding_r120/decision.json) gives all
five arms, denominators, intervals and failed gates. R119's historical 18/48
is a different panel; its same-input control is the R119 row above.

**Next question, not another run:** do the same frozen source and filter
distinguish *leasing to* from *leasing from* when both participants, the ID
and event status are explicit? Test sender/recipient roles against matched
same-word reversals on new rental wording, while tracking any loss on valid
active sentences. Separately measure whether the teacher adds permission
to reported events, especially with a distractor note. Keep WP2 closed.

## Reproduction

From the sealed tag or its standalone ZIP, run
`python scripts/review_r120.py --paired` with Python and NumPy (reference
2.2.6). It recomputes from raw tokens, logits and sealed annotations; no
weights, host, Git history or GPU needed. Optional `--tokenizers` also checks
every ordered model input and decoded teacher output using the original caches.
Recomputing a score is not independent human validation or hardware replication.

Both generation and classification ran twice from separate processes with
byte-identical raw evidence in each pair. Per teacher execution: 27,066
input tokens, 7,860 generated tokens, 26,413 decoded UTF-8 bytes and
8,366,756,352 peak allocated CUDA bytes. Per classifier execution: 111,264
logical ordered pairs (3,819,287 tokens), **33,830 actually evaluated**
(1,424,274 tokens; 2,248 forward batches; 953,750,528 peak allocated CUDA
bytes). Identical pair reuse did not cross processes. Wall times including
launch were 144.23/98.03 s for generation and 107.19/107.11 s for
classification. Regression tests shared the CPU; these are costs, not speed
benchmarks. GPU settings, R119 evidence and the WP2 design were unchanged.
Decision SHA-256: `d81387e9a726e4ee93b7192cf89dd1b4d562aa1895554875c81cf4868136d427`.
