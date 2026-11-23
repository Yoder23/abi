# R117: can ABI retain a paraphrase without accepting a new claim?

**WP1_FAIL.** This experimental filter retained all 192 authored paraphrases.
It also admitted 32 of 192 changes of milestone status. A project record that
authorizes a clerk to post a notice does not establish that the notice was
posted. Both entailment directions nevertheless endorsed that substitution in
all 12 relevant authored cases. No training corpus qualified. **WP2 is closed.**

## Design and custody

[Registration](../../experiments/grounded_normalization_r117/PROTOCOL.md) and
[conditional WP2 design](../../experiments/grounded_normalization_r117/WP2_DESIGN.md)
were published together at `dd7884d`. The tested method was tagged at
`dd56e03` before writing the eight evaluation constructions. The actual notes,
prompts, challenges and scoring driver were published at `dd8ef94` before
inference. A pinned Qwen model supplied wording, **not truth**. The investigator
read all 192 outputs and froze [the per-response judgments](../../results/grounded_normalization_r117/gold.json)
at `8353d6f` before any classifier predictions. No independent human labels.
The live rubric requires the named subject, milestone, participants and
prospective/actual status, without new assertions; uncertain outputs cannot pass.

There were 48 clusters (six nonce project pairs per wording family), four
contexts per cluster, 960 authored proposals, 96 no-notes controls and 192
live free responses. All four filters saw the same inputs. Repetition in a
fresh process tests reproducibility; it does not double the sample size.
Accepted text is unchanged source text; no repair, rewriting or selection
of easy families. Git contains the complete raw evidence, not links to private weights.

| Filter | Authored valid retained / 192 | Authored invalid accepted / 864¹ | Live valid retained / 148 | Live invalid or unclear accepted / 44 |
| --- | ---: | ---: | ---: | ---: |
| R99 summary pass-through | 192 | 864 | 148 | 44 |
| Project-ID / length screen | 192 | 672 | 148 | 44 |
| Ordinary one-way entailment | 192 | 266 | 145 | 40 |
| Experimental ABI subject binding + bidirectional entailment | 192 | 32 | 145 | 8 |

¹ Includes 96 no-notes controls. The experimental ABI filter admitted **0/192**
mixed claims, **0/192** omitted milestones, **0/192** wrong-project proposals
and **0/96** no-notes assertions. All 32 authored false admissions were
status changes: archive **12** (actual → intention), posting **12** (permission
→ actual), dispatch **8** (actual → clearance). Events and grants of permission
are not interchangeable in either direction.

The lexical control alone rejected all wrong-project challenges: those
sentences contained the other project ID. This is not evidence of semantic
entity resolution under pronouns or ambiguous references.

Only **32/48** clusters were completely correct (44 required). By family:
archive 0/6, briefing 6/6, dispatch 2/6, filing 6/6, handoff 6/6, inspection
6/6, interview 6/6, posting 0/6. The gate required at least 5/6 in **each**.
Relative to ordinary entailment, the authored invalid-admission rate fell by
27.1 percentage points (paired 95% cluster-bootstrap interval: −28.1 to −25.9
points for ABI minus control); authored retention difference 0 [0, 0] points.
This is a measured improvement on a disclosed finite panel, **not a registered
superiority pass**. The comparison changed binding and direction together and
cannot assign the improvement to either component alone.

## Free generation: coverage is not cleanliness

The source supplied 148 complete, 30 unsupported, nine incomplete and five
unclear answers. ABI admitted 145 complete, four unsupported, one incomplete
and three unclear. It withheld 39/192. Among accepted outputs, **8/153** were
unusable; among valid outputs, **3/148** were lost. Live valid retention was
98.0% (95% interval 96.5–99.4%); invalid acceptance was 18.2% (9.1–29.4%).
Of the 148 valid responses, 77 repeated the target sentence verbatim; ABI kept
all 77 and **68/71** valid nonverbatim responses. These are not 145 demonstrations
of novel wording. The aggregate cleared the minimum of 144 usable responses,
but four cells had fewer than three valid proposals out of six: dispatch /
one project / actual (0); handoff / one / prospective (0); inspection / two /
actual (2); posting / one / prospective (1). Reject-only filtering cannot
fix an inadequate supply of valid language.

[Machine-readable decision](../../results/grounded_normalization_r117/decision.json):
all arms, failure reasons, eight families, 32 factorial cells, counts and
paired intervals. Bootstrap: 10,000 draws of six clusters within each family;
no inference to unseen constructions or languages. In the authored breakdown,
off-target and omission are separate challenge types; a wrong-project sentence
can be true of another project but is never a valid answer for the target.

## Cost and review

No training, settings changes, downloads of new model weights, LayerCake
inference, LoRA or distillation runs. Per teacher execution: 192 greedy
sequences, 14,274 input tokens, 3,319 output tokens including stop tokens,
9,422 decoded output bytes, 8,291,458,048 peak CUDA bytes. Per classifier
execution: 3,744 ordered pairs, 123,831 input tokens, 873,675,264 peak CUDA
bytes. Initial/replay wall times including process launch: teacher 44.19/44.19 s;
classifier 17.41/16.91 s. All four ran on the same laptop GPU. Raw teacher
records and raw classifier records, including logits, each replayed byte for
byte. Receipts include peaks for process RSS, CPU time and model identities.
These are preparation/filtering costs, not host speed measurements.

From the research tag or its small evidence ZIP, with Python and NumPy
(reference version 2.2.6), run:

```console
python scripts/review_r117.py
```

No GPU, weights, sibling repository or Git history needed for that audit.
It checks the decision digest, both replay pairs and label bindings, then uses
the raw logits to recalculate counts, intervals and gates. It cannot establish
that the investigator's semantic judgments were correct. Optional `--tokenizers` checks every recorded input and
decoded output with the original local tokenizer caches. This is evidence
recomputation, not fresh model execution. For live execution the original
immutable model inventories and registered budgets are required. Do not launch
over the sealed output directory.

The failure belongs to this ABI normalization procedure; live coverage also
exposes a source limitation. It is not a LayerCake or transfer result.
[R116](R116_REVIEW.md) remains unchanged. A useful next *proposal* is to test
an explicit representation of who may do what versus what actually happened
against new paraphrases and role swaps; do not tune a keyword list on this
published panel. No successor is authorized by this result. The English-core
and whole-moonshot claims remain unproved.
