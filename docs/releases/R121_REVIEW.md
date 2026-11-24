# R121: could a perfect selector obtain enough faithful English from this teacher?

**No. Registered verdict: `STOP_SOURCE_RECIPE`.** Direct restatement met
the overall floor but failed the distribution required for usable source
data. Asking for an intermediate record was worse at the same output budget.
No filter, corpus, student or LayerCake evaluation was authorized by this run.

| Fresh paired source measure | Needed | Direct | Record first |
| --- | ---: | ---: | ---: |
| Valid final English / 288 | ≥216 | 235 | 0 |
| Valid authorized / planned / occurred (96 each) | see cells | 91 / 92 / 52 | 0 / 0 / 0 |
| Family × note-count × status cells with ≥3/6 valid | 48/48 | 40/48 | 0/48 |
| Valid authorized + occurred / 192 | ≥144 | 143 | 0 |
| Valid planned + occurred / 192 | ≥144 | 144 | 0 |
| Unsupported / omission or off-target / ambiguous | descriptive | 40 / 13 / 0 | 85 / 197 / 6 |
| Exact or allowed prefix-only copies of the source note | descriptive | 187 | 0 |

Seven direct-arm cells had only 2/6 valid, and loan/two notes/occurred
had 1/6. All eight concerned occurred events. Meeting even the cell floor
would require **at least nine more valid answers in those particular cells**,
not just one extra answer to lift the 143 window to 144. Thirty-eight of the
40 direct unsupported answers concerned occurred events. At least 187/235
valid direct outputs were copies under the narrow registered rule.

All 288 record-first answers supplied a parseable final and stopped normally.
Every one put the exact target identifier in its workspace but omitted it
from its final English. That explains the zero exact-task score; it does
**not** mean all 288 English sentences were false or the model lost the ID.
Nor would restoring the ID solve the problem: 85 finals made unsupported
assertions, 71 on occurred cases. Even crediting every other output would
give at most 203/288, below 216. This arithmetic is diagnostic; no outputs
were spliced, repaired or rescored. Neither arm had a truncation.

Record-first minus direct yield was **−81.60 percentage points** (paired
within-construction 95% interval −85.07 to −78.13; 0 gained, 235 lost).
On occurred cases it was −54.17 points [−63.54, −45.83]. These results
apply to this source, prompt design, fixed constructions and final-answer
contract; they do not falsify all structured acquisition or general English.
The [sealed decision](../../results/source_ceiling_r121/decision.json)
contains every cell and gate. **Stop this recipe.** A materially different
source or independently validated source of supervision would need a new
proposal before any downstream work; there is no automatic R122.

This is an ABI **source-supply** experiment. R120's valid answers numbered
135 and 139 in the two required 192-answer projections; even perfect selection
would have missed the 144-answer floor in both. R121 asks a different,
upstream question on fresh notes. It does not rerun or change the R120 filter.

**Design.** The [registration](../../experiments/source_ceiling_r121/PROTOCOL.md)
was published at `057211d`. The [actual packet](../../results/source_ceiling_r121/packet.json),
parser, generation code and scorer were frozen at `8949383` before inference.
The direct arm uses the R120 instruction. The structured arm first identifies
the requested project, directional participants, event and status, then writes
a final English restatement. Neither arm receives a completed record, target
answer, multiple attempts, an extra call, or a larger output allowance. The
same eight constructions, six project pairs, two note counts and three event
statuses give 288 matched contexts per arm. These are new cases in known
task families, not novel English tasks or independent human judgments.
The teacher was the same offline `Qwen/Qwen3-4B-Instruct-2507` snapshot
(`cdbee75f17c01a7cc42f958dc650907174af0554`), in BF16 with native greedy
generation. Equal ceilings do not imply equal token counts: the extra
instruction costs input tokens and the intermediate record costs output tokens.

**What counts.** Only the delivered English is judged. Missing output format
or truncation cannot be silently repaired. A valid response must preserve the
requested ID, both people, the direction and event status, without unsupported
or off-target content. A word-for-word copy of a supplied note can pass; this
experiment tests fidelity, not autonomous paraphrase diversity. The transparent
copy rule covers the exact note alone or after “Project ” or “Project ID ”;
other instances with the same meaning are reviewed individually by normalized
group. Copy counts are therefore a lower bound on broadly copy-like answers.
For the manual review the display contains only anonymized notes and the
candidate final; the arm, intermediate fields and outcome totals are hidden.
Stylistic differences can still give clues. No human rating is claimed.
All 227 nonautomatic group judgments and the initial scientific rows were
published at `3ff9b47`, before arm totals were computed. Correct meaning
with a missing project ID is an omission; asserting that a participant
personally gave authorization when the note only describes an authorized
arrangement is unsupported. Exact copies are counted mechanically for both
arms under the same rule.

A perfect selector is a counterfactual upper bound: it retains every valid
answer and no invalid answer. There is no such selector deployed here.
The 216/288 overall, 3/6 per cell, and both 144/192 projection floors are
unchanged. Nonempty-family retention and zero contamination are ceiling
properties, not a test of the real filter. The eight fixed construction
families are not sampled from general English. Replays do not increase N.

**Reproduction.** From the R121 tag or standalone evidence ZIP run
`python scripts/review_r121.py` (Python and NumPy; reference NumPy 2.2.6).
The reader checks both worker receipts and input identities, recreates blind
group assignments, reapplies all votes, recomputes gates and paired intervals,
and independently computes both oracle-window counts from the 48 cells.
From the full tagged checkout with original offline tokenizers, add
`--tokenizers` to check exact chat rendering, prompt tokens and decoded
generation. Recomputing judgments is not independent validation of their
semantic correctness. WP2 is closed; no filter, training or recipient run.

The [label freeze](https://github.com/Yoder23/abi/tree/4ba4d2d/results/source_ceiling_r121)
bound judgments to both byte-identical executions before scoring. For the
post-score count of IDs present only in the workspace, pass `--diagnostics`.
Each worker made 288 calls/36 batches per arm. Direct used 27,174 prompt
and 7,895 output tokens (26,398 decoded UTF-8 bytes); record-first used
46,182 and 11,263 (37,759 bytes including workspace). The two independent
processes took 253.45 and 252.55 seconds including launch on one RTX 3080
Laptop GPU; peak allocated CUDA memory was 8,440,554,496 bytes on the first
worker. CPU regression tests overlapped; these are costs, not a speed
benchmark or independent hardware reproduction. No settings changed.
Decision SHA-256: `f5e3c4155d1ed1c71a969fa3099cf05a96f7be6b6ef64acda8c59653489bf69e`.
