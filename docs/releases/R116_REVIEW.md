# R116: the source distinguishes missing notes from missing evidence imperfectly

**SOURCE_PROBE_NOT_QUALIFIED.** A blank record was easy: Qwen correctly withheld
support on 96/96 questions with the notes removed. The harder case is a record
with plausible *but insufficient* evidence. The control endorsed 20/96 such
claims when a different project had actually reached the milestone. It also
missed 2/96 supported claims. Recognition on this interface failed the published
source gate. It is not a failure of ABI extraction or LayerCake execution.

## What was actually measured

[Preregistration](../../experiments/progress_grounding_r116/PROTOCOL.md),
commit `6967263`; tested implementation `4cebd21`; [actual inputs](../../results/progress_grounding_r116/packet.json)
published at `5e43f79` before scoring. In each of 48 clusters the same two
names and claim text are used twice: the actual event and the anticipated
event exchange projects. Two queries per world, plus both with notes removed.
For example, a first shift *on the calendar* is not evidence that the crew
*has started*. Gold is assigned by construction, not by a teacher judge.

| Milestone family | Clusters with all four judgments correct / 6 |
| --- | ---: |
| Planned → started | 4 |
| Assigned → completed | 6 |
| Permitted → resumed | **0** |
| Prepared → delivered | 6 |
| Booked → inspected | 2 |
| Drafted → issued | 6 |
| Due → received | 3 |
| Scheduled → actually started on time | 6 |

**33/48** complete clusters, below 44/48 and below the 5/6 floor in four
families. 170/192 evidence-present questions correct; 96/96 withheld correct;
266/288 overall. No ties. The unrestricted top token agreed with the Yes/No
comparison on all 288 rows, so the misses did not come from discarding a third
token with a higher score. These are first-token judgments, not prose generation.
Eleven of the 20 false endorsements concerned permission to resume. This study
does not distinguish misunderstanding of permission from confusion about which
project the claim refers to: both factors were present. There were only six
clusters per family and the templates were disclosed. The four passing families
are not a post hoc qualification cohort.

The second process reproduced every byte of the 288-row initial log, including
the selected logits and unrestricted top token. There were **576 single-row
GPU forwards**, not 576 independent questions: 30,736 input tokens per worker,
61,472 total, zero generated continuations. Worker wall time 39.922 s and
40.032 s; CPU process time 35.266 s and 35.844 s; peak CUDA allocation
8,153,488,384 bytes in each worker. These are probe costs, not LayerCake
inference benchmarks. No training, weight exports, new downloads, hardware
setting changes or recipient inference. R115 was neither changed nor rescored.

## Separate the research questions

- **Source:** this pinned Qwen3 4B native interface is not qualified on this
  milestone-entailment probe. This does not establish that the weights lack
  English fluency. R115's positive sentence-format result stands.
- **ABI:** the registered stop closed the recipient comparison. The frozen
  R99 ABI-plan, LoRA and sequence-distillation files were identity-checked,
  not evaluated. They saw the same source rows, order, 192 training examples
  and 384 historical updates; loss-bearing tokens were 125,316 / 43,208 /
  43,208, respectively. R115's Qwen3 is a control, not the R99 training
  teacher. There is no new teacher-transfer or superiority claim.
- **LayerCake:** no external English package qualified for installation. The
  separate host's package tests and sealed benchmarks cannot supply a missing
  ABI quality result. Human evaluation and external hardware remain open.

**Next scientific decision:** separate entity binding from interpreting modality
before investing in another learner. A fresh design can cross one vs two named
projects with permission/plan vs witnessed action, using independent constructed
gold, not Qwen's opinion. For a transfer test, treat the supplied facts as the
content authority and the teacher as a *candidate source of English wording*.
Test whether normalization discards unsupported assertions **and retains useful
paraphrases** on fresh fact bundles; measure rejected records and retained
coverage before opening a new, matched ABI/LoRA/distillation comparison. Hold
out construction families, not just names. Do not silently relax R116 or filter
the evaluation set for teacher-correct examples. No successor was launched.

## Verify the negative result

The [research release](https://github.com/Yoder23/abi/releases/tag/abi-research-r116-2026-11-23)
supplies the nine-file evidence ZIP and standalone reviewer. Download both into
one directory and run `python review_r116.py --archive r116_evidence.zip` to
audit the logged result without a checkout, model weights, GPU or Python packages.
The [decision](../../results/progress_grounding_r116/decision.json) binds the full
packet, two raw logs, launch and process receipts. From the tagged checkout,
the first command needs only Python's standard library and committed evidence:

```powershell
python scripts/review_r116.py
python scripts/review_r116.py --archive r116_evidence.zip
```

These checks establish public byte custody and reproduce the classifications
and counters, not live model inference. The source-log digest is
`4f9a87b52654b868c86fbe61852064bb8d4089ffc22bf79e81e606f3f0bbee07`;
decision digest `d42f8c8b71bf8fde4891641ea2374a30289073e046ff15f6eb0a4e2fffdee279`;
ZIP digest `a64b3704ec436bbea911f6be1d791181cf153b74b828c8ad73e96a820e0edabf`.
The stdlib reviewer re-derives gold from the subject and the rendered note text;
its independent *implementation* is not independent human validation.
Six actual-evidence corruptions (missing/extra rows or files, order, logits,
decision, receipt) were rejected. All 18 focused software tests passed.
The local supported suite at implementation freeze passed 1,056 tests with
four existing skips; the separate exact historical-host fixture passed all 16.
The 18 focused post-run checks include the subsequently added public-log tests.
All 942 files in the predecessor R115 manifest still matched after the study.

Full local verification additionally re-tokenizes inputs and checks exact
software, source and recipient weights, Git/LFS payloads and sibling host bytes:
`python -m experiments.progress_grounding_r116.run verify`. Use ABI's exact tag
with its prereg/freeze/inputs tags fetched, the registered offline source cache,
and a sibling checkout named `layercake_release` at
`46a8c27eff00bd09280e630b3479fe6d8671a712` (not a later docs commit).
Missing prerequisites fail closed. Do not run the write-once workers over this
evidence. This is not an independent-hardware reproduction.
