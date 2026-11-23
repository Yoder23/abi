# R116: does a progress judgment follow the evidence or the project name?

Preregistered November 23, 2026, before prospective assembly or scoring.
Parent: R115 research commit 907bfb3da67269fbdbc350a70068ad009ea04229;
manifest 4de0dcc3105b9cf2a9d885d489dd0d7cd582125fdbb20c4a8e8ee1c69f1d1f9a.
Do not edit, rerate or replace R115. Its mechanically passing output
r115-012-a called a planned review "underway as scheduled" without evidence.

## Question and fixed endpoint

Is the source sensitive to which project actually reached a milestone, rather
than treating plans, readiness or another project's progress as evidence?
Use explicit supplied-note entailment, not an open-ended prose grader. “No”
means the notes do not establish the claim, not that its negation is true.
Gold follows a prospective world assignment, not a model judgment or a regex
applied to generated prose. This is a diagnostic of recognition, **not**
free generation, a replacement R115 score, or an English certification.

Eight author-disclosed families: planned vs started, assigned vs completed,
permitted vs resumed, prepared vs delivered, booked vs inspected, drafted vs
issued, due vs received, and scheduled vs actually started on time. Six pairs
of new project names per family. For each pair use the same two named projects
and two note worlds: put actual evidence on A and prospective evidence on B,
then exchange statuses without changing subjects, claim wording or note order.
Ask each project's affirmative claim in each world, then both claims with
notes withheld (negative control, gold No). 48 clusters x 6 = 288 probes/arm.
Names/order are deterministically derived from the frozen code commit; all
cases and gold are published before inference. Templates are disclosed;
lexical repetitions and replay are not independent semantic families.

Use Qwen/Qwen3-4B-Instruct-2507 at R115's pinned revision/cache/BF16/native
chat. Score the two single-token answers Yes and No using the same next-token
forward pass. Record both logits and unrestricted vocabulary argmax; ties
fail, not arbitrary yes/no. Do not call two-choice success free generation.
The scorer independently checks all four factorial cells, absent-evidence
controls, exact expected IDs/order, inputs and decision from raw logits.
No answer or row ID is rendered. Primary: clusters with **all four** evidence
judgments correct; separately count per-world accuracy, false support in the
foil cells, missing-evidence Yes and unrestricted top-token compliance.
Source opens recipient comparison only with >=44/48 complete clusters,
>=5/6 each family and 0/96 false support or ties on notes-withheld controls.
Replay all 288 rows in a new process, exact scientific fields including logits.

If source qualifies, run exactly four frozen historical arms: unchanged R76
parent and R99-FP32 ABI-plan, conventional LoRA, sequence distillation. All
recipients share the same plain question body plus a Response newline, identical
Yes/No choices, FP32 and the historical summarization task route (trained task,
not oracle answer). Source uses its own native wrapper, so its absolute margin
is not compared across tokenizers. All learners use the same evaluation interface;
ABI-plan's native plan-generation workflow is not being tested. The historical
R99 learners had matched data/order/steps, not necessarily equal supervised
tokens or FLOPs; the R115 control is not their historical training teacher.
Apply identical diagnostic gates/replay per arm and report all controls even
when a recipient fails quality. Paired ABI-minus-each-control complete-cluster
differences and 95% percentile CIs: 10,000 draws, seed 116001, sampling six
clusters with replacement within each family. Descriptive on these templates;
no superiority or causal acquisition claim from a single historical training seed.

## Execution and stopping

Publish protocol, then tested implementation, then actual input/identity packet
before inference. Separate fresh process per arm/repeat, GPU unchanged, seed
116001, four CPU threads, TF32 off, offline verified cache/checkpoints. No
intervention on power, clocks or other jobs. Record device, peak CUDA allocation,
CPU/wall time, token counts and bound hashes. At least 10 GiB free, at most
1 GiB new evidence, 900 seconds per worker, 3 hours total, maximum 2,880
single-row forwards, zero training, downloads or decoded continuations. Compile
and verify an additive ledger; preserve partial evidence on interruption. No
retry, tuning from scored rows, threshold change or replacement draws.
If source fails, **do not run recipients**. Attribute to source/probe interface,
not ABI or LayerCake. On passing source, a recipient deficit is a historical
checkpoint/interface result, not a new extraction or a native host failure.
R115's quality, throughput and independent-review gates remain unchanged.

Separately run the LayerCake checkout's existing v25 signed construct tests on
current CPU/CUDA. Keep that receipt in LayerCake, never as ABI acquisition or
product-quality evidence. Do not install an unqualified external core, move a
scientific seal or report a new host benchmark. Human and external hardware
validation remain open. No automatic training or new experiment in this registration.
