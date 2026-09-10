# R16 public factual-acquisition prerequisite

Status: `PUBLIC_PREREQUISITE_OPEN`

## Question

R16 asks whether ABI can recover facts already available in an unchanged
open-weight source, assign them to separate semantic namespaces, and emit a
compact normalized payload without giving the source the answer or allowing
post-answer reasoning to enter the captured representation.

This public prerequisite deliberately follows R15B but removes its central
shortcut. R15B captured the state before a terminal digit after the source had
already written that digit in its reasoning. R16 captures the prompt-end source
state before any assistant token is generated.

## Public scope

The public development set contains eight chemistry facts and eight geography
facts. Each fact is evaluated through:

- one answer-free, open-response prompt;
- three independently worded multiple-choice prompts;
- a different deterministic option permutation in every view; and
- a prompt-end final residual plus four frozen output-head rows captured before
  source generation.

The source must receive zero training. All open responses and all shuffled
multiple-choice views must match the registered oracle. The three views for a
fact must agree on one normalized value. A generic relation classifier must
assign every row to the correct registered namespace without using a file path,
capability ID, success ID, answer key, or package identity.

## Public gates

- 16/16 open answers exact after registered normalization;
- 48/48 generated multiple-choice answers exact;
- 48/48 prompt-end projected selections exact;
- 16/16 extracted facts agree across all three views;
- 48/48 semantic namespace assignments exact;
- zero source optimizer steps and zero trainable source parameters;
- representation, prompt, completion, model-revision, and information counts
  preserved in raw evidence; and
- zero answer text in the question stem. Candidate values may appear only in
  the explicitly shuffled option set and no option may be marked as correct.

## Claim ceiling

A pass establishes only that this public source/interface can support a
bounded factual acquisition and semantic-segregation experiment. It does not
certify held-out extraction, autonomous open-world domain discovery, package
execution, LayerCake ingestion, English transfer, global minimality, or
superiority over LoRA or distillation.

No held-out R16 certification is authorized until this public prerequisite is
preserved and reviewed.

## Public diagnostic ledger

- `public_v1`: failed. The unchanged source produced 15/16 exact open answers,
  43/48 exact generated and prompt-end-projected multiple-choice selections,
  and 13/16 three-view-consistent extracted facts. Semantic namespaces were
  48/48 exact. The sole open mismatch was the valid accented spelling
  `Brasília`, which the v1 normalizer did not equate with registered `Brasilia`.
  All five multiple-choice misses were chemistry rows; generated and projected
  selections agreed on every miss, identifying the shuffled-letter interface
  rather than tensor reconstruction as the limiting factor. This formulation
  is preserved and closed.

The next public formulation may replace letter selection with answer-free
open generation plus candidate-sequence scoring over a registered domain value
vocabulary. It must not delete facts, relax the all-row gate, or select a
passing subset after observing v1.

- `public_v2` changes only that measured interface and Unicode normalization.
  It keeps all 16 facts and all three question templates, removes answer
  options from source prompts, captures the residual before any assistant
  token, and ranks every registered value in the fact's relation by conditional
  token likelihood. Accented and unaccented Unicode forms compare after NFKD
  normalization. The 48/48 open-generation, 48/48 candidate-scoring, 16/16
  cross-view fact-agreement, and 48/48 namespace gates remain exact.
  Revision 001 completed source inference but failed closed before emitting a
  receipt because its aggregate gate requested the nonexistent key
  `facts_extracted_total` instead of registered `facts_total`. That partial
  output is not scientific evidence. The aggregation repair is unit-tested and
  revision 002 must rerun the complete source workload into a new directory.

  Revision 002 passed: 48/48 answer-free open generations, 48/48
  candidate-sequence selections, 16/16 three-view-consistent extracted facts,
  and 48/48 semantic namespace assignments. The source remained unchanged with
  zero trainable parameters. It stored 384 candidate score values and 172,032
  prompt-end residual values, used 135 generated tokens, and completed in
  44.41 seconds with 15,326,131,200 peak allocated GPU bytes. This opens a
  separately preregistered held-out R16 experiment; it is not itself a held-out
  extraction or package-execution claim.
