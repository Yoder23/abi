# R20 public instruction-conditioned realization prerequisite

R20 materially broadens R19 from an explicit grammatical signature and slot
dictionary to one raw prompt containing a natural-language instruction and
three opaque supplied content fields. It covers six independently scored
behaviors: prose composition, concise summarization, email drafting, bullet
formatting, clarification, and abstention from an unsupported request.

The disclosed public set contains 72 extraction prompts and 120 distinct
evaluation prompts, exactly 20 per behavior. Extraction and evaluation use
disjoint content and instruction combinations. Unchanged pinned
Qwen2-7B-Instruct realizes every prompt on GPU. Source exactness and functional
quality are reported but are not an all-or-nothing prerequisite; compilation
requires at least six parseable teacher outputs for each of six independently
discovered output-program clusters.

The zero-training compiler receives only shuffled IDs, raw prompts, and raw
teacher outputs. It receives no task labels, expected outputs, evaluation rows,
oracle, or success IDs. It delexicalizes teacher outputs into programs, then
learns an instruction character-ngram centroid for each program. The runtime
accepts only a raw prompt and an immutable package. The teacher is absent.

A causal control rotates extraction instructions across behavior families
while preserving each row's data and output. The control is compiled through a
separate physical capsule. Package removal must abstain.

Public passage requires:

- 120/120 package functional outputs;
- zero regressions on teacher-correct evaluation rows;
- package quality at least equal to the teacher;
- at least 19/20 package-functional outputs in each behavior;
- at most 24/120 control-functional outputs;
- 120/120 removal abstentions;
- zero source, host, or bridge training;
- no source parameters, logits, or activations in the package;
- complete prompt, token, byte, time, memory, and artifact accounting;
- strict recomputation, hostile mutation rejection, and fresh physical replay.

Passing is only a disclosed public prerequisite for bounded instruction-
conditioned supplied-content realization. It is not unrestricted English,
open-ended conversation, arbitrary summarization or rewriting, autonomous
knowledge discovery, LayerCake ingestion, global minimality, or superiority to
LoRA or distillation. A new hidden instruction/content replication is required
before promoting the bounded mechanism. The full ABI moonshot remains open.
