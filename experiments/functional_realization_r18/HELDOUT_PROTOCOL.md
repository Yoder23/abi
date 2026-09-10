# R18 held-out lexical replication protocol

## Prerequisites

The public R18 mechanism, strict verifier, physical live replay, and hostile
audit must be committed before the hidden seed commitment. The exact compiler,
factorization axis, package format, source revision, row counts, evaluator,
controls, and gates are code-hash bound before reveal.

## Hidden selection

A 256-bit secret orders a frozen bank of lexical frames independently for
each of the 24 registered mood x tense x polarity x subject-number signatures.
The first three are extraction frames and the next two are evaluation frames.
Every bank entry is lexically disjoint from R17/R18 public data. Only the
secret commitment and reveal-file hash are committed initially.

The unchanged pinned Qwen2-7B-Instruct source freely realizes all 120 frames
on GPU. Raw outputs are sealed before ABI compilation. Source exact-string
agreement is measured, not used as a prerequisite. Compilation is authorized
only when at least two of three extraction outputs per signature can be parsed
into a slot template.

## Isolation and gates

The frozen R18 compiler receives only the 72 shuffled extraction IDs,
signatures, lexical slots, and source outputs inside the Linux
pivot-root/no-network capsule. It receives no prompts, expected answers,
evaluation frames, reveal, secret, or success IDs. The source is absent from
compilation and package execution.

The held-out replication passes only if:

- all 24 signatures have at least two parseable extraction observations;
- exactly one 24-template English realization package is produced;
- package functional exactness is 48/48 on the hidden evaluation frames;
- no row answered functionally correctly by the source regresses;
- package functional exactness is at least the independent modal baseline;
- package removal abstains 48/48;
- the mood-permutation control is at most 10% functional exact;
- source and host training steps are zero;
- package, prompt, token, byte, isolation, and runtime costs are accounted;
- strict recomputation, live physical replay, and hostile mutation audits pass.

## Claim ceiling

A pass proves a replicated bounded teacher-derived compositional surface-
realization package over this registered 24-signature grammar family and
hidden lexical selection. It does not prove unrestricted English fluency,
prompt understanding, conversation, reasoning, autonomous labeling,
arbitrary-domain extraction, production LayerCake ingestion, global
minimality, or superiority to LoRA or distillation. The full ABI moonshot
remains open.
