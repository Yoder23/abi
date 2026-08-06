# ABI capability-compiler Phase 3 pointer-core V24 report

Status date: 2026-08-06

V24 is complete failed and closed. Phase 3 remains uncertified; Phases 4
through 8 remain locked.

## Controlled representation comparison

V24 held V23's data, imported-information budget, 4,011,040-parameter topology,
4,575-entry fixed vocabulary, seed, initialization, sampler sequence, optimizer,
and 4,000-step budget fixed. The record-sequence hash is identical. The intended
change was target representation: 44,336 eligible actions across all 7,000
training records became LayerCake-native source-position pointers, with every
training target still reconstructed byte-exactly.

V24 scored 601/1,400 (42.93%; Wilson 95% CI 40.36%-45.54%) versus V23's
504/1,400. The paired capability-stratified bootstrap delta is +6.93 points
(95% CI +4.43 to +9.43; 10,000 replicates). That measured effect cannot
override the absolute gates.

V24 had 139 repetition collapses versus 77 for V23, 31 generation errors versus
zero, and prompt grounding fell from 50/100 to 39/100. Coherence and fact-free
reasoning remained 0/100. Every discriminating absolute gate failed. Even an
impossible best case that grants a functional pass to all 31 errored prompts is
only 632/1,400. The teacher comparison, controls, remaining seeds, and final
material were not reached.

## Ownership boundary

The principal result is an ABI acquisition/representation failure: broad
pointer supervision improves some template tasks but does not produce reliable
grounded English and substantially worsens collapse.

The run also exposes a separate LayerCake host-interface validity gap. The
byte-lexeme action surface can independently address fragments of multibyte
UTF-8 characters, so an autonomous action sequence can assemble invalid UTF-8;
all 31 generation errors were strict `UnicodeDecodeError`s. This is not a
sealed-host regression and does not explain the ABI failure: removing every
such error still leaves the candidate far below the gates. It requires a
separate LayerCake post-release audit, not another V24 training run.

## Verification

The verifier binds the protocol, runtime repair, checkpoint, tokenizer, model
configuration, receipt, and raw JSONL. It recomputes functional and collapse
scores for all 1,400 prompts, reconstructs aggregates and Wilson intervals,
proves the matched V23 seed/sequence/architecture/information identity, and
runs the preregistered 10,000-replicate paired bootstrap. Tests reject receipt,
raw-output, candidate-governance, and decision mutations.

## Claim boundary

V24 is negative evidence. It does not establish teacher-relative quality,
Phase 3, LayerCake performance, or ABI superiority over LoRA or distillation.
