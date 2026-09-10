# R15B public source-qualification protocol

Status: `PUBLIC_QUALIFICATION_OPEN`

R15B asks whether an unchanged pretrained open-weight source already carries a
non-enumerable capability that can be recovered through bounded internal
representation access and converted into the frozen R11 package/runtime
boundary. No R15B held-out certification is authorized until this public
source prerequisite passes.

The first public candidate is compositional arithmetic modulo eight:

- `increment`: `x -> x + 1 (mod 8)`;
- `triple`: `x -> 3x (mod 8)`; and
- `decrement`: `x -> x - 1 (mod 8)`.

These are three noncommuting affine permutations supported by the unchanged
R11 `3 x 8 x 8` neural transition package. The source is not trained or
adapted. Prompts name the operations in ordinary mathematical English and ask
for one canonical digit. The public screen samples disjoint deterministic
programs at depths 1, 2, 4, 8, and 12 and records both canonical output
probabilities and the final-layer residual representation for every row.

This is a source qualification only. A pass does not establish extraction,
semantic labeling, package construction, recipient execution, English/domain
transfer, minimality, or superiority over LoRA/distillation. A failure closes
this candidate formulation and must be preserved before another materially
different public candidate is tried.

## Public diagnostic ledger

- `public_mod8_v1`: failed. Qwen2-1.5B-Instruct scored 67/480 (0.1396)
  overall; Qwen2-7B-Instruct scored 19/152 (0.1250). Generation inspection
  showed that the one-word operation `triple` was interpreted inconsistently.
  This formulation is closed.
- `public_mod8_v2`: replaces ambiguous one-word verbs with complete arithmetic
  instructions while preserving the operations and evaluation generator.
  Qwen2-1.5B-Instruct again scored 67/480 (0.1396). Greedy inspection showed a
  new prompt artifact: the model emitted the repeatedly mentioned operand `3`
  for every inspected atomic row. This formulation is also closed. These are
  source/prompt failures; no ABI extractor or recipient was exercised.
