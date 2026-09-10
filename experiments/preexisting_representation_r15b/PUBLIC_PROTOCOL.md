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
- `public_mod8_v3`: renders the same programs as explicit nested integer
  expressions. It is a materially different source interface intended to
  separate arithmetic competence from ambiguous natural-language execution.
  Qwen2-1.5B-Instruct scored 64/480 (0.1333). Qwen2-7B-Instruct improved to
  43/152 (0.2829), including 18/24 atomic rows, but did not sustain composition.
- `public_mod8_v4`: exercises the source's ordinary deterministic generation
  path, permits explicit intermediate reasoning, and parses only a terminal
  `FINAL: [0-7]` marker. It failed at the interface boundary: the carried-over
  user prompt still required a bare digit, so 0/152 responses contained the
  registered marker. This is a real protocol defect, not a source result.
- `public_mod8_v5`: removes that contradictory output instruction and places
  the `FINAL:` contract in both the system and user messages before rerunning
  the same deterministic generation screen. Qwen2-7B-Instruct scored 24/24 at
  depth 1, 32/32 at depth 2, and 28/32 at depth 4, but 0/32 at depths 8 and 12;
  only 92/152 completions were parseable within 192 tokens.
- `public_mod8_v6`: replaces deeply nested expressions with indexed state
  assignments (`x0`, `x1`, ...) after v5 showed perfect depth-1/2 behavior,
  28/32 depth-4 behavior, and verbose truncation plus skipped operations at
  depth 8/12. The transition semantics and deterministic row set are unchanged.
  All 152 responses were parseable, but accuracy fell from 20/24 at depth 1 to
  4/32 at depth 12. This confirms a substantive deep-composition source limit,
  not only truncation. Track A teacher-behavior equality therefore remains
  failed for this candidate.

## Public representation preflight

`public_extraction_v1` passed the narrower prerequisite. The unmodified
Qwen2-7B-Instruct source produced all six affine anchors exactly. ABI retained
the six anonymous pre-answer final residuals and eight corresponding frozen
output-head rows, then reconstructed the source's digit logits to maximum
absolute error `1.1161773727508262e-08`. The zero-parameter generic decoder
recovered labels `[1,2,0,3,7,0]`, emitted an unchanged 2,053-byte R11 package,
and scored 10,000/10,000 on deep package evaluation. Zero, random, shuffled,
and output-head-shuffled controls were all at or below 0.2507.

The package has a companion semantic label
`mathematics/modular-arithmetic/compositional-affine-mod8`. This is still a
public same-process preflight. Physical isolation, a hidden operator-to-slot
mapping, live replay, recipient execution, hostile verification, and a
preregistered claim are required before R15B can pass.
