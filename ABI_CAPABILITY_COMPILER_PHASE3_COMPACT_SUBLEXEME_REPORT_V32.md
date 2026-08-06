# ABI Phase 3 Compact Sublexeme V32

Status: **COMPLETE FAILED**

Training-only substring budgets of 128, 256, and 320 all stayed under 5,000
fixed actions, but none removed the length failures. Development sequences over
320 were 9, 9, and 8; the best maximum remained 461 actions. Training remained
at zero overlength sequences.

The failure is structural: retaining all 4,569 training lexemes leaves too
little vocabulary budget for reusable fallback pieces and provides no
sublexeme supervision on training targets. Training inventory shows 637 output
lexemes never occur in prompts. A successor may prune only those output-only
lexemes and replace them with training-derived sublexemes, preserving every
prompt lexeme and creating real fallback supervision. Do not expand the total
fixed-action budget or use development words.

Result SHA-256: `e59273da28652fc6ef84e4118bc8111ed86d08bc53bf2da9a01fd17371f9a9dd`

Evidence SHA-256: `41259896b0fe0cbb1e5bd4f1f6a417f163c8399c732cf7f499b70db38f4c959c`
