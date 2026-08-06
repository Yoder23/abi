# ABI Phase 3 Representation Bake-off V28

Status date: 2026-08-06

Status: **COMPLETE FAILED — NO REPRESENTATION QUALIFIED**

V28 trained no model and accessed no final data. It tested Unicode-character
fallback behind source pointers at deterministic 10%, 25%, and 50% exposure,
plus a character-only control.

| Candidate | Train coverage | Development coverage | Train mean actions | Development mean actions | Train/dev over 320 | Qualifies |
|---|---:|---:|---:|---:|---:|---:|
| hybrid pointer + char 10% | 7,000/7,000 | 1,394/1,400 | 41.04 | 51.24 | 1 / 3 | No |
| hybrid pointer + char 25% | 7,000/7,000 | 1,394/1,400 | 45.80 | 54.82 | 51 / 19 | No |
| hybrid pointer + char 50% | 7,000/7,000 | 1,394/1,400 | 55.91 | 65.24 | 185 / 29 | No |
| character only | 7,000/7,000 | 1,394/1,400 | 96.90 | 106.30 | 433 / 84 | No |

The training output alphabet lacks `%` and `/`, both required by development
teacher outputs. Six development records therefore fail at the first missing
character. Every candidate also violates the 320-action bound; broad forced
character exposure makes that problem worse.

This is not evidence that LayerCake cannot host a capable artifact. It is a
measured ABI/host-interface requirement: a finite alphabet learned only from
the acquisition outputs is not open vocabulary, while naive character spelling
can violate the existing action-length contract.

The next design may be preregistered only if it supplies a development-
independent universal syntax alphabet/curriculum and preserves compact lexeme
actions. It must not add `%` and `/` as post-hoc exceptions. A principled option
is the complete printable-ASCII alphabet plus every non-ASCII scalar observed
in training, with deterministic syntax-only conformance examples and fallback
used only for unseen lexemes. Any host extension remains a separately owned
LayerCake task after the representation itself passes.

Evidence:

- Result: `results/abi_capability_compiler_phase3_representation_bakeoff/representation_bakeoff_v28.json`
- File SHA-256: `660a3d631eb86a7d9894d70ef1d83f02a586d1fcb2c055b771fe2ecfca4ee6e8`
- Embedded evidence SHA-256: `8e4f32670777ef90a347a496fb7d650f6a6d2b3b5f70484422b7916cb405e3fa`

V28 proves only a negative representation result. Phase 3 remains uncertified,
and ABI has not been shown stronger than LoRA or distillation.
