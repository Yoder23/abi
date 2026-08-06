# ABI Phase 3 Length Attribution V31

Status: **COMPLETE — COMPACT SUBLEXEME SUCCESSOR SUPPORTED**

V31 recomputed the nine V29 development failures without training or changing
LayerCake. All nine belong to `instruction_following`. The maximum is 643
actions. Together they exceed the 320-action bound by 2,101 actions, while
character spelling of unseen lexemes contributes 2,691 avoidable actions.
Every record individually has enough fallback expansion to clear its excess.

The failures are driven by ordinary unseen words and identifiers—such as
`quarterly`, `completion`, `marketing`, `security`, and `department`—being
spelled character by character. This supports one training-derived compact
sublexeme representation. It does not support increasing the 320-action host
limit or adding development-derived whole words.

Result SHA-256: `75f41d5ee10997ebc0cb8b0d751f4286a5020ec4ac74563bdbc7ae7e48865ecc`

Evidence SHA-256: `34960b845906bf2cd1254ad25be1da6c05c6a824bd652066b2b180d5580989b7`

Phase 3 remains uncertified; training remains closed until the compact
representation passes all 8,400 bound targets within the unchanged limits.
