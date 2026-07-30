# ABI release working contract

## Current state

The bounded ABI-to-LayerCake reference is sealed by
`ABI_MOONSHOT_CERTIFICATE_V2.json`, but the full English-product moonshot is
open. `ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json` records the controlling
0/28 LayerCake versus 19/28 frozen-source audit failure.
`ABI_ENGLISH_CORE_DOMAIN_SEGREGATION_CONTRACT_V2.json` is the controlling
contract for future acquisition work. It supersedes
`ABI_ENGLISH_SUCCESSOR_CONTRACT.json` without modifying that historical
contract.

`LAYERCAKE_ACQUISITION_PROTOCOL.json` and `ABI_ARCHITECTURE_REVIEW.md` are
historical records and must not be edited to imply that they were written
after the result.

The sibling `../layercake_release` repository is sealed at commit
`04cf2927a16fba686cd640e18a78708e5658bbda`. Never edit, move, reset, retag, or
silently dirty it from this repository.

## Permanent claim rules

- Never call domain discovery exhaustive.
- Never call transfer from a non-LayerCake teacher "lossless." Certify measured
  source-relative generation quality. Reserve "lossless" for verified exact
  LayerCake package/archive/tensor transfer.
- Never infer a semantic label from tensor location without a validated
  intervention and held-out behavioral gate.
- Runtime token IDs are authoritative; character estimates are not token
  counts.
- Account prompts, unique UTF-8 bytes, teacher outputs, tokens, logits,
  activations, copied parameters, learned substrate, bridges, disk, RAM, time,
  and hardware separately.
- "Minimum" means lowest passing preregistered tested nested budget paired with
  the largest lower failing budget. It is not a global minimum.
- Keep exact byte/tensor identity separate from bounded semantic retention.
- "Pure English core" is a bounded corpus and behavioral isolation claim. Do
  not claim that finite tests prove literally zero world knowledge in weights.
- `.abix` contains teacher material and is never a deployable cake.
- A new model, catalog, host, or package is unqualified until the same final
  teacher-free LayerCake clears functional, isolation, identity, CPU speed,
  TTFO, RSS, and three-seed/fresh-host gates.

## Development rules

- Preserve all negative evidence. Never overwrite, reset, or mass-delete the
  experiment ledger.
- Pin new source revisions and hash exact source weights.
- Use disjoint search, validation, and final-test splits. Final data cannot
  select prompts, budgets, checkpoints, sources, or repairs.
- Exclude unselected domain records from composed acquisition material.
- New successor training bundles must use segregated v2 records, a validated
  user-governed domain ontology, and a passing core/domain segregation manifest.
- English-core records may teach linguistic form only. They cannot carry
  specialist labels or claims or introduce declared unsupplied facts. Use
  abstract/nonce material, supplied non-domain context, interpersonal
  pragmatics, and domain-free instructions.
- Specialist facts, procedures, code, and domain-labeled calculations belong
  in separately selectable domain artifacts. Quarantine unknown,
  cross-domain, and disputed records.
- Preserve record-level provenance; multi-source conflicts fail closed.
- Test tampering, wrong source/key, leakage, duplicate identifiers, traversal,
  non-nested budgets, teacher-at-inference, inactive execution, and
  finite-suite claim inflation.
- Do not modify files bound by a frozen certificate. Create a versioned
  successor and preserve the prior certificate.
- Do not repair the measured generalization failure with more template or
  symbolic special cases. A successor needs broad acquisition and held-out
  natural-language generalization.

## Verification

```powershell
C:\Python310\python.exe -m abi.moonshot_release verify
C:\Python310\python.exe -m pytest -q
```
