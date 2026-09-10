# R15A v5 fresh blind red-team

Reviewed commit:
`36ad0dba5778fbdc60955c4df7d91ab7f9153606`

Verdict: `PASS`

No critical or high-severity blocker was found for the bounded local R15A
claim.

## Evidence supporting the verdict

- The worktree was clean at the exact reviewed commit.
- The v5 preregistration commit
  `b92bd2ac6de06f43780abf55ff2b678b91c3c8b0` is the reviewed commit's
  immediate parent. The v5 reveal first appears in the child, and no R15
  scientific code or configuration changed after reveal.
- The secret commitment recomputes correctly. All eight v5 capabilities are
  distinct and overlap neither the 320 public events nor v1-v4.
- Each extraction capsule contains only the anonymous delta, frozen frontend
  and specification, and generic worker. Pivot-root removes the old root and
  Windows mount, and a new network namespace is used.
- The delta is the effective targeted output-head change. The initial effective
  update is zero because LoRA B starts at zero; the after-state delta is
  `B^T A^T / rank`. Verification independently derives it from saved factors
  within preregistered tolerances.
- All nine tracked package hashes, manifest identities, certificate
  cross-references, and evidence hashes validate.
- Independently regenerated evaluation gives 1.0 for every after package over
  all 80,000 unseen and 8,000 counterfactual rows.
- Public frontend retraining reproduced the exact frozen tensors and all 64/64
  development capabilities.
- Recipient code uses the unchanged hash-bound R11 executor and codec boundary,
  contains no optimizer path, and reports zero recipient steps with no source
  model loaded. AFTER and RESTORED are 1.0; negative controls are at most
  0.17578125; removals equal BASE.
- Strict verification recomputes capability identities, rows, delta controls,
  package transitions, active recipient neural states, and gates. Live replay
  re-executes source behavior, isolation capsules, and recipients. The hostile
  audit rejects 9/9 mutations.
- Focused R15/R11 tests passed 12/12.

## Non-blocking limitations

- **Claim boundary:** Qwen learned an exhaustively supervised 24-row atomic
  table, not compositional source behavior. Its held-out long-composition
  accuracy is approximately 0.110-0.134. The valid claim is recovery of
  deliberately installed atomic-transition information from a targeted
  8-by-896 LoRA delta, not teacher-behavior cloning or generalized source
  capability.
- **Replay wording:** live verification reloads saved source adapters and
  regenerates source observations; it does not rerun the 1,000-step acquisition
  training. Claims must say source-behavior replay, not training reproduction.
- **Local assurance:** ignored bulk deltas, adapters, raw rows, and mount records
  are not in the tracked compact tree. A clean clone therefore cannot rerun
  full strict/live verification. This does not defeat the explicitly local
  claim, but it prevents public clean reconstruction or independent
  reproduction.
- **Chronology assurance:** Git establishes DAG ordering, not independent
  timestamping or third-party secret custody.

## Explicit ceiling

R15A does **not** prove extraction of pre-existing English or domain knowledge.
It proves bounded recovery of a deliberately installed synthetic atomic
transition table from a before/after targeted Qwen neural update into the
frozen R11 runtime package.
