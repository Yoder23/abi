# Active mission

## Status

`R7_BOUNDED_TECHNICAL_VALIDATION_PASSED_EXTERNAL_REVIEW_OPEN`

ABI R7 has passed local strict validation, pre-public hostile validation,
durable public publication, clean public-manifest reconstruction, post-public
hostile replay, and a fresh blind Codex red-team. The public release is:

<https://github.com/Yoder23/abi/releases/tag/abi-final-validation-v2-repaired-r7-2026-08-30>

This opens human and independent-hardware review. It does not complete them.

## What R7 proves

R7 proves one bounded capability ABI result: four immutable packages execute
through one canonical capability-runtime/conformance contract across three
declared environments without capability files or success IDs being present
during host certification.

The controlling evidence contains:

| Check | Result |
|---|---:|
| Physical certification environments | 3/3 |
| Certification roundtrips | 384/384 |
| Reachable inventory rows | 301,543 |
| Reachable bytes content-scanned | 11,681,888,205 |
| Capability signatures / success IDs in certification | 0 / 0 |
| Locked matrix rows | 5,043/5,043 |
| Live causality rows / distinct condition PIDs | 3,072 / 24 |
| Live isolation rows / target successes | 2,100 / 0 |
| Pre-public hostile controls | 19/19 rejected |
| Public reconstruction focused tests | 17/17 passed |
| Blind USTAR/V7 prefix controls | 12/12 passed |
| Blind post-public hostile replay | 19/19 rejected |

The blind reviewer returned `VERDICT: PASS` for this exact scope.

## What remains open

| Gate | State |
|---|---|
| Three-rater human packet | OPEN; 0/21,000 judgments |
| Different-hardware reproduction | OPEN; not yet executed |
| Registered minimum-information certification | OPEN; not yet executed |
| Teacher-to-ABI English extraction | RESEARCH; not proven by R7 |
| Capability labeling and segregation from arbitrary teachers | RESEARCH |
| Quality parity/superiority versus teacher, LoRA, or distillation | RESEARCH |
| Globally minimal English substrate | RESEARCH |

## Current objective

Prepare and execute external review without changing the R7 artifacts:

1. give the frozen human packet to three independent raters;
2. give the public archive and external reproduction checklist to an
   independent different-hardware operator;
3. preserve every returned raw row, command, environment inventory, signature,
   and failure;
4. verify the returned packets fail closed; and
5. only then update the corresponding external gates.

Teacher extraction and minimization belong to a subsequent additive campaign.
They must not be inferred from R7 conformance evidence.

## Historical failures that remain authoritative

- R3: public reconstruction succeeded, but blind review rejected incomplete
  content binding.
- R5: blind review found first-window archive scanning and ordinary-row
  commitment defects.
- R6: blind review found valid tar streams hidden behind arbitrary prefixes and
  a Windows long-member reconstruction failure.
- R7: repairs both R6 defects and passes fresh public/blind validation.

Negative evidence is preserved under `results/abi_final_validation_v2/` and
must not be deleted or overwritten.
