# ABI proof ledger

Updated: 2026-08-30

This ledger separates software portability from neural transplantation. A
result in one row must never be promoted into another row without its own
registered evidence.

| ID | Claim | Current evidence | State |
| --- | --- | --- | --- |
| ABI-C1 | Immutable packages retain exact identity and canonical semantics | R7 public release, reconstruction, isolation, and hostile verification | Bounded pass |
| ABI-C2 | Compatible hosts can execute a package through a shared capability runtime | R7 across the three named conformance environments | Bounded pass |
| ABI-C3 | A source learning event can be extracted into a canonical package | R8 synthetic transition source; exact atomic extraction | Bounded synthetic pass |
| ABI-C4a | Identical source-extracted canonical-IR bytes can be pasted, removed, and restored through a shared runtime across frozen heterogeneous hosts | R10: all 61,440 recipient rows passed; exact live replay | Bounded synthetic component pass |
| ABI-C4 | End-to-end source-behavior copy/paste into frozen heterogeneous hosts with zero host training | R10 recipient runtime passed, but source-native generalization failed at 0.594-0.652 | Failed exact R10 contract; open for a materially different successor |
| ABI-C5 | The recipient's native neural computation internalizes the foreign capability | R8 and R9 recipient realization failures | Failed for tested mechanisms |
| ABI-C6 | ABI extracts and segregates fluent English and arbitrary domains from an open-weight LLM | No sufficient evidence | Open |
| ABI-C7 | ABI matches teacher generation quality and supersedes LoRA/distillation | No sufficient evidence | Open |
| ABI-C8 | The full ABI moonshot is complete | Dependencies C4-C7 are not all passed | Not passed |

## Exact distinctions

### Runtime-owned copy/paste

The package is immutable and recipient-independent. A fixed capability-blind
interpreter executes its canonical semantics, while a host codec realizes the
result through the host output interface. Removing the package removes the
behavior; restoring identical bytes restores it. This is useful capability
portability, analogous to portable code plus an instruction-set runtime.

### Native neural transplantation

The same package changes a recipient's neural computation through a fixed
capability-blind neural backend, with no capability-specific optimization, and
the recipient produces teacher-equivalent behavior. The backend may not be an
external answer solver. R8 and R9 did not establish this.

### Teacher-quality English/domain extraction

An open-weight source is diagnosed, English and domain knowledge are separated,
the selected information is packaged, the source is absent at inference, and
the resulting LayerCake passes teacher-relative generation-quality gates. No
existing ABI release establishes this.

## Promotion rule

R10 promotes only ABI-C4a, the bounded runtime component. It does not promote
ABI-C4 because its registered source-native gate failed, and it cannot promote
ABI-C5, ABI-C6, ABI-C7, or ABI-C8. Native transplantation remains a separate
research line; English/domain acquisition remains a separate teacher-quality
line.
