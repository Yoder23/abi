# Repository layout

ABI contains an installable alpha API, experimental implementations, frozen
protocols, immutable evidence, and reviewer handoffs. This map identifies the
right entry point for each kind of work.

## Top-level paths

| Path | Purpose | Use it when |
| --- | --- | --- |
| `abi/` | Installable compiler, labeling, packaging, accounting, and CLI code | Integrating the supported alpha API |
| `abi_v2/` | Canonical runtime/conformance and strict release verification | Reconstructing or auditing the bounded R7 line |
| `examples/` | Small model-free examples | Learning the supported API |
| `tests/` | Supported, exact-lineage, adversarial, and historical tests | Validating changes or replaying a named protocol |
| `experiments/` | Additive research drivers such as R11-R97 | Reproducing one specifically named experiment |
| `contracts/` | Frozen interface and campaign contracts | Checking a registered requirement |
| `evidence/current/` | Compact retained bindings and current campaign state | Locating the exact evidence identity named by a verifier |
| `results/` | Raw rows, packages, receipts, certificates, and negative results | Auditing a named evidence lineage |
| `review_packet/` | Ordered human and independent-review handoff | Starting external review |
| `reviews/` | Preserved independent audits and receipts | Inspecting prior findings without rewriting them |
| `external_reproduction/` | Different-hardware operator workflow | Running external Phase 8 review |
| `docs/` | Current navigation, architecture, claims, results, and handoffs | Understanding the project |
| `scripts/` | Maintenance and verification helpers | Following an exact documented command |

Root-level JSON files are content-addressed protocols, certificates, and
campaign results retained at their historical paths because verifiers bind
those paths and hashes. Do not reorganize them as ordinary documentation.

## Common tasks

| Task | Start here |
| --- | --- |
| Install and smoke-test ABI | [Getting started](GETTING_STARTED.md) |
| Build an English/domain segregation record | `examples/segregate_capabilities.py` |
| Inspect the current claim boundary | [Project status](PROJECT_STATUS.md) and [claims](../CLAIMS.md) |
| Reconstruct R7 | [R7 public validation](R7_PUBLIC_VALIDATION.md) |
| Audit the current review candidate | [Independent-review handoff](INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md) |
| Conduct human ratings | [Phase 2 handoff](PHASE2_HUMAN_RATING_HANDOFF_V1.md) |
| Run different-hardware reproduction | [Phase 8 workflow](PHASE8_EXTERNAL_REPRODUCTION_V1.md) |

## Evidence navigation

Do not start by averaging files under `results/`. Use this order:

1. identify the evidence lineage in [PROJECT_STATUS.md](PROJECT_STATUS.md);
2. read the claim and its exact ceiling in [CLAIMS.md](../CLAIMS.md);
3. open the named protocol and certificate;
4. verify its hashes and raw rows with the exact bound verifier; and
5. check whether all required public/LFS payloads and host commits are present.

R7, V1089, R97, and R21-R23 are separate lineages. A result from one cannot
fill a missing gate in another.

## Generated and local-only paths

Model caches, build outputs, temporary reconstruction workspaces, local human
evidence, and generated intermediate files are excluded by `.gitignore`.
Required immutable release payloads belong in Git LFS or a hash-addressed
GitHub Release, with their manifests committed first. Missing evidence must
fail closed; it must never be replaced with a trusted status flag.
