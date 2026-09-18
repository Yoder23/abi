# ABI Capability Compiler

ABI is an alpha research compiler for acquiring, labeling, segregating,
packaging, and verifying bounded capabilities from open-weight teacher models.
It produces immutable, provenance-bound artifacts for a separate host such as
[LayerCake](https://github.com/Yoder23/layercake).

ABI and LayerCake have deliberately separate jobs:

- **ABI** qualifies teachers, probes bounded capabilities, labels and
  segregates records, accounts for imported information, constructs artifacts,
  and verifies the evidence.
- **LayerCake** installs, composes, routes, and executes compatible capability
  artifacts. Its runtime and performance claims belong to that repository.

## Current status

**Whole-moonshot verdict: `FULL_MOONSHOT_NOT_PROVEN`.**

ABI has strong bounded results, but it has not yet proved automatic extraction
of a minimal fluent-English substrate from an arbitrary pretrained model. It
also has not completed human evaluation, independent different-hardware
reproduction, global minimality, or a universal superiority comparison against
LoRA and distillation.

| Evidence line | What is established | What remains open |
| --- | --- | --- |
| R7 public release | Bounded capability runtime/conformance across three named environments; public reconstruction passed | Human ratings and independent hardware |
| V1089 local campaign | Bounded machine evidence through Phase 7 and a same-machine Phase 8 rehearsal | One coherent public final artifact, Phase 5 clean replay, external review |
| R97 | Prospective two-hop reasoning transfer: 1,399/1,400 with causal controls and byte-identical replay | Broad English, arbitrary capabilities, independent reproduction |
| R21-R23 | Bounded label-separated supplied-content generation and prospective semantic replication | General English and autonomous discovery |

These lineages must not be combined into one certification. The controlling
public technical release remains
[`abi-final-validation-v2-repaired-r7-2026-08-30`](https://github.com/Yoder23/abi/releases/tag/abi-final-validation-v2-repaired-r7-2026-08-30).
The current default branch is a later peer-review candidate.

Read the [project status](docs/PROJECT_STATUS.md) and [claim
ledger](CLAIMS.md) before citing a result.

## Five-minute start

ABI requires Python 3.10 or newer. Git LFS is required for a complete research
checkout.

```bash
git clone https://github.com/Yoder23/abi.git
cd abi
git lfs install
git lfs pull
python -m venv .venv
python -m pip install -e ".[dev]"
python -m abi status --json
python -m abi self-check
python -m examples.segregate_capabilities
```

The example builds and validates a small in-memory English/domain segregation
manifest. It does not download a teacher, train a model, or claim scientific
quality.

Install optional dependencies only for the workflow you need:

```bash
python -m pip install -e ".[host]"       # Torch host/runtime helpers
python -m pip install -e ".[extraction]" # Teacher extraction research
python -m pip install -e ".[human]"      # Frozen human-rating workflow
```

See the [getting-started guide](docs/GETTING_STARTED.md) for platform notes,
the supported Python API, and verification commands.

## What the supported package does

The public alpha API in `abi/__init__.py` supports:

- immutable source-model manifests;
- bounded capability inventories and user selection plans;
- explicit English-core, specialist-domain, and quarantine records;
- nested teacher-information budgets and cost accounting;
- content-addressed acquisition bundles and verification; and
- deterministic release status and self-check commands.

The canonical flow is:

```text
pinned teacher -> bounded probes -> labeled records -> selection/budget
               -> immutable acquisition artifact -> external host conformance
```

An acquisition artifact is not itself a deployable model and does not silently
include a teacher at inference. Host-side installation and execution are
separate, explicitly measured steps.

## Choose a path

| You want to... | Start here |
| --- | --- |
| Understand the project | [Documentation index](docs/README.md) |
| Run the supported API | [Getting started](docs/GETTING_STARTED.md) |
| Understand the code and evidence tree | [Repository map](docs/repository-layout.md) |
| Review current claims | [Project status](docs/PROJECT_STATUS.md) -> [claims](CLAIMS.md) |
| Reproduce R7 | [R7 public validation](docs/R7_PUBLIC_VALIDATION.md) |
| Conduct independent review | [Independent review handoff](docs/INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md) |
| Run human scoring | [Human-rating handoff](docs/PHASE2_HUMAN_RATING_HANDOFF_V1.md) |
| Run external hardware review | [Phase 8 reproduction](docs/PHASE8_EXTERNAL_REPRODUCTION_V1.md) |
| Contribute | [Contributing guide](CONTRIBUTING.md) |

## Scientific boundaries

ABI does **not** currently establish:

- complete diagnosis of everything a teacher knows;
- fluent, minimal English extraction from arbitrary open-weight models;
- exhaustive or ontology-free domain discovery;
- native tensor transplantation between unrelated architectures;
- zero-loss foreign-teacher transfer;
- human-rated teacher parity;
- independent-hardware reproducibility; or
- general superiority over LoRA, distillation, or fine-tuning.

Within this repository, "lossless" is reserved for exact archive, manifest,
tensor, package, or replay identity inside a declared compatibility boundary.
It does not describe knowledge extraction from a foreign model.

## Verification

Run the lightweight supported surface:

```bash
python -m pytest -q \
  tests/test_public_release.py \
  tests/test_capability_pipeline.py \
  tests/test_capability_segregation.py
python -m ruff check abi/__init__.py abi/__main__.py \
  tests/test_public_release.py examples/segregate_capabilities.py
python scripts/check_docs.py
```

The complete default research suite has additional Git LFS and exact sibling
LayerCake requirements. Follow the [independent review
handoff](docs/INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md); do not interpret a
missing historical payload as a pass.

## Source-of-truth order

When documents disagree, use this order:

1. an immutable tag plus exact content hashes;
2. frozen protocol and campaign contracts;
3. raw observations and immutable packages;
4. fail-closed verifier output;
5. claim and status documents; and
6. roadmap or explanatory prose.

Documentation cannot promote a scientific claim. Negative and superseded
results remain in the repository for auditability.

## License and maturity

ABI is Apache-2.0 licensed alpha research software. Historical experiment
modules are not a stable production API and should not be exposed directly to
untrusted network input. See [SECURITY.md](SECURITY.md), [LICENSE](LICENSE),
and [CONTRIBUTING.md](CONTRIBUTING.md).
