# ABI Capability Compiler

ABI is an alpha research compiler for acquiring, labeling, segregating,
packaging, and verifying bounded capabilities from open-weight teacher models.
It produces immutable, provenance-bound artifacts for a separate host such as
[LayerCake](https://github.com/Yoder23/layercake).

[Get started](docs/GETTING_STARTED.md) | [Documentation](docs/README.md) |
[Current status](docs/PROJECT_STATUS.md) | [Versions](docs/VERSIONING.md) |
[Changelog](CHANGELOG.md)

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
| R108 | Pinned inputs and preserved resource-budget stop: 13/96 initial responses, no replay or recipients | Control qualification and trained-task comparison remain incomplete |
| R107 | No-training residual intervention and exact live replay; modest natural-text loss interference, with a repetition trade-off | No fluent core, trained-route retention or native-host promotion |
| R7 public release | Bounded capability runtime/conformance across three named environments; public reconstruction passed | Human ratings and independent hardware |
| V1089 local campaign | Bounded machine evidence through Phase 7 and a same-machine Phase 8 rehearsal | One coherent public final artifact, Phase 5 clean replay, external review |
| R97 | Prospective two-hop reasoning transfer: 1,399/1,400 with causal controls and byte-identical replay | Broad English, arbitrary capabilities, independent reproduction |
| R21-R23 | Bounded label-separated supplied-content generation and prospective semantic replication | General English and autonomous discovery |
| R98 / C1 / R99 | Two source failures preserved; R99 data qualified, but its ABI-plan learner scored 10/48 vs LoRA 27/48 and distillation 20/48 | Failed bounded learning pilot; no fluent-English core promoted |
| R99-D1 / R100 | Fit audit completed; one full-model control scored 20/48 on fresh prompts, equal to retained LoRA | Quality and fit gates failed; acquisition bottleneck not yet isolated |
| R101 | Sampled execution paths agree; 288 exact unassisted replays; primary oracle-prefix continuation contrast is null | Data/generalization, optimizer fit and parent suitability remain unresolved |
| R102 | Matched renaming/exposure factorial completed; neither primary factor supported and no quality gate passed | Recipient/interface suitability remains unresolved |
| R103 | Raw control qualification failed at 41/48; all evidence preserved | Recipient/interface comparison not executed; control/evaluator validity remains unresolved |
| R104 | 155 finite scorer cases passed; raw control failed 29/48, with additional scorer-scope limits disclosed | No recipient comparison; general evaluator validity and control qualification remain open |
| R105 | A frozen NLI evaluator failed external scope labels: 779/1,200; exact live replay | Rejected as a judge; no English-control or acquisition promotion |
| R106 | Separate grammatical-discrimination control failed 178/200; exact live replay | No recipient comparison or training; supplied-token ranking is not fluent generation |

These lineages must not be combined into one certification. The controlling
public technical release remains
[`abi-final-validation-v2-repaired-r7-2026-08-30`](https://github.com/Yoder23/abi/releases/tag/abi-final-validation-v2-repaired-r7-2026-08-30).
The current default branch is a later peer-review candidate.

The latest [R108 attempt](docs/R108_TRAINED_GROUNDING.md) hit its fixed 25-minute
control-process budget under reported GPU power/thermal throttling. Only 13/96
initial responses completed; no replay, recipients or training ran. The complete
verifier rejects it. This is a resource-limited prerequisite, not an English
quality failure, acquisition result or native LayerCake defect.
The [public reconstruction receipt](docs/R108_PUBLIC_RECONSTRUCTION.md) verifies
the preserved interruption and confirms that the complete verifier still rejects it.

The earlier [R106 diagnostic](docs/R106_GRAMMAR_INTERFACE.md) failed its narrow
grammar control at 178/200 against 190/200. It uses external sentence-pair labels,
not the rejected NLI judge; all target tokens are supplied. No recipient or
training run followed. Source-label limitations and exact replay are preserved;
this does not qualify English generation or promote a product claim.

The [R106 research release](https://github.com/Yoder23/abi/releases/tag/abi-research-r106-2026-11-11)
and additive receipt are public. Same-machine reconstruction matched all 581
evidence assets and passed 52 focused checks; tagged CI is green. Publication
does not change the negative scientific decision.

The preceding [R105 evaluator falsification](docs/R105_EVALUATOR_SCOPE.md) rejected
an off-the-shelf semantic judge on external gold labels, with exact live replay.
This is a measured evaluation limitation, not an ABI or LayerCake breakthrough.
No recipient ran, and earlier English-control gates remain closed.

Its [research prerelease](https://github.com/Yoder23/abi/releases/tag/abi-research-r105-2026-11-11)
and reconstruction receipt are public: 40 evidence assets, eight evaluator files
and 41 focused checks verified on the same machine. This publishes the rejection,
not an English-core breakthrough.

The preceding [R104 diagnostic](docs/R104_EVALUATOR_CONTROL.md) stopped at its control
gate: 29/48 under strengthened instruction checks. The legacy screen gives 44/48
but still fails the clarification task floor. Two scorer-scope limitations are
preserved, not patched after observing outputs. No recipient ran and no native
LayerCake defect follows. See also the [weight-space research assessment](docs/WEIGHT_SPACE_RESEARCH_2026_11.md):
direct editing is a future competitor, not an ABI superiority result.
The [R104 research release](https://github.com/Yoder23/abi/releases/tag/abi-research-r104-2026-11-10)
is public; same-machine public recomputation matched 566 assets and 729 bindings
and passed 43 focused checks. No scientific claim is promoted by publication.

The preceding [R103 diagnostic](docs/R103_RECIPIENT_INTERFACE.md) stopped at control
qualification: 41/48 raw passes versus the required 44/48. Recipient comparisons
and training did not run. A temporal false-positive in the frozen contradiction
proxy is documented without changing the result. No native LayerCake defect follows.
The [sealed research release](https://github.com/Yoder23/abi/releases/tag/abi-research-r103-2026-11-10)
includes the complete manifest and public-review helper.

The published [R102 study](docs/R102_COVERAGE_EXPOSURE.md) completed its matched
entity-renaming/exposure controls. Neither primary factor was supported and no
checkpoint passed quality. The evidence is preserved; no new fluent core or
ABI superiority is claimed. Its immutable research tag, checkpoints and review
manifest are [public](https://github.com/Yoder23/abi/releases/tag/abi-research-r102-2026-11-10).

The [R100 acquisition-capacity control](docs/R100_ACQUISITION_CAPACITY.md)
also failed: full-model fine-tuning scored 20/48, equal to retained LoRA and
below the unchanged 44/48 gate. More trainable weights alone did not solve this
bounded acquisition problem. No new core or ABI superiority is promoted.

The [R101 diagnostic](docs/R101_EXECUTION_CONTINUATION.md) found no measured
cached/padded execution disagreement. Correct answer prefixes did not improve
the primary remaining-fact coverage contrast. Assisted scores are not an
English-core breakthrough; the next acquisition question is still open.

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
