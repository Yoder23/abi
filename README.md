# ABI

ABI is an open research compiler for acquiring measured capabilities from
frozen, open-weight language models. It records source provenance, labels
English-form and specialist-domain material separately, produces immutable
selection artifacts, and verifies that downstream packages do not silently
cross capability boundaries.

ABI does **not** serve models. [LayerCake](https://github.com/Yoder23/layercake)
is the separate execution host that installs, composes, routes, and runs
capability packages.

> Status: alpha research software. ABI V2's first final-validation certificate
> is historical and superseded: adversarial review found that certification
> relied on path filtering, causality replayed stored evidence, and final
> verification consumed summary fields. The repaired validation uses physically
> isolated certification capsules, new live causal interventions, fresh raw
> isolation executions, and fail-closed recomputation with zero trusted
> scientific gate booleans. The bounded result remains a standalone capability-runtime
> across the LayerCake v25, Qwen2.5-0.5B, and
> Pythia-160M codec/conformance environments using one capability-blind,
> zero-parameter frozen adapter per environment and the same four immutable packages.
> Promotion of the repaired candidate additionally requires public immutable
> release assets, clean reconstruction from those assets, and a fresh blind
> red-team. External reproduction, real human ratings, and minimum-information
> certification remain open. The bounded technical declaration is
> `ABI TECHNICAL MOONSHOT: PROVEN`; it is not a full externally validated
> moonshot or a tensor-transplant claim. See
> [Final-mile status](docs/final-mile-status.md).

The superseded certificate is preserved at
[`results/abi_final_validation/release_certificate.json`](results/abi_final_validation/release_certificate.json).
Repaired raw evidence is under `results/abi_final_validation_v2/`; the old
certificate must not be used to claim current readiness.

## What is implemented

- Immutable manifests for source models, weights, tokenizers, and revisions.
- Probe-bounded capability inventories; ABI never calls a finite survey an
  exhaustive account of everything a teacher knows.
- Explicit segregation of English linguistic form, specialist knowledge, and
  quarantined ambiguous material.
- User-selected English and domain extraction plans.
- Nested teacher-information budgets with byte, token, parameter, RAM, disk,
  and runtime accounting.
- Content-addressed `.abix` and `.abicir` acquisition artifacts.
- Fail-closed verification for stale hashes, unsafe paths, mixed domains,
  unqualified records, and undeclared teacher material.
- A bounded ABI-to-LayerCake reference integration in which the teacher is
  absent at inference and selected packages execute independently.
- A representation-neutral standalone capability-runtime ABI V2 with exact UTF-8 anchors,
  fixed semantic channels, capability-blind host certification, immutable
  adapter/package checks, and a three-host/four-capability conformance matrix.

An ABI acquisition artifact is not itself a deployable LayerCake cake. ABI
prepares and certifies acquisition material; LayerCake owns production runtime
packages. The ABI V2 runtime in this repository is a research conformance
reference, not a replacement for LayerCake serving.

## Install

```bash
git clone https://github.com/Yoder23/abi.git
cd abi
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m abi self-check
```

Python 3.10 or newer is required. GPU-backed teacher extraction requires a
compatible PyTorch installation and the extraction extras:

```bash
python -m pip install -e ".[extraction]"
```

The manifest, labeling, packaging, and verification APIs require no model
download and keep the default installation lightweight.

The optional LayerCake host/certification modules use a separately declared
runtime extra:

```bash
python -m pip install -e ".[host]"
```

## Start with the supported API

```python
from abi import build_source_model_manifest

source = build_source_model_manifest(
    model_id="organization/model",
    revision="a" * 40,
    revision_is_immutable=True,
    architecture="ExampleForCausalLM",
    parameter_count=500_000_000,
    tokenizer_id="organization/model",
    tokenizer_revision="a" * 40,
    license_id="Apache-2.0",
    weight_files=[
        {
            "relative_path": "model.safetensors",
            "sha256": "0" * 64,
            "bytes": 1_000_000,
        }
    ],
)

assert source["promotion_eligible"]
```

Run `python -m abi status` for the machine-readable claim boundary and
`python -m abi self-check` for a dependency-light integrity check. The complete
segregation example is in
[`examples/segregate_capabilities.py`](examples/segregate_capabilities.py).

Independent human raters use the optional signing dependency and one command
per sealed form:

```bash
python -m pip install -e ".[human]"
abi human-rate --rater R1
abi human-rate --rater R2
abi human-rate --rater R3
```

External operators use the turnkey `abi-reproduce verify`, `certify-hosts`,
`capability-matrix`, `causality`, `isolation`, `performance`, `hostile-audit`,
and `report` sequence. Certification uses a private mount namespace whose
filesystem contains no capability package or source-success ledger. External
execution commands fail closed on the development hardware.

For the ABI V2 three-host/four-capability result, use the dedicated
[different-hardware clean-room procedure](results/abi_v2/external_reproduction/README.md).
The tracked builder creates one content-addressed archive with public evaluation
material and no model weights or development caches; archive construction alone
does not count as independent reproduction.

Verify the controlling host-independence release layer with:

```bash
python -m abi_v2.verify_host_independence --check-existing
```

## Architecture

```text
frozen teacher
     |
     v
source manifest -> bounded probes -> labeled records -> quarantine
                                             |
                                             v
                                  user selection + budgets
                                             |
                                             v
                          immutable ABI acquisition artifact
                                             |
                              canonical external interface
                                             v
                                      LayerCake host

immutable capability package -> canonical ABI V2 runtime -> frozen codec adapter
                                                        -> native UTF-8/token units
```

The label boundary is deliberately strict: English-core records may encode
linguistic behavior but cannot carry declared specialist claims. Domain facts,
procedures, reasoning, and code are routed to named domain artifacts. Ambiguous
records are quarantined rather than guessed into the core.

Read [Architecture](docs/architecture.md) for the component contract and
[Repository layout](docs/repository-layout.md) before extending the system.

## Evidence and claims

The default branch contains the current compact state and certificates in
[`evidence/current`](evidence/current). The full 5,191-file experimental ledger,
including negative results and bulk generated evidence, is preserved on the
[`research-history-v1089`](https://github.com/Yoder23/abi/tree/research-history-v1089)
branch.

The current evidence supports bounded, exact claims—not universal ones:

- Phases 0 and 1: complete under the registered campaign.
- Phase 2: machine packet ready; 0/21,000 independent human preferences.
- Phase 3: machine gates pass, conditional on Phase 2.
- Phases 4–7: certified for their registered bounded machine scopes.
- Phase 8: the former 18/18 certificate is superseded. The repaired candidate
  passes local strict raw recomputation; immutable publication, public-manifest
  reconstruction, and blind red-team remain mandatory before external review.

The original final-mile V1 campaign remains important negative evidence: the
LayerCake-native tensor contract was executable by only 1/3 receiver families.
ABI V2 changes the abstraction rather than forcing those tensors into foreign
residual coordinates. Each named environment is certified once against a
canonical extension/runtime boundary, then the unchanged English, Python,
chemistry, and civics packages are installed without fitting or calibration.

| ABI V2 local technical gate | Result |
| --- | --- |
| Physically isolated capability-blind host certification | 3/3 pass |
| Frozen host/capability matrix | 12/12 pass |
| Frozen source-success retention | 5,043/5,043 |
| Source-output byte identity | 5,043/5,043 |
| Cross-host specialist action identity | 900/900 |
| Fresh live isolation | 0/2,100 target successes |
| Fresh live causal interventions | 3,072/3,072 recomputed |
| Generic adapter overhead | all three within 10% on 20 observations |
| Real human ratings | 0/21,000; open |
| Immutable public release assets | pending |
| Clean public-manifest reconstruction | pending |
| Fresh blind Codex red-team | pending |
| Independent different-hardware run | closed until the three preceding gates pass |
| Stable minimum-information frontier | open |
| Repaired local strict verifier | pass; 59 input files bound, 0 trusted booleans |

The precise claim is that ABI V2 demonstrates a standalone capability-runtime
with capability-independent package installation across the three named
codec/conformance environments. Fresh live intervention runs show that real,
neutral, zero, random, shuffled, and removed host state cannot alter capability
semantics because the frozen runtime exposes no host-state semantic input.
Adapter and capability removals fail in live execution. The frozen Qwen/Pythia
checkpoints provide conformance probes and their tokenizers provide native unit
representations.
This is an extension/runtime ABI result, not host-model generation or tensor
transplantation into base weights.

ABI does not currently claim a global information minimum, universal
superiority over LoRA or distillation, exhaustive teacher-knowledge discovery,
zero semantic loss for arbitrary models, universal LLM compatibility, human
quality completion, or independent production release certification.

## Repository map

- `abi/` — Python library and research implementations.
- `tests/` — unit, adversarial, and campaign-verifier tests.
- `examples/` — supported API examples.
- `docs/` — architecture, status, contribution, and reproduction guidance.
- `evidence/current/` — compact V1089 state and bounded certificates.
- `contracts/` — preregistered final-mile host-portability contract.
- `results/abi_final_mile/` — compact final-mile evidence; large local package
  payloads and private signing custody remain intentionally untracked.
- `experiments/` — reusable experiment drivers, not the active public API.
- `artifacts/` — small checked-in schemas and reference artifacts.

- `abi_v2/` — canonical ABI V2 specification, conformance harness, matrix,
  and inference-free release verifier.
- `results/abi_v2/` — V1 freeze, three host certifications, raw 3x4 matrix
  evidence, summaries, hostile audit, and technical release certificate.
- `results/abi_v2/external_reproduction/` — independent-operator commands,
  raw evidence schema, and archive receipt. The generated archive is a release
  payload and is intentionally not committed to Git.

- `results/abi_final_validation/` — preserved, superseded first final-validation
  certificate and its historical evidence.
- `results/abi_final_validation_v2/` — repaired physical certification, live
  causality/isolation rows, strict recomputation, and adversarial evidence.
- `review_packet/` — concise claim-to-evidence handoff for technical reviewers.
- `external_reproduction/` — final turnkey independent-hardware workflow.

## Development

```bash
python -m pytest -q tests/test_public_release.py \
  tests/test_capability_pipeline.py tests/test_capability_segregation.py
python -m ruff check abi/__init__.py abi/__main__.py \
  tests/test_public_release.py examples/segregate_capabilities.py
python -m build
```

The three certified compiler modules exercised by these tests retain their
exact evidence-bound bytes and are therefore not autoformatted in place.
Plain `pytest` runs this same supported default-branch suite. Historical
campaign tests remain addressable by explicit path, but many require the bulk
ledgers and generated fixtures preserved on `research-history-v1089`.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Scientific
corrections, independent reproductions, new teacher adapters, and hostile
verifier tests are especially welcome.

## License

Source code is licensed under Apache-2.0. Teacher models, datasets, generated
records, and derived artifacts may carry additional terms; contributors must
record and satisfy those terms independently.
