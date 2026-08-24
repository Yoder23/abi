# ABI

ABI is an open research compiler for acquiring measured capabilities from
frozen, open-weight language models. It records source provenance, labels
English-form and specialist-domain material separately, produces immutable
selection artifacts, and verifies that downstream packages do not silently
cross capability boundaries.

ABI does **not** serve models. [LayerCake](https://github.com/Yoder23/layercake)
is the separate execution host that installs, composes, routes, and runs
capability packages.

> Status: alpha research software. The bounded local machine campaign has
> strong positive results, but independent Phase 8 reproduction and Phase 2
> human preferences remain open. See [Research status](docs/research-status.md).

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

An ABI acquisition artifact is not itself a deployable LayerCake cake. ABI
prepares and certifies acquisition material; LayerCake owns runtime packages.

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
- Phase 8: local clean-export rehearsal passes; independent operator and
  different CPU/CUDA hardware are still required.

ABI does not currently claim a global information minimum, universal
superiority over LoRA or distillation, exhaustive teacher-knowledge discovery,
zero semantic loss for arbitrary models, or production release certification.

## Repository map

- `abi/` — Python library and research implementations.
- `tests/` — unit, adversarial, and campaign-verifier tests.
- `examples/` — supported API examples.
- `docs/` — architecture, status, contribution, and reproduction guidance.
- `evidence/current/` — compact controlling state and bounded certificates.
- `experiments/` — reusable experiment drivers, not the active public API.
- `artifacts/` — small checked-in schemas and reference artifacts.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Scientific
corrections, independent reproductions, new teacher adapters, and hostile
verifier tests are especially welcome.

## License

Source code is licensed under Apache-2.0. Teacher models, datasets, generated
records, and derived artifacts may carry additional terms; contributors must
record and satisfy those terms independently.
