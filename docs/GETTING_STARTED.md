# Getting started with ABI

This guide exercises the supported dependency-light alpha API. Teacher-model
experiments and historical campaign reproduction require additional artifacts
and should be run only from their exact protocols.

## Requirements

- Python 3.10 or newer
- Git
- Git LFS for a complete evidence checkout
- Optional: a CUDA environment for extraction experiments that explicitly
  require it

## Install from a checkout

```bash
git clone https://github.com/Yoder23/abi.git
cd abi
git lfs install
git lfs pull
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Activate the virtual environment using the command for your shell, or invoke
its Python executable directly.

## Verify the installation

```bash
python -m abi --help
python -m abi status --json
python -m abi self-check
python -m examples.segregate_capabilities
```

`status` reports the claim boundary; it is not a release certificate.
`self-check` validates a deterministic source manifest and a domain-free
English record without downloading a model. The example constructs one
English record and one chemistry record and verifies their segregation.

## Optional dependency groups

```bash
python -m pip install -e ".[host]"
python -m pip install -e ".[extraction]"
python -m pip install -e ".[human]"
```

Use `host` for Torch-based runtime helpers, `extraction` for teacher research,
and `human` only for the frozen rating workflow. Installing a dependency group
does not qualify a machine, model, or result.

## Supported Python surface

The stable-for-this-alpha import surface is exported by `abi/__init__.py`.
This minimal example creates an immutable source manifest:

```python
from abi import build_source_model_manifest

manifest = build_source_model_manifest(
    model_id="organization/model",
    revision="a" * 40,
    revision_is_immutable=True,
    architecture="ExampleForCausalLM",
    parameter_count=1,
    tokenizer_id="organization/tokenizer",
    tokenizer_revision="a" * 40,
    license_id="example-license",
    weight_files=[
        {
            "relative_path": "model.safetensors",
            "sha256": "0" * 64,
            "bytes": 1,
        }
    ],
)
print(manifest["source_manifest_sha256"])
```

For a complete labeling/segregation example, read
[`examples/segregate_capabilities.py`](../examples/segregate_capabilities.py).
Phase-numbered and experiment-numbered modules are research surfaces, not
stable APIs.

## Run supported checks

```bash
python -m pytest -q \
  tests/test_public_release.py \
  tests/test_capability_pipeline.py \
  tests/test_capability_segregation.py
python -m ruff check abi/__init__.py abi/__main__.py \
  tests/test_public_release.py examples/segregate_capabilities.py
python scripts/check_docs.py
python -m build
```

The complete default research suite is intentionally stricter and requires
Git LFS payloads plus exact LayerCake sibling commits. Follow the
[independent-review handoff](INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md).

## Human review

Do not edit the sealed rating packet. Three independent humans must each
complete one blinded 7,000-row form. Begin only after reading the
[human-rating handoff](PHASE2_HUMAN_RATING_HANDOFF_V1.md). The current count is
0/21,000; test fixtures and software validation never count as human ratings.

## Common mistakes

- Treating `abi status` as scientific certification.
- Combining R7 runtime evidence with V1089 speed and R97 quality.
- Calling a bounded ontology exhaustive capability discovery.
- Calling exact package replay lossless foreign-teacher knowledge transfer.
- Running an independent-hardware workflow on the development machine.
- Ignoring missing LFS payloads or the disclosed Phase 5 tensor gap.

For claim language, use [CLAIMS.md](../CLAIMS.md), not roadmap text.
