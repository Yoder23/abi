# Contributing

ABI is alpha research software. Contributions are welcome when they improve
the supported compiler surface, scientific validity, portability, or
reproducibility.

## Good contributions

- Independent reproductions on genuinely different hardware.
- Teacher adapters with immutable source revisions and license metadata.
- Better English/domain segregation and quarantine tests.
- LoRA and distillation comparisons under identical information and compute
  budgets.
- Smaller qualified artifacts without weakened gates.
- Security, path-traversal, signature, provenance, and mutation tests.

## Before opening a pull request

```bash
python -m pip install -e ".[dev]"
python -m abi self-check
python -m pytest -q tests/test_public_release.py \
  tests/test_capability_pipeline.py tests/test_capability_segregation.py
python -m ruff check abi/__init__.py abi/__main__.py \
  abi/capability_pipeline.py abi/capability_segregation.py \
  abi/layercake_acquisition.py tests/test_public_release.py
python -m build
```

Include the exact command, seed, hardware, source revision, raw measurements,
and claim boundary for experimental changes. Never edit a locked result in
place; publish a superseding artifact and retain the negative or stale result
on a research-history branch.

Large generated results do not belong on the default branch. Add compact,
content-addressed evidence and document how to retrieve the full artifact.

## Conduct

Be direct, rigorous, and kind. Challenge claims with evidence. Do not expose
private data, model credentials, restricted weights, or third-party material
without redistribution rights.
