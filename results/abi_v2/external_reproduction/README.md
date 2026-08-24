# ABI V2 independent different-hardware reproduction

This directory defines the external gate; it does not claim that gate is complete.
The generated archive contains the exact four capability packages, frozen host
adapters, canonical ABI V2 implementation and test vectors, public evaluation
suites and reference outputs, minimal LayerCake runtime, local reference
evidence, commands, and an evidence schema. It excludes development caches and
Qwen/Pythia weights.

The expected outputs are public reference material, not hidden answers. This is
an exact-execution reproduction of a preregistered frozen matrix, not a new
blind holdout-quality experiment.

## 1. Verify the archive before extraction

Compare the archive SHA-256 with `archive_receipt.json` from the release. After
extracting, enter `abi-v2-clean-room` and run:

```bash
python -m abi_v2.verify_external_bundle --root . --strict
```

The command must report `PASS_EXACT_BUNDLE_IDENTITY` before model download or
matrix execution.

## 2. Create an isolated runtime

Use Python 3.10. The development runtime is pinned in
`abi_release/abi_v2/external_runtime_manifest.json`. Install ABI from the
included source plus PyTorch appropriate for the independent CUDA host:

```bash
cd abi_release
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[extraction,human]" psutil
```

On Windows, use `.venv\Scripts\python.exe` instead. Record `python --version`,
`pip freeze`, OS, CPU, GPU, driver, RAM, and VRAM in the operator receipt.

## 3. Obtain and verify exact host checkpoints

Download only these frozen revisions without `trust_remote_code`:

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --local-dir ../models/qwen2.5-0.5b
huggingface-cli download EleutherAI/pythia-160m --revision 50f5173d932e8e61f858120bcb800b97af589f46 --local-dir ../models/pythia-160m
```

Run capability-blind certification into new, empty directories. Do not expose
any `.cake` path to these commands:

```bash
python -m abi_v2.host_certification --host layercake --device cpu --output-dir operator/host_certification/layercake
python -m abi_v2.host_certification --host qwen2 --device cuda --snapshot ../models/qwen2.5-0.5b --output-dir operator/host_certification/qwen2
python -m abi_v2.host_certification --host pythia --device cuda --snapshot ../models/pythia-160m --output-dir operator/host_certification/pythia
```

Each adapter SHA-256 must equal the frozen hash in
`results/abi_v2/adapters/manifest.json`. A snapshot-inventory mismatch is a hard
failure; do not substitute a nearby model revision.

## 4. Execute the frozen 3-host × 4-capability matrix

The matrix consumes the already frozen adapters under
`results/abi_v2/host_certification/initial/`. Run each host once into a new
output directory:

```bash
python -m abi_v2.capability_matrix --protocol abi_v2/matrix_protocol_amendment3.json --host layercake --device cuda --output-dir operator/matrix/layercake
python -m abi_v2.capability_matrix --protocol abi_v2/matrix_protocol_amendment3.json --host qwen2 --device cuda --snapshot ../models/qwen2.5-0.5b --output-dir operator/matrix/qwen2
python -m abi_v2.capability_matrix --protocol abi_v2/matrix_protocol_amendment3.json --host pythia --device cuda --snapshot ../models/pythia-160m --output-dir operator/matrix/pythia
```

Every result must be `PASS_COMPLETE_FROZEN_ADAPTER_FOUR_CAPABILITY_MATRIX`. Keep
the entire `operator/` tree; do not copy development-laptop timings into it.

## 5. Verify and return evidence

Run the focused code tests and the immutable local-reference verifier:

```bash
python -m pytest -q tests/test_abi_v2_host_conformance.py tests/test_abi_v2_capability_matrix.py tests/test_abi_v2_verify_release.py tests/test_abi_v2_external_bundle.py
python -m abi_v2.verify_release --check-existing
```

The independent operator must return:

- the extracted archive manifest hash;
- machine/runtime receipt;
- three certification directories;
- three matrix directories;
- console logs and exit codes;
- SHA-256 inventory of all returned files;
- a signed statement that the run occurred on hardware different from the
  development RTX 3080 Laptop GPU system.

Only a separately audited receipt may change
`independent_different_hardware_reproduction` to true. Human ratings and the
minimum-information frontier remain separate gates.
