# ABI final independent-hardware reproduction

This package is turnkey for a genuinely independent operator. Running it on the
development laptop is a clean-room rehearsal, not independent reproduction.
Preserve every first-run output, including failures.

## Preconditions

Use a machine not owned or controlled by the ABI developer and not the recorded
RTX 3080 Laptop GPU system. Record CPU, GPU, RAM, VRAM, OS, Python, compiler,
drivers, package versions, commands, exit codes, raw outputs, and timings.
Complete `operator-attestation.template.json` and save it as
`operator-attestation.json` before execution.

Create an isolated Python 3.10 environment and install only the included ABI
source plus declared dependencies from `environment.lock.json`. Choose the
PyTorch build appropriate to the independent hardware and record its full
version. Do not expose any `.cake` file to host certification.

Materialize only the pinned runtime files below at the default locations. Do
not point the verifier at a complete Hugging Face repository checkout: files
such as README, LICENSE, `.gitattributes`, alternate weight formats, or cache
metadata correctly make the exact frozen inventory fail. A convenient method
is to download the exact revision into an isolated cache, then copy only these
named files into the empty destination directories:

```text
external_reproduction/models/qwen2
external_reproduction/models/pythia
```

- `Qwen/Qwen2.5-0.5B` at `060db6499f32faf8b98477b0a26969ef7d8b9987`:
  `config.json`, `generation_config.json`, `merges.txt`,
  `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`
- `EleutherAI/pythia-160m` at `50f5173d932e8e61f858120bcb800b97af589f46`:
  `config.json`, `model.safetensors`, `special_tokens_map.json`,
  `tokenizer.json`, `tokenizer_config.json`

`model_download_manifest.json` gives the required byte length and SHA-256 for
every file. The commands verify the complete runtime inventories and fail on a
nearby revision, partial download, extra file, alternate weight format, or
cache metadata. Preserve an initial inventory failure instead of deleting it
from the operator log.

## Exact command sequence

Run from the archive's `abi_release` directory:

```text
abi-reproduce verify
abi-reproduce certify-hosts
abi-reproduce capability-matrix
abi-reproduce causality
abi-reproduce performance
abi-reproduce hostile-audit
abi-reproduce report
```

The sequence verifies bound bytes, performs three new capability-blind host
certifications, executes the 3-environment × 4-capability matrix, reruns causal
and hostile checks, recomputes performance from the new raw files, and produces
one operator report. No source teacher is downloaded or executed.

## Return package

Return the entire `external_reproduction/raw/` tree, the completed signed
operator attestation, `pip freeze`, hardware/runtime inventory, console logs,
and a SHA-256 inventory. Do not copy development timings or certificates into
the operator output tree.

The custodian must independently inspect the receipt before changing the
external gate. A local pass, archive build, or CI run cannot do so. Human review
and `PENDING_AFTER_EXTERNAL_VALIDATION` minimum-information work are separate.
