# R108 review packet

Current packet: **resource-stop**, not completed qualification. Expected observer
status: `RESOURCE_BUDGET_STOP_NOT_QUALIFIED`. The frozen complete verifier must
reject it. Research tag: `abi-research-r108-resource-stop-2026-11-14`.

The [result page](../R108_TRAINED_GROUNDING.md) is the human entry point. R108
does not certify broad English, ABI superiority, a deployable core, or LayerCake
performance. Read the evidence mode before interpreting any verifier success.

## Pre-inference identities

- Protocol first published in the ancestry of `259cfe98cc3030ef959b4d5ae69fc7fd38f8892e`.
- Input freeze commit: `58d45c2a4bb438b4c08a3e05806cfdefe9d8f55d`.
- Freeze SHA-256: `b8c58cb2d2809df0f92ab4ba321b5b8edaf2821017abdfa43baf531c29f3509d`.
- Native host checkout: `2c3c03fd749616d253798bd33bdb01dc8cc02b01` in a sibling
  directory named `layercake_release`.
- External control: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554` (Apache-2.0).
- Frozen environment: Python 3.10, torch 2.7.1+cu118, transformers 4.57.3,
  tokenizers 0.22.1, safetensors 0.7.0, numpy 2.2.6, psutil 7.0.0.

The source snapshot is 8,060,915,998 bytes. Its exact file hashes and public
Git/LFS identities are in the frozen source receipt. Source weights remain
upstream; they are not vendored into ABI or installed into LayerCake.

## Reconstruct without development assets

Use the final annotated tag named by the public manifest. Clone ABI and
LayerCake as sibling `abi_release` / `layercake_release` directories, with
Git LFS automatic smudging disabled initially. Check out the exact commits.
Pull only the four LFS paths listed by the manifest, not every historical model.
Install ABI's host/test dependencies with the frozen versions above.

Prepare the upstream source cache, without writing into the checked-out packet:

```powershell
python -c "from huggingface_hub import snapshot_download; from experiments.trained_grounding_r108.core import SOURCE, REVISION, SOURCE_FILES; snapshot_download(SOURCE, revision=REVISION, allow_patterns=list(SOURCE_FILES))"
```

The public manifest pins required repository files, four retained checkpoints,
native host dependencies, and the external source files. Run the published
review helper from the tagged public checkout, using the release's manifest
hash and exact commit, with a new receipt path outside both repositories:

```powershell
python scripts/review_r108_public.py review --root PATH_TO_PUBLIC_ABI --commit TAG_COMMIT --manifest-sha256 RELEASE_MANIFEST_SHA256 --output NEW_EXTERNAL_RECEIPT.json
```

This is raw-evidence recomputation, not another live generation run or
independent-hardware validation. The helper checks origins, clean checkouts,
payload hashes, the declared evidence mode, focused hostile tests, and docs.

## Two deliberately distinct verification paths

For a **completed** experiment, the frozen command is:

```powershell
python -m experiments.trained_grounding_r108.run verify
python -m pytest -q tests/test_trained_grounding_r108.py tests/test_r108_completed_verifier.py
```

It requires all registered cells and fresh-process replays, except that a fully
recorded negative control qualification closes the recipient gate. Missing raw
rows, receipts, hashes, or replay cannot qualify.

For a **resource-stop** packet, the additive command is instead:

```powershell
python -m experiments.trained_grounding_r108.interruption
python -m pytest -q tests/test_trained_grounding_r108.py tests/test_r108_interruption.py
```

That observer requires the original budget timeout and intact raw prefix. It
reports `RESOURCE_BUDGET_STOP_NOT_QUALIFIED`; public reconstruction reports
`REPRODUCED_RESOURCE_STOP`, not a scientific pass. The frozen complete verifier
must continue to reject the experiment as interrupted. No partial pass rate is
promoted and no recipient/acquisition conclusion is drawn.

Do not rerun the write-once `pilot`, `freeze`, or `prepare` commands into the
published evidence directory. A future authorized live retry needs a separate
evidence namespace and must retain the interrupted attempt unchanged.
