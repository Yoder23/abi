# R109 research review packet

This packet reproduces a bounded trained-task experiment, not a fluent-English
product certificate. The [result page](../R109_TRAINED_GROUNDING.md) is the entry
point. A successfully recomputed negative result remains negative.

The actual result is `CONTROL_QUALIFICATION_FAILED`: panels 35/48 and 34/48,
paired 32/48 against 44; exact 96-row replay, no collapse/truncation, no recipient
or training run. The additive scope audit documents seven problematic approval
flags; even granting every flagged row a pass still fails the original gates.

## Fixed scientific identity

- Preregistration published as `1ead549` before the input freeze.
- Input freeze published as `c302bc4` before any R109 model call.
- Freeze SHA-256: `88149d5be624ba13072659b2d2b70855b2d14125211fe5700b32bda905b1bc1e`.
- Exact host: `763e6da3cca1aa7564e6bce4b64dbee4effa552d`, in a sibling
  `layercake_release` directory. This is not a new native-host certification.
- Control: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`; Apache-2.0.
- Environment: Python 3.10, torch 2.7.1+cu118, transformers 4.57.3,
  tokenizers 0.22.1, safetensors 0.7.0, numpy 2.2.6, psutil 7.0.0.

R108 remains unchanged and incomplete. R109's population, author-fixture audit,
matched-training provenance and source receipt are byte-identical to R108.
The population is reused, not a newly blind holdout. Hardware operating state
differs between attempts without an agent setting change; no controlled speed
comparison between R108 and R109 is claimed.

## Reconstruct the published packet

Use the final annotated tag and exact commit named by the public manifest.
Clone ABI and LayerCake as sibling `abi_release` and `layercake_release`
directories. Disable automatic Git LFS smudging for the initial checkout, then
pull only the four checkpoint paths listed in the manifest. Do not download
all historical weights. Install the frozen dependencies.

The pinned source remains upstream, not vendored in ABI or LayerCake. Prepare
its cache without writing into the evidence packet:

```powershell
python -c "from huggingface_hub import snapshot_download; from experiments.trained_grounding_r109.core import SOURCE, REVISION, SOURCE_FILES; snapshot_download(SOURCE, revision=REVISION, allow_patterns=list(SOURCE_FILES))"
```

Run the helper from the exact tagged checkout, writing a new receipt outside
both repositories:

```powershell
python scripts/review_r109_public.py review --root PATH_TO_PUBLIC_ABI --commit TAG_COMMIT --manifest-sha256 RELEASE_MANIFEST_SHA256 --output NEW_EXTERNAL_RECEIPT.json
```

It verifies clean public origins, all manifest assets, source payloads and host
bindings, strictly recomputes the raw/replay result, runs hostile packet tests,
and checks documentation. Success is `REPRODUCED_COMPLETE_BOUNDED_RESULT`, not
an English-quality or moonshot pass. Missing evidence must fail closed.

Direct checks:

```powershell
python -m experiments.trained_grounding_r109.run verify
python -m pytest -q tests/test_trained_grounding_r109.py tests/test_r109_completed_verifier.py
```

Never run `pilot` or `freeze` over the published packet. A new live run needs
separate authorization, an evidence namespace and unchanged prospective gates.
Public raw recomputation does not replace live replay, human ratings or
independent different-hardware validation.
