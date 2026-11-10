# R102 public-review setup observations

Date: 2026-11-10. These are operational observations, not scientific gates.
The R102 experiment, freeze, decision and annotated tag remain unchanged.

## Fresh public checkouts

New sibling directories were created beneath
`C:/Python310/layercake_merged_nextgen_perfectA_option1_full_ready/_public_r102_20261110/`.
Both repositories were cloned from their public GitHub remotes with
`--filter=blob:none --no-checkout`; no development Git objects, evidence or
checkpoint files were copied. Git LFS smudging was disabled for checkout so the
ABI payload download could be restricted to the complete public manifest.

The first LayerCake checkout reported `Filename too long` for nine historical
artifact paths beneath `artifacts/moonshot/phase2_recertification/`. It still
returned exit code zero and left nine files missing, so that exit code alone
was not treated as successful reconstruction. The exact missing paths were
filled from the same pinned public Git objects using Git's long-path option.
Plain `git status` also required that option to inspect them. Setting
`core.longpaths=true` only in this new clone then produced a clean status.
The ABI clone used `--config core.longpaths=true` from creation. No developer
checkout or global Git setting was changed.

## Fresh source-tokenizer cache

An initial five-file `snapshot_download` into `hf_cache` failed with Windows
`WinError 1314` during symlink creation for the pinned Qwen `config.json`.
The incomplete cache was retained, not used as successful evidence.

A separate `hf_cache_copy` directory used the library's supported `local_dir`
download mode, which writes ordinary files without symlinks. Only these public
files were downloaded from `Qwen/Qwen2-7B-Instruct` revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c`:
`config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`.
They were placed in that fresh cache's pinned snapshot location. An offline
snapshot lookup succeeded. The frozen verifier independently checks their hashes
and token/text identities; no teacher weights or development tokenizer assets
were copied. The public-review process sets `HF_HOME` to this separate cache.

## Interpretation

These Windows setup failures and their operational remedies are not ABI learning
failures, native LayerCake defects, successful English acquisition or changes to
scientific thresholds. Public evidence recomputation has its own additive JSON
receipt. It uses the same physical machine and installed Python dependencies,
not an independent operator/hardware environment or new live generation.

## Initial GitHub CI dependency failure

The first public verification workflows for the scientific tag and additive
helper commit failed. Run `42865966979` recorded 113 passing tests and one
failure on Python 3.11: `test_merged_checkpoint_does_not_change_live_lora` could
not import `transformers.pytorch_utils.Conv1D`. The workflow installed only the
`host` extra, which supplies Torch but not Transformers or NumPy. Local testing
had those research dependencies installed.

The additive workflow correction installs `transformers==4.57.3` and
`numpy==2.2.6`, matching the registered research environment. It does not remove,
skip or change the frozen test or alter scientific files. The original failed
GitHub runs and tag remain intact; a subsequent workflow must pass separately.
