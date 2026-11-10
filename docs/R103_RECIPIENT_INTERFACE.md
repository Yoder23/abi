# R103 recipient and interface suitability

Status: completed 2026-11-10; control qualification failed at 41/48.
The registered recipient/interface comparison was not executed.
`FULL_MOONSHOT_NOT_PROVEN` remains controlling.

R102 did not qualify any English core or support its two registered primary
factors. R103 tests one measured interface difference: the R76 acquisition
used raw prompts followed by a newline, whereas R98-R102 used a Task/Response
wrapper. There is no training, checkpoint selection, optimizer change or new
LayerCake runtime implementation.

Read the [locked protocol](../experiments/recipient_interface_r103/PROTOCOL.md).

## Fixed design

First qualify the pinned Qwen2-7B-Instruct control on 48 new supplied-content
inputs using native chat and raw autonomous outputs. Require at least 44/48,
at least 5/6 per task, zero collapse and zero truncation. Qualification failure
ends this run before recipient comparisons; no substitute model or retry.

If that prerequisite passes, compare both formats on a separate 48-input set:

| Checkpoint | Native interface | Benchmark interface |
| --- | --- | --- |
| Qwen positive control | System/user chat template | Plain Task/Response completion |
| R76 parent | Raw prompt plus newline | Plain Task/Response completion |
| R100 full-model control | Raw prompt plus newline | Plain Task/Response completion |
| R99 LoRA | Raw prompt plus newline | Plain Task/Response completion |
| R99 adapter distillation | Raw prompt plus newline | Plain Task/Response completion |

All recipients retain their exact weights, tokenizer and capability route.
Qwen is an external task-feasibility control, not a size-matched transformer
baseline, transferred English core or teacher resident inside LayerCake.
Differences in scale, precision and prior acquisition preclude architecture-only
causal claims. Within-checkpoint input-format comparisons are paired.

The two primary contrasts are native-minus-benchmark for the parent and the
full-model control, with 10,000-resample paired 97.5% intervals. Other checkpoints
are descriptive controls. A recipient must still pass every unchanged mechanical
quality condition and teacher-relative noninferiority; no screen alone promotes
a fluent product core, broad English or CPU performance.

## Limits and custody

The new inputs reuse known supplied-content task templates, with new identifiers,
day assignments and note order. They are not unseen semantic tasks. Every arm
receives the same explicit 120-word request, fixed before qualification to align
with the existing bounded-response screen. No source-span normalization, answer
prefix, output repair, retrieval or teacher scoring is used.

All input sets, scientific code, tests, source inventory and checkpoint dependencies
are frozen and committed before GPU work. Raw tokens/text, exact prompt rendering,
route traces and resource receipts are retained. The bound is 90 minutes, at most
144 control and 384 recipient generations, zero optimizer steps and 1 GiB new
evidence. Historical failures and unrelated running work are preserved.

Failure to qualify the control is a control/setup finding. If the control passes
but the recipients fail, the limitation belongs to those acquisition/checkpoint/
interface combinations until a separate test localizes a native LayerCake defect.
This diagnostic does not certify or disprove the underlying host architecture.

```powershell
python -m experiments.recipient_interface_r103.run verify
python -m pytest -q tests/test_recipient_interface_r103.py tests/test_r103_completed_verifier.py
```

The verifier is for completed evidence, not permission to overwrite a run.

Software preflight: 20 focused tests passed, and the full review passed 313
tests with four declared skips outside the Windows sandbox. The initial sandbox
run's two temporary-file rename permission failures remain in the
[operational record](../results/review_2026_11_10/r103_preflight_operations.md)
and separate failed JUnit receipt. No model-result claim follows from these checks.

## Completed qualification result

The pinned native-chat control produced all 48 raw responses. Strict verification
returned `CONTROL_QUALIFICATION_FAILED`: 41/48 passed versus the locked 44/48
minimum, and three tasks missed their 5/6 floors. There was zero collapse and
zero truncation. No recipient model was evaluated, no optimizer step ran, and no
checkpoint or native LayerCake source changed.

| Task | Raw passes / 6 |
| --- | ---: |
| Summarization | 5 |
| Rewriting | 6 |
| Email drafting | 4 |
| Conversation | 6 |
| Format control | 4 |
| Tone control | 4 |
| Clarification | 6 |
| Abstention | 6 |

The seven failed rows include three required-literal omissions, two bullet-line
format failures, one overlength response, and one flagged approval contradiction.
The identifier error overlaps one omission. These counts are mechanical proxy
outputs, not independent human judgments of English fluency.

### Evaluator limitation found after the run

The contradiction regex flags `as soon as the details are confirmed` in
`r103-qualification-020-tone_control`. That is a future condition, not an assertion
that approval already exists. This is an observed limitation of the literal
proxy, preserved in an additive regression test. The response may have other
unmeasured adherence issues; no new semantic score is assigned to it.

The frozen evaluator and 41/48 result remain unchanged. Even disregarding that
one flag cannot reach 44/48, and email/format task floors still fail. This
post-run sensitivity observation is not a repaired or promoted quality result.
Do not claim that this control cannot speak fluent English, that recipient
interfaces were compared, or that ABI transfer or native LayerCake failed.

The next bounded proposal should validate the evaluator's semantic boundaries
and establish a genuinely qualified control before resuming recipient attribution.
No new prompt, source substitution, model run or gate change is authorized by R103.

## Identity and costs

- Preregistration: `2c62b74`; successful freeze commit: `aac5f08`.
- [725-input freeze](../results/recipient_interface_r103/freeze.json):
  `709b4c6a5da6802a53ea53d64041d60db0786b984aeceb4cc2d3425c8df8b31a`.
- [Decision](../results/recipient_interface_r103/decision.json):
  `3420c88c6e608b32801bdc144031ac011abaf085af6f6c3861efa4afda1b31bc`.
- [Raw responses](../results/recipient_interface_r103/qualification/control/raw.jsonl):
  `d05ac3f2bd344c87acbd397f631ebb102f30823087d289a620504207e75658bd`.

The external control used 48 requests, 8,690 input tokens and 4,190 generated
tokens (including terminal tokens); unique rendered prompts contain 39,285 UTF-8
bytes and outputs 17,895 bytes. Generation time was 248.01 seconds; the complete
control process recorded 415.51 seconds wall and 408.64 CPU-process seconds,
including identity checks and loading. These are diagnostic costs, not CPU
inference benchmarks or acquisition-efficiency claims.

The 7,615,616,512-parameter control used local GPU FP16, with peak CUDA allocation
15,330,695,680 bytes and sampled peak RSS 2,328,981,504 bytes. Sampled RSS is not
an OS high-water guarantee. Its parameters were not copied into any recipient.
The four inherited 87,710,428-parameter research checkpoints remain bound inputs,
not newly acquired or English-qualified cores. Their previous acquisition costs
are not erased. There are no new checkpoints, imported logits/hidden activations,
trained bridges, or training tokens from this diagnostic.

## Final local assurance

The supported review passed 297 default tests with four declared skips plus
16/16 exact historical Phase 7 tests (313 total). Separately, 12 additive
completed-evidence checks passed in 453.81 seconds: the actual failure recomputed,
ten hostile mutations failed closed, and the observed temporal-regex limitation
was reproduced. Tampered copies were used; the original study files did not change.
The [hostile-test receipt](../results/review_2026_11_10/r103-completed-hostile.xml)
is separate from the original preregistered 20-test scientific surface.

Documentation checks, focused lint and source/wheel builds passed. Package API
version remains `0.4.0a1`. Software success does not qualify the control or fill
the unexecuted recipient matrix. Public reconstruction is reported separately.

The [complete review manifest](../results/review_2026_11_10/r103_manifest.json)
inventories 560 repository assets and all four required inherited LFS checkpoint
paths, with external host bindings recorded separately. SHA-256:
`6a9cf5a7c46d02ea48331828cfd643209624b72b8c9215ebb6ce68c1d96dddc7`.
Unlike an inference replication, raw evidence recomputation needs only the five
listed control tokenizer/config files, not the source's full model weights.

## Versioned public review

The [research prerelease](https://github.com/Yoder23/abi/releases/tag/abi-research-r103-2026-11-10)
is `abi-research-r103-2026-11-10`, source commit
`cc6b59cd8b0b72730a0ae847015c42a9c7a22ab5`, annotated tag object
`db1d4b999e6351b28e552115603516a3d3f1c1fc`. No earlier tag was moved.

Use sibling public checkouts named `abi_release` and `layercake_release`, with
the host at `865a2b9e59e6221caa1185f9b64479d2077953f3`. Windows users should
enable Git's `core.longpaths` only in their review clones. Install the Python
dependencies named in the manifest and obtain the five pinned public Qwen
tokenizer/config files. The [R102 cache setup](R102_COVERAGE_EXPOSURE.md#public-release-and-reconstruction)
uses the same source revision and avoids Windows symlink privileges.

Download `r103_manifest.json` and `review_r103_public.py` from the release outside
the checkouts. Require manifest SHA-256
`6a9cf5a7c46d02ea48331828cfd643209624b72b8c9215ebb6ce68c1d96dddc7`
and helper SHA-256
`11f309d9749bf2099e2084a5aa36b6886c1c7d50aa2397e481130c9455820fb8`.
Hydrate exactly the manifest's four LFS paths, then run the helper:

```powershell
$r103Manifest = Get-Content r103_manifest.json -Raw | ConvertFrom-Json
$r103PayloadPaths = $r103Manifest.lfs_paths -join ','
git -C abi_release lfs pull --include=$r103PayloadPaths --exclude=
python review_r103_public.py --abi-root abi_release --abi-commit cc6b59cd8b0b72730a0ae847015c42a9c7a22ab5 --manifest-sha256 6a9cf5a7c46d02ea48331828cfd643209624b72b8c9215ebb6ce68c1d96dddc7 --output public_r103_reconstruction.json
```

Set `HF_HOME` to the public tokenizer cache before running this command. Existing
receipts are never overwritten. This recomputes raw evidence and runs 32 focused
checks; it does not run a teacher, evaluate a recipient, or train anything.

### Published reconstruction result

Public-only sibling checkouts matched all 560 inventoried assets, including the
four inherited LFS paths, and all 725 frozen bindings. The unchanged verifier
recomputed `CONTROL_QUALIFICATION_FAILED` with the identical decision hash;
32 focused tests passed. Both source checkouts remained clean. GitHub verification
passed on Python 3.10, 3.11 and 3.12 for the sealed source revision.

The [public reconstruction receipt](https://github.com/Yoder23/abi/releases/download/abi-research-r103-2026-11-10/public_r103_reconstruction.json)
has SHA-256
`e224006dfe1a19c0eab10bfb185d41502d5ca181b1bb256a934cf3b1f3f22306`
and verdict `BOUNDED_CONTROL_QUALIFICATION_FAILURE_RECOMPUTED`.

This reused the public-only R102 checkout directories, their already downloaded
public LFS objects and public tokenizer cache, the same physical machine and
installed Python dependencies. The R103 tag and helper/manifest were fetched
from GitHub; no development artifacts were copied into those checkouts. This
was not a brand-new clone, live generation replay, independent hardware or
human quality review. It does not qualify the control or establish recipient
quality. The receipt is additive; the scientific tag and frozen scores are unchanged.
