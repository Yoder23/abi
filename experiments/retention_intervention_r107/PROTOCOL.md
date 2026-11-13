# R107: natural-text retention and residual intervention

Registered 2026-11-11 before R107 model execution. This materially different,
zero-training diagnostic does not reopen the failed R103-R106 qualification
screens, qualify an English control, or lower any product gate. Their outcomes
are immutable. Native LayerCake implementation remains unchanged.

## Question

R101 tested execution consistency and oracle prefixes on acquired tasks. It did
not test natural-text retention or intervene on the task residuals. R47 already
copied a DistilGPT2-derived six-block capacity control; repeating that construction
would not be a breakthrough. Here ask whether the current recipient's learned
residuals interfere with its inherited natural-text prediction, before spending
more compute on acquisition. This is an ABI research-lineage diagnostic, not a
test of the separately sealed LayerCake product or a new artifact interface.

## Fixed population and cells

Use cached Salesforce/wikitext revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, configuration wikitext-103-raw-v1,
validation parquet. Preserve the complete source and attribution. Select unique
non-heading lines with 128-512 GPT2 tokens, ordering by SHA256 of
`r107:107001:<original row index>:<unmodified text>`, first 64. Score tokens 33-128
conditioned on the first 32 tokens. This is deliberate fixed-window measurement,
not silent context truncation. Preserve original text, indices and token IDs.
No population, route, interface or model selection may depend on outputs.

Nine cells: unchanged upstream DistilGPT2
(`2290a62682d06624634c1f46a6ad5be0f47f38aa`), and active/neutral residual
conditions for each of R76 parent, R100 full-model, R99-FP32 matched LoRA and
adapter-distillation checkpoints. Raw text only, no instruction/chat wrappers;
fixed coherence route 1 for all recipients. All tokenizer encodings must match.
The source is an ancestral reference, NOT a newly qualified instruction-following
control. LoRA/distillation retain their historical matched budgets; this test
does not create a new training comparison or equate them to the source's budget.

Neutralization zeros ONLY the 14 final-cake and 84 deep-adapter `up.weight`
tensors in memory. Record each pre/post tensor digest, verify all other state
unchanged, and never save altered model weights. Merged LoRA/trunk changes remain.
This intervention removes residual effects, not compute, and is NOT a deployable
candidate, sparse speed optimization or a new acquisition method.

## Measurements and registered interpretation

Record selected target logits, log normalizers, argmax IDs, target IDs and exact
target UTF-8 bytes; independently cross-check NLL against cross entropy. Primary
contrast: parent ACTIVE minus NEUTRAL BPB per passage, paired 10,000-resample
percentile bootstrap, seed 107001. A strictly positive 95% lower bound supports
residual interference on THIS natural-text population. Otherwise report no
support. Other three paired contrasts and source-relative losses are descriptive.
Report token-weighted NLL/perplexity, byte-weighted BPB and passage counts.
Do not substitute mean passage BPB for aggregate BPB.

On the first 16 selected passages, generate at most 64 tokens greedily from the
same 32-token prefix, stopping on EOS; no rescue/repetition processor. Publish
all text/IDs, termination and repeated-4-gram fraction. These are descriptive
continuations, not automatic fluency ratings or qualification. WikiText includes
facts/domain content; its likelihood is NOT a pure-English or purity measure.
Its validation split is held out by the dataset, not proven unseen to any
pretrained/checkpoint lineage. Contamination and prior inspection remain possible.

Repeat every cell in a fresh process. Require exact raw JSONL replay, including
token evidence, before any diagnostic conclusion. Missing inputs, hashes, rows,
replays, receipts, costs, invalid routes or nonfinite evidence fail closed.
Recompute aggregates and contrasts; never accept saved scientific booleans.

## Custody and bounds

Freeze code, protocol, tests, all ABI runtime Python, LayerCake runtime Python,
source/tokenizer files, recipient files and external corpus before inference.
Commit the registration and freeze before GPU work. FP32, eval, seed 107001,
four CPU threads, TF32 disabled; one cell per process, no parallel model jobs.
Maximum 20 minutes/cell, 90 minutes cumulative, 512 MiB new evidence, initially
at least 20 GiB free. Record wall/CPU time, sampled RSS, CUDA peak, versions,
parameters and machine. Existing checkpoints are rehashed; zero optimizer steps,
new acquisition prompts, trained parameters or capability packages. Count all
evaluated target tokens/bytes and stored scalar logits/normalizers separately.
Upstream DistilGPT2 is itself distilled and retains all six transformer blocks;
this reference is not novel ABI, compressed extraction, teacher-free promotion,
or superiority over distillation. No new model download is required.

One complete audit, then publish positive AND negative evidence. A positive
intervention can motivate a separately registered retention-constrained learning
experiment, but does not authorize an automatic training/sweep follow-on. Keep
ABI acquisition, ABI integration, native-host correctness and product qualification
distinct. No CPU/GPU advantage, package certification or full moonshot follows.

Sources: [DistilGPT2 model card](https://huggingface.co/distilbert/distilgpt2)
(Apache-2.0); [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
(CC-BY-SA-3.0/GFDL). Source text is a measurement input, not newly acquired core data.
