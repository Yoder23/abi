# ABI V2 final-mile release report

Status: **TECHNICALLY PROVEN — EXTERNAL VALIDATION PENDING**

Technical declaration: **ABI TECHNICAL MOONSHOT: HOST-INDEPENDENT CORE PROVEN**

## What passed

- Three capability-blind host certifications passed before any capability reveal: LayerCake v25, Qwen2.5-0.5B, and Pythia-160M.
- Each host uses one frozen zero-parameter adapter for all four immutable packages.
- All 12 host/capability cells passed. English retained 1,381/1,381 frozen source successes per host; Python, chemistry, and civics each retained 100/100 per host.
- All 5,043 receiver outputs were byte-identical to the frozen source outputs and cross-host capability outputs were bit exact. All 900 specialist action sequences were also identical across hosts.
- English-only specialist leakage was 0/900. Wrong-capability success was 0/1,200.
- Adapter removal failed closed; all 12 capability removals failed closed and reinstallation restored exact output; all 24 equal-size random/shuffled package mutations were rejected before execution per host (72 rejections total).
- Generic adapter overhead stayed within the preregistered 10% maximum on 20 observations: LayerCake 0.039615, Qwen -0.005511, Pythia 0.031223.

## Exact claim boundary

This proves a representation-neutral extension/runtime capability ABI across the three named hosts. The canonical runtime owns execution of the immutable capability package, and each frozen host adapter realizes authoritative UTF-8 as an exact native tokenizer generation sequence. Qwen and Pythia base checkpoints participate in frozen native conformance probes but do not generate or alter capability semantics. No claim is made that LayerCake residual tensors were transplanted into Qwen/Pythia base weights, that all LLMs are compatible, or that human/external validation is complete.

ABI V1's 1/3 structural-incompatibility result remains immutable historical evidence. The initial ABI V2 LayerCake matrix failure caused by an identical small-file shuffle control also remains preserved; the preregistered instrumentation repair changed no semantic output.

## Remaining external gates

1. Human ratings: three real independent raters, 0/21,000 judgments currently complete.
2. Independent hardware: a separate operator must reproduce the release archive on different hardware.
3. Minimum information: the registered minimum-information search remains frozen and is not globally certified.
