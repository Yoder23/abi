# ABI V2 final-mile release report

Status: **TECHNICALLY PROVEN — EXTERNAL VALIDATION PENDING**

Technical declaration: **ABI TECHNICAL MOONSHOT: HOST-INDEPENDENT CORE PROVEN**

This report supersedes `release_report.md` only to correct a non-gating count:
the campaign ran eight random/shuffled corruption controls per host, 24 total.
The original report and its SHA-256 remain preserved in
`release_report_erratum1.json`.

## What passed

- Three capability-blind host certifications passed before capability reveal:
  LayerCake v25, Qwen2.5-0.5B, and Pythia-160M.
- Each host uses one frozen zero-parameter adapter for all four immutable
  packages.
- All 12 host/capability cells passed. English retained 1,381/1,381 frozen
  source successes per host; Python, chemistry, and civics each retained
  100/100 per host.
- All 5,043 receiver outputs were byte-identical to frozen source outputs. For
  300 specialist tasks, all three hosts produced identical action sequences
  (900 host-level specialist results).
- English-only specialist leakage was 0/900. Wrong-capability success was
  0/1,200.
- Adapter removal failed closed. All 12 capability removals failed closed and
  reinstallation restored exact output. All 24 equal-size random/shuffled
  signed-package corruptions—eight per host—were rejected before execution.
- Generic adapter overhead stayed within the preregistered 10% maximum on 20
  observations: LayerCake 3.96%, Qwen -0.55%, and Pythia 3.12%.

## Exact claim boundary

This proves a representation-neutral extension/runtime capability ABI across
the three named hosts. The canonical runtime owns execution of each immutable
capability package, and the frozen host adapter realizes authoritative UTF-8 as
an exact native tokenizer generation sequence. Qwen and Pythia base checkpoints
participate in frozen native conformance probes but do not generate or alter
capability semantics.

No claim is made that LayerCake residual tensors were transplanted into
Qwen/Pythia base weights, that all LLMs are compatible, or that human/external
validation is complete.

ABI V1's 1/3 structural-incompatibility result remains immutable historical
evidence. The initial ABI V2 LayerCake matrix failure also remains preserved;
its sole failed gate was an identical small-file shuffle control, and the
preregistered instrumentation repair changed no semantic output.

## Remaining external gates

1. Human ratings: three real independent raters; 0/21,000 judgments complete.
2. Independent hardware: a separate operator must reproduce the clean-room
   archive on different hardware.
3. Minimum information: the registered search remains frozen and no global
   minimum is claimed.
