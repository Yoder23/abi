# R21 label-separated generative transfer result

## Verdict

`PUBLIC_PREREQUISITE_PASSED_HIDDEN_REPLICATION_FAILED`

R21 establishes a bounded public result and then fails its fresh hidden
replication. It does not establish general English extraction or the ABI
moonshot.

## Design

R21 replaced R20's exact-template compiler with a generative, LayerCake-native
portable token-plan representation. It compared three methods on the same 600
teacher-response rows and the same 60,000 row exposures per method and seed:

1. raw sequence distillation into one monolith;
2. label-normalized distillation into one monolith; and
3. six independently signed ABI factors with selected-only execution.

All methods used seeds 21021, 21022, and 21023. The source was pinned
Qwen2-7B-Instruct revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c`. The source is absent at package
execution, no source parameters are copied, and receiver training after
package installation is zero.

## Labeling evidence

Teacher-side self-labeling is not certified:

| Interface | Exact labels | Verdict |
|---|---:|---|
| Free label generation | 583/600 | failed 594 gate |
| Restricted A-F next-token scoring | 534/600 | failed 594 gate |

The restricted-score failures were 66 prose rows classified as summary or
bullets; all other tasks were 500/500. This branch is closed.

To isolate realization from teacher self-labeling, a registered ABI word
classifier was fitted from 36 unique public instruction/label seed pairs. It
scored 600/600 on extraction instructions and 120/120 on the public instruction
split. That is supervised registered-ontology labeling, not autonomous
ontology discovery.

## Public result

| System | Functional results / 120 |
|---|---|
| Teacher | 96 |
| Raw sequence, three seeds | 73, 80, 71 |
| Labeled monolith, three seeds | 95, 94, 99 |
| ABI factorized, three seeds | 116, 116, 116 |

For the preregistered headline seed, ABI minus teacher was `+0.1667` with 95%
paired bootstrap CI `[+0.1000, +0.2417]`; ABI minus labeled monolith was
`+0.1833` with CI `[+0.1083, +0.2583]`; ABI minus raw sequence was `+0.3000`
with CI `[+0.2167, +0.3833]`.

The factorized system has 525,401 deployed parameters across six packages and
96,216 active parameters for one selected task, versus 558,331 deployed and
active parameters for the labeled monolith. Total factor package size is
2,171,001 bytes per seed.

Fresh verification reproduced 1,080/1,080 GPU outputs byte-for-byte, matched
54/54 task-covering CPU outputs, removed and restored all 24 packages, rejected
all 24 targeted tensor corruptions, verified all signatures, and loaded no
source-model software. Strict recomputation passed without trusting stored
scientific booleans.

## Preserved assurance failures

The complete evidence follows four preserved operational/verifier failures:

- a self-referential evidence-hash recomputation defect;
- a relative/absolute package custody-path mismatch;
- omission of the LayerCake direct-decoder input-contract mode; and
- an ineffective final-ZIP-byte mutation that touched ignorable metadata.

Each repair was additive, separately frozen, and limited to the measured
interface. No data, weights, seeds, quality thresholds, or LayerCake source
were changed.

## Hidden replication

The hidden generator and gates were committed before a random 256-bit seed was
revealed. The pinned source then produced 120 new responses. The existing 24
packages executed without retraining.

| System | Functional results / 120 |
|---|---|
| Teacher | 107 |
| Raw sequence, three seeds | 49, 51, 32 |
| Labeled monolith, three seeds | 92, 95, 98 |
| ABI factorized, three seeds | 113, 113, 113 |

The ABI factors were perfect on labeling, adherence, numeric hallucination,
repetition collapse, and CPU/GPU execution, and passed five of six task minima.
Summary was 13/20 for every seed, below the frozen 15/20 requirement, so the
result is `FAIL_HIDDEN_REPLICATION`. The 6/120 functional advantage over the
teacher had a paired 95% bootstrap interval of `[-0.0083, +0.1083]`, which
includes zero.

The seven summary failures are semantically plausible paraphrases such as
"showed stability" for "remained stable" and "latency decreased" for "latency
fell". That identifies a possible lexical-oracle limitation, but it does not
revise the frozen verdict and cannot promote R21 post hoc.

## Claim ceiling and next work

R21 proves a bounded public teacher-response-to-signed-LayerCake generative
package mechanism with registered ontology labels and real CPU/GPU execution.
It does not prove hidden replication, unrestricted English fluency, autonomous
capability discovery, arbitrary-domain extraction, global minimality, or
superiority to LoRA/distillation.

A successor must be separately registered. It should use a materially broader
semantic-plan representation and a proposition-level evaluator frozen before
a new hidden selection. R21's hidden seed, rows, and thresholds must not be
rerun or edited.
