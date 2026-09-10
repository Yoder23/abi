# R15B pre-existing representation result

Status: `R15B_BOUNDED_PREEXISTING_REPRESENTATION_RECOVERY_PASSED_LOCAL`

R15B is the first ABI experiment to pass a preregistered extraction path from
an unchanged pretrained open-weight source. It advances the acquisition
frontend while retaining the frozen R11 package and recipient boundary. It
does not complete the ABI moonshot.

## Result

Qwen2-7B-Instruct, revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c`, received zero training or
adaptation. A secret selected four distinct mappings between anonymous slots
and three ordinary modular-arithmetic operations. For each mapping, the source
generated six atomic answers. ABI retained the final residual immediately
before the terminal answer plus the eight corresponding frozen output-head
rows.

Each anonymous 200,944-byte representation bundle entered a Linux
pivot-root/no-network capsule containing only the bundle, generic pure-stdlib
decoder, and decoder specification. Prompts, answers, completions, secret
mappings, operation labels, semantic ontology, source model, development tree,
old root, and Windows mount were absent. The decoder recovered all 24 hidden
transition labels with positive margins.

| Gate | Observed |
| --- | ---: |
| Secret capability mappings | 4/4 exact |
| Pretrained source anchors | 24/24 exact |
| Source training steps | 0 |
| Package deep evaluation | 40,000/40,000 exact |
| Worst extraction control | 0.2534 |
| Recipient hosts | Pythia, Qwen2, T5 |
| Recipient live rows | 33,792 byte-exact |
| Recipient optimizer steps | 0 |
| Hostile mutations | 10/10 rejected |
| Strict verification | PASS |
| Live verification | PASS |

The four final R11 packages contain 768 float32 transition values and occupy
8,212 bytes total. The source teacher is absent during recipient execution.
The package hashes are bound separately to a registered semantic namespace and
operation labels; the unchanged R11 package schema itself contains no teacher,
prompt, answer, model, or host payload.

## Information accounting

- 24 raw source prompts, representing six unique prompt strings and 1,164
  unique UTF-8 bytes;
- 5,260 teacher-output bytes and 1,572 generated tokens;
- 86,016 stored residual values (344,064 bytes);
- 114,688 stored output-weight values across four bundles (458,752 bytes),
  representing 28,672 unique transient source parameters;
- 803,776 representation-bundle bytes;
- zero trained bridge parameters;
- zero frozen source parameters in the final packages; and
- 93.59 seconds of measured source inference across the four acquisitions.

Peak CPU RAM and GPU memory were not instrumented and are explicitly recorded
as `NOT_MEASURED`. R15B is not a global minimality result.

## Interpretation and ceiling

R15B proves a bounded pretrained representation-to-portable-package mechanism.
Unlike R15A, it does not first install the capability into the source through a
LoRA learning event. The anonymous secret mappings and physical isolation show
that package selection depends on the supplied source representations.

The result is nevertheless narrow. The arithmetic expressions explicitly
describe their operations, the source-generated reasoning often states the
answer before the captured `FINAL:` position, and the semantic labels come
from a registered external ontology. The decoder therefore extracts a compact
canonical transition from source reasoning; it does not autonomously discover
or label unknown knowledge.

Public Track A evidence also bounds source behavior: the 7B source achieved
24/24 at depth 1, 32/32 at depth 2, and 28/32 at depth 4, but failed the depth
8/12 formulations. The package's perfect deep oracle score is supplied by the
registered affine transition inductive bias, not proof of lossless deep Qwen
behavior cloning.

R15B does not prove English fluency transfer, arbitrary factual/domain
extraction, open-world labeling, production LayerCake ingestion, teacher-level
natural generation, minimal English substrate, or superiority to LoRA or
distillation. Those are the remaining moonshot gates.

## Chronology and evidence

- Implementation freeze: `0d4a75c771cbc46ce2680c81a57ce5ce193acdd0`
- Preregistration: `8a51dae0512d983798bd4b4c3afb0157ce9a6456`
- Reveal: `dafefc97b785d4ed8d3344a5904389560790e2b5`
- Certificate: `results/preexisting_representation_r15b/heldout_v1_certificate.json`
- Strict verification: `results/preexisting_representation_r15b/heldout_v1_strict.json`
- Live verification: `results/preexisting_representation_r15b/heldout_v1_live/receipt.json`
- Hostile audit: `results/preexisting_representation_r15b/heldout_v1_hostile_v2.json`
- Information accounting: `results/preexisting_representation_r15b/heldout_v1_accounting.json`

The result is locally sealed pending a fresh blind review and public clean
reconstruction. R7 remains the controlling published release.
