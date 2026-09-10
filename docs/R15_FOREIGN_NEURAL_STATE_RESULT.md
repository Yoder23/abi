# R15A foreign neural-state capability result

Status: `R15A_BOUNDED_FOREIGN_NEURAL_STATE_RECOVERY_PASSED_LOCAL`

R15A is the first ABI experiment to pass a preregistered path from a foreign
model's learned neural change into the unchanged R11 portable package and
recipient boundary. It is a material prerequisite for the ABI moonshot, not
completion of it.

## What was tested

Eight fresh hidden affine capabilities were each learned by an ordinary
rank-16 output-head LoRA event on frozen Qwen2.5-0.5B. A generic frontend,
trained and qualified on 320 public events before the held-out reveal, received
only an anonymous 7,168-element effective before/after weight delta. It received
no prompts, behavior rows, answers, capability IDs, operation tables, candidate
program search, or oracle access.

Each extraction ran in a Linux pivot-root/no-network capsule that physically
excluded the Windows development tree, old root, held-out reveal, operation
tables, capability archives, and success IDs. The frontend emitted the six
labels needed to construct an unchanged R11 recurrent-transition package.

The preregistration was frozen at commit
`b92bd2ac6de06f43780abf55ff2b678b91c3c8b0` before the v5 secret was revealed.
V1, v2, v3, and v4 remain additive failed or incomplete executions; none was
retroactively promoted.

## Result

| Gate | Required | Observed |
| --- | ---: | ---: |
| Public frontend development capabilities | 64/64 | 64/64 |
| Held-out source atomic fits | 8/8 | 8/8 |
| Held-out neural-state extractions | 8/8 | 8/8 |
| Package unseen accuracy | 1.0 | 1.0 over 80,000 rows |
| Package counterfactual accuracy | 1.0 | 1.0 over 8,000 rows |
| Delta negative-control maximum | <= 0.30 | 0.265 |
| Recipient AFTER / RESTORED | 1.0 / 1.0 | 1.0 / 1.0 on all three hosts |
| Recipient negative-control maximum | <= 0.30 | 0.17578125 |
| R14 black-box exact recoveries | comparison | 0/8 |
| Strict verification | PASS | PASS |
| Fresh live verification | PASS | PASS |
| Hostile mutations rejected | 9/9 | 9/9 |

The eight content-addressed packages are 2,053 bytes each. They executed with
zero recipient optimizer steps and without the Qwen source on frozen Pythia,
Qwen2, and T5 recipient hosts. Live verification regenerated 19,584 source
rows, reran eight physical extraction capsules, and reproduced all 135,168
recipient observation rows byte-for-byte. Strict verification recomputed every
active recipient neural state from the package and prompt and consumed zero
stored scientific status booleans.

## Interpretation

The result establishes a bounded learned-state decoder: supervised public
meta-training can teach one generic frontend to recognize the structure of a
new capability in a conventional model's parameter update, convert it into a
small canonical package, and execute that package on heterogeneous frozen
recipients. The failed 256-query R14 frontend recovered none of these eight
capabilities, while the weight-delta frontend recovered all eight.

This is stronger than ABI's earlier runtime-only and enumerable-probe results.
It also exposes the next real research boundary: the current source capability
was deliberately inserted by a controlled atomic training event, and ABI used
the before/after delta. The experiment did not discover or extract knowledge
already present in Qwen's pretrained weights.

## Claim ceiling

R15A does not prove:

- extraction of pre-existing English or specialist knowledge;
- discovery, semantic labeling, or English/domain segregation;
- after-weights-only extraction without a before checkpoint;
- teacher-equivalent natural-language generation;
- a minimal fluent English substrate;
- ingestion by a production LayerCake English core or domain layer;
- superiority to LoRA, distillation, or fine-tuning; or
- independent different-hardware reproduction.

The full ABI moonshot remains open. The next valid stage is R15B: test whether
the same frozen package/recipient boundary can recover non-enumerable
capabilities already present in an independently trained source, beginning
with a bounded activation/representation access protocol and explicit semantic
labels. Natural English and domain transfer remain later gates and may not be
claimed from R15A.

## Evidence

- Certificate:
  `results/foreign_neural_state_r15/heldout_v5_certificate.json`
- Strict verifier:
  `results/foreign_neural_state_r15/heldout_v5_strict_verification.json`
- Live verifier:
  `results/foreign_neural_state_r15/heldout_v5_live_verification/receipt.json`
- Hostile audit:
  `results/foreign_neural_state_r15/heldout_v5_hostile_audit.json`
- Protocol and append-only ledger:
  `experiments/foreign_neural_state_r15/`
