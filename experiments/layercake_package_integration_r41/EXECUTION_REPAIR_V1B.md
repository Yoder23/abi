# R41 execution repair V1B

Execution V1 built and strictly installed all twelve packages, then failed at
the first CPU module load before generating a scientific output. The frozen
state contains `target_position.weight` with shape `[384, 64]`; the adapter
incorrectly declared `maximum_target_actions=96`, producing a `[96, 64]` host
tensor. Strict LayerCake loading rejected it.

Execution V1 and its twelve rejected packages remain immutable under
`results/layercake_package_integration_r41/execution_v1/`.

V1B changes only geometry declaration and the live generation ceiling:

- derive source/target position counts directly from the frozen checkpoint;
- use the checkpoint-authoritative 384 maximum actions in host calls.

No source component, tensor, tokenizer, package schema, fixture, task, scorer,
device, gate, or threshold changes. Execute into `execution_v2`.

