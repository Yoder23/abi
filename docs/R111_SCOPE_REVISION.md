# R111 — scope coverage and unconditional revision

Status: preregistered design; implementation and execution pending. No model
calls, gate changes or GPU setting changes. [R110](R110_SCOPE_ADHERENCE.md) remains
an immutable negative prerequisite result, not an ABI acquisition or host test.

The [protocol](../experiments/scope_revision_r111/PROTOCOL.md) separates a new
800-case finite parser challenge from one conventional draft/revision comparison.
The revision always receives its own draft and original request, not scorer
feedback or a reference answer. Both stages retain the same quality gates.
The extra source call and larger 2048-token input envelope are explicitly
accounted for; no one-pass fluency, acquisition-efficiency or novelty claim.

The actual challenge must be published after the implementation freeze and
before scoring. A failed challenge closes GPU execution. If it passes, two
fresh source processes cover 96 paired prompts, draft + revision, with exact
replay (384 calls maximum). No training, recipient execution, native LayerCake
change or automatic follow-on. All existing matched controls are preserved.
