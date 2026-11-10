# R102 pre-execution construction amendment A1

Registered 2026-11-10 before a successful input freeze, teacher call, optimizer
step or model-quality observation. Preserve the original protocol and failed
sampler at `b647de0`, and its recorded diagnosis at `1fcefbb`.

The initial joint random pairing failed at the second variant of one example,
after 317 completed aliases. None of its 4,096 recorded pairs matched both token
lengths. Yet the same pool contained 46 admissible project-ID components and
69 admissible person-ID components. Combining recorded proposals 40 and 144
preserved prompt/target lengths (119/44), all text under inversion, and the
mechanical screen. This is evidence of a construction sampler bottleneck, not
a model failure or a positive acquisition result.

Before creating the first successful R102 freeze, replace joint-pair rejection
with component-profile selection. Draw from the same seeded proposal stream,
at most 4,096 tuples per original row/variant. Select the first unused NOTE
component and first unused co-referent Person component that each preserve the
original full prompt and target token lengths. Combine them, then independently
recheck full lengths, original-screen equality and exact inverse text/row
identity. Record both proposal indexes. No extra random pool, changed token
budget, identifier-overlap relaxation or model-based selection is allowed.

The failed preflight remains separate immutable evidence. This is a documented
pre-freeze implementation repair, not a rerun of a measured learner or a
post-result scientific gate change. The original joint sampler remains in code
for reproducing the failure; the new freeze explicitly selects the profiled
constructor. If this constructor fails, preserve that preflight too.

All scientific factors, hypotheses, teacher/learner budgets, seeds, evaluation
inputs, stopping gates, bootstrap rules, methods, optimizer settings, target
wording and claim limits in PROTOCOL.md remain unchanged. Publication still
awaits the user's privacy preference. No automatic training extension follows.
