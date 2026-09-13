# R60 preregistration: prospective frozen-endpoint evaluation

R59 passed two disclosed 1,400-row matrices and exact live replay. R60 is the
single prospective evaluation authorized by that result. The R59 neural
checkpoint, sparse adapters, router, generation rule, lexical postprocessor,
thresholds, evaluator implementation, and gates are frozen before this
catalog is materialized or any source/candidate output is observed.

## Frozen endpoint

- R59 evidence seal: git commit `8526fbd`;
- neural checkpoint SHA-256:
  `4f302650a86964042996db1990e753388c84af7326253088f2a4bbeb698f1be6`;
- checkpoint metadata SHA-256:
  `262fa8994dad29b8ded7039c5ce52a6e32af5ec91e2581b1bce220f06682b4ae`;
- router SHA-256:
  `ba88cf2a18ada898d739aa14a25805dae1c5212ffdcb4bfbfbd99c1c43b682db`;
- candidate decoding: greedy, stop before accepting a sixth identical runtime
  token, then apply the explicitly disclosed inherited R55 lexical repetition
  truncation at threshold 1;
- parent decoding: ordinary R55 greedy decoding with that same inherited
  lexical postprocessor and no R59 circuit breaker; and
- selected capability route only; no planner, symbolic answer path, teacher at
  inference, fallback generation, alternate-token selection, or weight update.

## Prospective surface

`build_catalog_v1.py` deterministically creates exactly 100 new prompts for
each of the 14 frozen English capabilities. It reuses the already frozen
functional evaluator kinds but changes prompt wrappers, content indexes,
probe IDs, and seeds. Exact prompt overlap with every earlier JSON catalog is
prohibited. The catalog and a disjointness receipt must be sealed in a later
commit before source or candidate generation begins.

## Ordered execution

1. Freeze this protocol and catalog-builder implementation.
2. Materialize the catalog, prove its shape and exact-prompt disjointness, hash
   it, and freeze those artifacts.
3. Capture all 1,400 outputs live from the pinned local Phi-3 source using GPU.
4. Execute the unchanged parent and R59 endpoint live using GPU.
5. Recompute all metrics from raw rows and replay the endpoint live.

No endpoint repair, evaluator repair, prompt removal, threshold change, or
replacement prospective catalog is authorized after step 2.

## Pass gates

- exactly 1,400 rows and 100 per capability;
- candidate functional passes at least 1,260/1,400;
- candidate functional count at least the fresh parent count and fresh source
  count;
- at least 65/100 candidate passes for every capability;
- at least 94% retention of fresh source-passing rows;
- paired bootstrap lower 95% bound for candidate-minus-source at least -0.02;
- zero candidate collapse and zero generation error;
- exact route on all 1,400 rows and exactly one selected cake physically
  executed on every model invocation;
- all endpoint artifacts unchanged;
- every parent trajectory that is below the raw token-collapse boundary is
  byte-identical after the common inherited lexical postprocessor; and
- strict raw-row recomputation plus 1,400/1,400 exact live candidate replay.

A pass establishes prospective bounded English capability transfer on this
catalog and machine. It does not alone establish unrestricted English, global
minimum information, human preference, independent-hardware reproduction, or
superiority to LoRA/distillation. The full ABI moonshot remains open until
those separately registered claims are supported.

