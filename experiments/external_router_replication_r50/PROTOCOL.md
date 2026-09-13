# R50 preregistration: untouched final-test replication

R49 passed development validation and exact live replay.  R50 freezes that
complete system and evaluates the catalog's pre-existing but previously
unevaluated `final_test` split.  No model, router, decoder, prompt, evaluator,
or threshold may change.

Frozen system:

- neural checkpoint SHA-256
  `65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b`;
- router package SHA-256
  `20ae6562432cc59ece5b9cb552caaa16da4824323e62460c6c33feb0579df863`;
- R49 result SHA-256
  `a9014281204b1b1c05be8df28b414c2df1c2a524b2a223609aa8fc31275446a1`;
- catalog and source hashes are unchanged from R49;
- split: exactly 100 `final_test` prompts for each of 14 capabilities;
- external hashed router, forced canonical LayerCake route, neural greedy
  generation, and lexical repetition threshold 1; and
- zero training or teacher execution.

Pass requires 1,260/1,400 functional, at least 65/100 per capability,
candidate count at least stored source count, at least 94% source-pass
retention, zero collapse/error, exact routing, physical one-cake execution, and
unchanged artifacts.  Strict recomputation and a live replay are required
before certification.

This is a held-split replication inside the same machine and catalog lineage.
It does not substitute for new task families, live-source comparison, human
ratings, independent hardware, minimization, or LoRA/distillation baselines.
