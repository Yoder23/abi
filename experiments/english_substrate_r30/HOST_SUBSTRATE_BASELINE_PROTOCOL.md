# R30 v9 frozen LayerCake host-substrate baseline

R30 v7 and its compute-matched v8b sequence-distillation control both failed
the absolute functional gate.  Before changing ABI acquisition again, v9
separates the recipient substrate from the ABI frontend.

V9 loads the already certified LayerCake Phase 2 English checkpoint at
`layercake_release/artifacts/moonshot/phase2_shallow_sparse_pretrained/student2400-seed-9824`.
The checkpoint and tokenizer are read-only and hash-bound.  No ABI package is
installed, no Qwen output is used for decoding, and no parameters are trained.
The source-discovered R30 route is mapped to the matching frozen LayerCake task
route.  Generation is live greedy cached decoding with the existing
deterministic repetition controls.

The same 144 R30 evaluation prompts and v7 semantic scorer are used.  Results
are a host-substrate diagnostic only:

- a pass at 132/144 with at least 10/12 per task and 12/12 abstention means the
  frozen host is already sufficient and ABI should focus on a tiny conformance
  package;
- a partial result identifies which behavior must be supplied by an artifact;
- a failure does not invalidate R29 labeling or R30-v3 neural diagnosis and
  must not be reported as an ABI extraction failure.

This disclosed diagnostic cannot certify general English or the ABI moonshot.

