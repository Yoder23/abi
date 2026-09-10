# R21 fresh live verification

The R21 public candidate is not promoted from stored `PASS` fields. This
separately frozen verifier binds the candidate result, all 1,080 raw rows, all
24 signed cakes, the ABI configs, and the complete Python source inventory of
the LayerCake host used for execution.

In a fresh process it must:

1. recompute every stored row's score and every aggregate;
2. install the exact packages into new registries and reproduce all 1,080 GPU
   outputs byte-for-byte;
3. reproduce one prompt from every task on CPU for every system (54 rows);
4. remove and restore every one of the 24 packages, proving absence rejects and
   restored output is exact;
5. mutate every package in a fresh registry and prove all 24 are rejected;
6. verify signatures, manifests, parameter accounting, exposure, comparisons,
   and all quality gates from bound evidence; and
7. verify the source-model software was never loaded in the live process.

Any missing/stale file, row, hash, package, score, output, gate, or control
fails closed. Passing promotes only the bounded public six-task supplied-data
result and authorizes a fresh hidden replication. It does not prove general
English, autonomous ontology discovery, arbitrary-domain extraction, global
minimality, or the full ABI moonshot.
