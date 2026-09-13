# R59 disclosed endpoint certificate

Status: **PASS on both disclosed development matrices; prospective promotion is
not yet earned. The full ABI moonshot remains OPEN.**

R59 freezes the R55 nonlinear sparse-adapter checkpoint and R53 router. It adds
no trained parameters and makes no logit substitution. Before accepting a
greedy token, the runtime terminates if that token would create a sixth
identical-token run. After decoding, it retains the inherited R55 lexical
truncation at threshold 1. That inherited postprocessor was missing from the
historical R55 metadata and is now disclosed explicitly.

## Disclosed results

| Split | Candidate | Source | Retention | Collapse | Route / rows | Verdict |
|---|---:|---:|---:|---:|---:|---|
| validation | 1,277 / 1,400 | 1,220 / 1,400 | 95.164% | 0 | 1,400 / 1,400 | PASS |
| final test | 1,262 / 1,400 | 1,154 / 1,400 | 94.367% | 0 | 1,400 / 1,400 | PASS |

Every capability is at least 65/100. Exact physical sparse execution is
1,400/1,400 on each split. Training steps and teacher calls during both screens
are zero. The teacher is absent at candidate inference.

The parity audit establishes 1,400/1,400 byte-identical validation outputs
against R55. On final test, 12 outputs changed, no functional verdict changed,
no new collapse was introduced, and all three previously detected collapses
were eliminated. The remaining nine changes interrupted raw degenerate runtime
token loops that the inherited lexical postprocessor had hidden from the
historical re-tokenized collapse metric.

## Immutable evidence

- checkpoint: `4f302650a86964042996db1990e753388c84af7326253088f2a4bbeb698f1be6`
- checkpoint metadata: `262fa8994dad29b8ded7039c5ce52a6e32af5ec91e2581b1bce220f06682b4ae`
- router: `ba88cf2a18ada898d739aa14a25805dae1c5212ffdcb4bfbfbd99c1c43b682db`
- validation result: `154d723c6227b1dbf3b99a44e70b786906218418544c9c49a8516ad98248848a`
- validation raw rows: `056bfe0516c59820cf8d4a4ad58d3efb702aac9dd59c987b4f2e26ece20c1934`
- validation live receipt: `dfe480013374b7937578595cf87138afbf4ad7fe3833651a22769c55937b3aa1`
- final-test result: `67fb70bc357b583eb1b1aa4e540b4e5f4ecf358c0b5e7afbda10637e21550343`
- final-test raw rows: `063ccf9b89ab1a14f559433773579256515ff55cdeeaebd658bd6ed7c5ef85f1`
- final-test live receipt: `0ded909733ac4052d97ae61f1eda5f38370dc11c8807727b856d6b57346cd08b`
- parity audit: `abb5465b4396d83967d7db64c5f2aba06300b3c3cfd608eb0ced6fde286d01c4`

The strict verifier recomputed all 2,800 raw rows and then performed 2,800
fresh live neural executions. Both live receipts report 1,400/1,400 exact.

## Scope boundary

This certificate authorizes exactly one post-freeze prospective 100-prompt-per-
capability evaluation and new live pinned-source capture. It does not establish
unrestricted English, global minimality, human preference, independent
hardware reproduction, or superiority to LoRA or distillation.

