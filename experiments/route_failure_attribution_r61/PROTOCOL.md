# R61 preregistration: prospective route-failure attribution

R60 failed with 90/100 format-control prompts routed incorrectly and no other
route errors. R61 changes only those 90 route selections to the catalog's
already sealed capability label. It changes no prompt, checkpoint, adapter,
decoder, evaluator, source row, or threshold and performs no training.

The remaining 1,310 rows reuse R60 outputs only because the selected route is
already identical; their byte-exact live replay is bound by R60 receipt
`9e9070d820ed241d6c6f14590110338c38914a9935840c0817c864e64a513db2`.
Every changed row is executed live with the same frozen R59 endpoint.

The result is diagnostic and cannot promote a system. It measures the exact
functional and final-collapse recovery attributable to corrected capability
selection. If format control or the total score remains far below the R60
gates, routing is not the limiting cause and prompt/payload invariance must be
tested separately. The full ABI moonshot remains OPEN.

