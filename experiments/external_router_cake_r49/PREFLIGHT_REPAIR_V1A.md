# R49 v1a preflight repair

The first R49 launch fitted and validated the router, then failed closed before
generating any integrated row.  The code incorrectly required the union of the
two frozen source bundles to equal the selected catalog IDs exactly.  The
bundles contain all 1,400 selected IDs plus 200 historical v2 abstention and
coherence IDs.  Missing selected IDs were zero.

V1a changes only that set assertion.  It now requires the selected catalog IDs
to be a subset of the immutable source inventory, fails on any missing selected
ID, and records the count and SHA-256 of the sorted surplus IDs.  The catalog,
model, source bundles, router representation, fit schedule, decoding, gates,
and evidence schema are otherwise unchanged.  The failed launch produced zero
integrated generation rows and is preserved at
`results/external_router_cake_r49/development_v1/`.
