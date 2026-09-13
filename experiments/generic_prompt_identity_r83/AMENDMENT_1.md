# R83 pre-observation instrumentation amendment 1

The first invocation of `screen_host_v1.py` stopped in source-evidence preflight
before loading either the parent or candidate model. Candidate validation
outputs observed: zero. Parent validation outputs observed: zero.

The frozen screen compared the stored R81 evidence digest with
`evidence_hash(source_result)` after the digest field had already been inserted.
R81 originally calculated that digest before inserting `evidence_sha256`, so the
screen attempted an impossible self-referential recomputation. The only change
is to copy the R81 result, remove `evidence_sha256`, and hash that unsigned
object. Quality gates, rows, model hashes, training contract, generation,
metrics, and thresholds are unchanged.

The failed binding remains immutable as historical evidence. The candidate
checkpoint remains byte-identical. Its non-weight metadata is reissued solely
to bind this amended screen, protocol, and amendment; a new binding is created
before any validation generation.
