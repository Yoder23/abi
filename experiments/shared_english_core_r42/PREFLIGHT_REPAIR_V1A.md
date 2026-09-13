# R42 preflight repair V1A

Development V1 completed selection and tokenizer construction, then stopped
before model construction and before optimizer step one. The reused R41 helper
exposes public LayerCake package and host objects but not the trainable
`PortableTokenPlan` constructor, so the trainer's dictionary lookup raised
`KeyError`.

V1 remains immutable. V1A imports `PortableTokenPlan` explicitly from the
already validated LayerCake root immediately after the R41 identity check.
There is no change to the 282 rows, tokenizer, model geometry, seed, optimizer,
6,396-step compute budget, fixture, package contract, scorer, or gate. Execute
into `development_v2`.

