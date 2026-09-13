# R59 v1 rejected despite numeric gates

R59 v1 reported 1,278/1,400 functional, zero final collapse, exact routing,
and exact sparse model execution. It is nevertheless rejected because the
runtime invariant recorded 22 boundary-token rejections on a validation split
where ordinary R55 had zero final collapsed rows. This violates R59's stronger
preregistered claim that every naturally non-collapsed trajectory remains
byte-identical.

The false interventions came from applying the lexical regex to a decoded
prefix whose final BPE-composed word was not yet closed. The additive v1a
repair is documented in
`experiments/irreversible_collapse_invariant_r59/LEXICAL_BOUNDARY_REPAIR_V1A.md`.
No final-test was run from v1.

Raw evaluation SHA-256:
`ef1c41303014b3c9bafaa2f8cf58169f925eac25ab926841a9f61793acc8aff3`.
Result file SHA-256:
`8ccce5c39bc6c6eb84736c0e16cec19ee8ee46684ea21dbf38c67d7468590875`.
