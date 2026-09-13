# R67 preregistration: compositional-reasoning coverage acquisition

## Measured bottleneck

R64/R66 isolate domain-independent nonce reasoning as the largest adjudicable
transfer deficit: the live Phi source passed 82/100 while R63 passed 8/100 and
R66 passed 23/100. The immutable broad training archive contains only 100
unique reasoning records, versus thousands for several surface capabilities.

R67 expands only this measured coverage deficit. It creates 2,000 unique,
search-only, domain-free transitive-rule prompts with invented symbols, varied
chain depths, varied natural phrasings, and deterministic semantic evaluators.
No prompt contains real-world or specialist facts. Every row is labeled
`english_core`, `domain_independent_reasoning`, `english_linguistic_form`, with
empty domain labels and claims.

The catalog builder and shape verifier are committed before materialization or
source inference. The pinned local `microsoft/Phi-3-mini-4k-instruct` revision
`f39ac1d28e925b323eae81227eaba4464caced4e` is then surveyed on CUDA. Only live
source-passing search rows may enter a v3 segregated training archive. Teacher
tokens, bytes, runtime, model identity, hardware, provenance, and bundle hashes
must be accounted. No candidate is loaded during acquisition.

## Decision rule

At least 1,800/2,000 source rows must pass with zero source collapse for this
archive to authorize one new LayerCake training successor. Failure closes this
catalog design. R67 is training-material evidence only: it cannot promote a
host or certify the ABI moonshot.

