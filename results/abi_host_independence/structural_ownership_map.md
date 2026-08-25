# Structural ownership map

ABI V1 failed on Qwen and Pythia because executable tensor schema, tokenizer,
vocabulary, residual coordinates, normalization, position, output head, and
lifecycle were co-defined with LayerCake. Equal width did not imply equal
meaning.

Host width, residual/embedding/output bases, tokenizer/vocabulary, positional
encoding, and normalization are host-owned. Typed context, sequence, precision,
lifecycle, fusion, and installation semantics are canonical-ABI-owned. The
frozen package owns its private routing and LayerCake-era implementation fields;
those fields cannot escape its authenticated runtime class.

The architectural consequence is Family A: standardize observable meaning and
exact anchors, not hidden coordinates. The adapter maps a host to that boundary
once. No residual projection, tokenizer transplant, or per-capability receiver
fit is permitted.
