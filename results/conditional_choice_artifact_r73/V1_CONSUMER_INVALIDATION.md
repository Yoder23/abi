# R73 v1 consumer invalidation

The immutable v1 archive
`reasoning-weight-selected-search-v1.abix` is structurally and cryptographically
valid, but its selection object used the source-survey purpose and requested no
English capabilities. The complete LayerCake training loader therefore
correctly rejects it even though the lower-level row materializer accepts its
individual evidence bindings.

The v1 archive and its verification are preserved as historical evidence and
are not authorized for training. A v2 archive must declare the exact
`domain_independent_reasoning` English capability subset and must pass the full
`load_english_training_rows` consumer, not only the private row materializer.

This invalidation changes no source evidence and authorizes no training from
v1.
