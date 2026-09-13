# R84 pre-observation metadata amendment 1

The single R84 training run completed 6,000 optimizer steps and wrote its
checkpoint. The additive finalizer then stopped before candidate binding or
validation generation because the sealed prompt-pointer trainer emitted
`capability_cake_expansion: null` instead of carrying forward the unchanged
R78 parent topology. Candidate validation outputs observed: zero. Parent
validation outputs observed: zero.

The checkpoint contains the inherited capability cakes and all 84 deep-adapter
tensors, and its recorded state hash excluding the new pointer equals the
pre-training R78 state hash exactly. This amendment copies the already frozen
R78 expansion ledger into candidate metadata, records the correction, and
recomputes only the metadata manifest. It does not change candidate weights,
training, generation, gates, screen logic, or thresholds. The extension runner
is corrected so a reproduction performs this inheritance before finalization.
