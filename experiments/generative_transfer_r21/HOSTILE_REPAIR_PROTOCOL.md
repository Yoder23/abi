# R21 targeted hostile-control repair

The first fresh live verifier executed all nine systems but failed its lifecycle
gate because flipping the last byte of a tar archive modified only padding. A
fresh installer correctly accepted the unchanged signed members. No raw replay
rows were persisted and the candidate remained pending.

The failed record called the archive a tar stream; byte inspection establishes
that these `.cake` files are ZIP archives. This additive repair may change only
the mutation location. It restores the last padding/comment-length byte
supplied by the frozen verifier, locates the uncompressed
`tensors.safetensors` ZIP member, and flips one byte at the middle of that
member. The original candidate packages are never modified. The complete
1,080-GPU-row, 54-CPU-row, 24-removal, 24-restoration, and 24-corruption fresh
verification must rerun from the beginning in a new process.

The repair passes only if an independent strict recomputation validates all raw
rows, hashes, scores, aggregates, comparisons, controls, and gates. Its claim
ceiling remains the bounded public six-task supplied-data result.
