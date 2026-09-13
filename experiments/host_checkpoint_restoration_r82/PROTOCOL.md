# R82 protocol: exact restoration of the cleaned v51 host lineage

The August 29 storage cleanup intentionally removed large, reproducible
`.safetensors` files while preserving an immutable manifest, metadata, and
hashes. R81 subsequently authorized R79, whose preregistered parent is v51.
Preflight correctly found v51's weight file physically absent.

R82 is artifact restoration, not a new quality experiment. Starting from the
still-present Phase 2 base checkpoint
`9e0e6b9add32b4c460f7b570a32584f380e59bf6d631e313ff813069d24e09e1`,
it replays the five historical runs with their original input archives,
budgets, seeds, step counts, batch sizes, optimization settings, recovery
schedules, preservation streams, and decoding contracts. A stage may populate
its historical `model.safetensors` only when the reproduced bytes match the
historical SHA-256 and byte count exactly. A mismatch stops the chain.

Expected chain:

| stage | expected checkpoint SHA-256 | steps |
|---|---|---:|
| v6 | `5bae16dec3a55759e92a8481ec5ee519a5ee8687ad1a6f019edc1df3225d092e` | 4,000 |
| v21 | `b6bda948f24eb93e4abbfebf1e451b8911b4614a1829527c257007d8052753e1` | 23,998 |
| v29 | `537b274c752d167f4a636bf7608120f35f8fa158d604bebd018484cc8dfe79b9` | 12,000 |
| v41 | `22df8256a0479596516c1d09c29200fc19c1cb810ac8204979b2192ce39372e0` | 6,000 |
| v51 | `012c5443dca21fb73874f1329bcfb6a10526284092e15d7408fff15614dd563f` | 12,000 |

Restoration does not alter historical metadata or retroactively create new
evidence. Reproduction outputs and per-stage restoration receipts are retained.
R79 remains blocked until all five hashes reproduce exactly.
