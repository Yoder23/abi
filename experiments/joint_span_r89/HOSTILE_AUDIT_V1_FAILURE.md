# R89 hostile-audit v1 failure

The first R88/R89 hostile audit is a preserved assurance failure.

The unmodified baseline passed. Nine physical mutations were rejected:
missing or truncated R88 raw evidence, a forged R89 gate, changed candidate
checkpoint, changed parent checkpoint, changed imported artifact, changed
candidate binding, changed holdout screen, and changed R89 catalog.

Deleting `tokenizer.json` was incorrectly accepted. `AutoTokenizer` silently
reconstructed an equivalent tokenizer from `vocab.json` and `merges.txt`, so
the scientific outputs still recomputed, but the verifier failed the stricter
physical-custody requirement. No original evidence file was changed and no
v1 pass receipt was emitted.

The additive repair requires every tokenizer file by exact byte length and
SHA-256 before loading. All ten hostile cases must be rerun; this failure may
not be relabeled as a pass.
