# ABI Phase 3 Host Conformance V36

Status: **COMPLETE FAILED — LAYERCAKE V2 INTERFACE UNSUPPORTED**

The exact V34 tokenizer loads successfully and every non-special piece is valid
UTF-8. The unchanged `lc-direct-neural-core/2` document loader rejects it with
`ValueError: Unicode-atomic tokenizer document changed`. No host file changed.

This assigns the next repair to the LayerCake host interface, not ABI
extraction or model training. A separate LayerCake-owned concatenative-BPE host
successor is authorized. ABI neural training remains closed until that host is
independently construct-certified.

Result SHA-256: `488ff92664d66286d77f1d21a5e2522895768d8fe2a19e4e737d55c1bcb1ff52`

Evidence SHA-256: `766ac6ab436b3d4f55ab973089a0f62be7bbf0035ff998465ffd9af27a6c43e8`
