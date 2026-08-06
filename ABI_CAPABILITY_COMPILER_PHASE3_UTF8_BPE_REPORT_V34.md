# ABI Phase 3 UTF-8 BPE V34

Status: **PASS — REPRESENTATION ONLY**

V34 fit a deterministic BPE vocabulary using Phase 1 training prompts and
outputs plus the development-independent printable-ASCII syntax line. Pieces
concatenate directly to the exact UTF-8 response; no byte escape, normalization,
development vocabulary insertion, neural training, or LayerCake change occurs.

| Requested vocabulary | Fixed actions | Train max | Development max | Qualifies |
|---:|---:|---:|---:|---:|
| 1,000 | 1,003 | 298 | 413 | No |
| 2,000 | 2,003 | 176 | 361 | No |
| 3,000 | 3,003 | 122 | 344 | No |
| 4,000 | 4,003 | 104 | 328 | No |
| 4,996 | 4,999 | 99 | 317 | Yes |

The smallest preregistered passing budget is 4,996. All 8,400 targets
reconstruct exactly; training and development have zero sequences above 320.
This opens a separately preregistered LayerCake host-conformance feasibility
test for this exact tokenizer identity. It does not authorize model training.

Result SHA-256: `abf78d1f796cec1c3e28641b10210ebb02d7a9914965e649682f7bb7e8f15f91`

Evidence SHA-256: `9c6705f44c0e8f30c3f0f1f8f97d06af8aa81aa1fd1896d8bc4a01c63b30f1af`
