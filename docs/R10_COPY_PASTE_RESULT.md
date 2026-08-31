# R10 copy/paste result

Updated: 2026-08-30

R10 resolved one ambiguity in the ABI evidence: the canonical runtime can
cleanly paste, remove, and restore a source-extracted synthetic capability, but
this is not the same as losslessly transplanting the source model's native
behavior.

## Result

| Gate | Recomputed result |
| --- | ---: |
| Frozen recipient families | 3: GPT-NeoX, Qwen2, T5 encoder-decoder |
| Synthetic capabilities | 4 |
| Distinct prompts per capability | 512 |
| Conditions | 10 |
| Raw recipient rows | 61,440 |
| AFTER accuracy | 1.000 for every host/capability |
| RESTORED accuracy | 1.000 for every host/capability |
| Largest negative-control accuracy | 0.15625 |
| REMOVED and INTERPRETER_REMOVED | exact row-level BASE match |
| Recipient optimizer steps | 0 |
| Interpreter learned parameters | 0 |
| Recipient state mutation | none; full state hashes exact |
| Fresh live replay | 61,440/61,440 recipient and 4,096/4,096 source rows byte-exact |
| Source-native AFTER accuracy | 0.59375 to 0.65234375 |
| Registered source gate | at least 0.99 and gain at least 0.70 |
| Overall R10 verdict | **FAIL_SOURCE_NATIVE_GENERALIZATION** |
| Runtime component | **PASS_BOUNDED_SYNTHETIC** |

The five content-addressed packages are 2,040 bytes each. Four contain the
extracted AFTER states and one contains the BEFORE state. Package files contain
no prompt, answer, row ID, solver, model/tokenizer identity, hidden width, or
host matrix.

## What is proven

For this registered synthetic transition IR, identical immutable package bytes
are sufficient for a zero-parameter canonical interpreter and generic host
codec to produce exact behavior across the three frozen model families. Package
removal removes that behavior, and pasting the identical bytes restores it.
The fresh replay regenerated every stored source and recipient row exactly.

`CanonicalCapabilitySlot` exposes this bounded mechanism directly:

```python
from pathlib import Path

from experiments.copy_paste_r10.slot import CanonicalCapabilitySlot

slot = CanonicalCapabilitySlot()
receipt = slot.paste(Path("sha256-...abipkg"))
probabilities = slot.execute(["Opaque program: start 1 ; apply vok tem ; result ="])
slot.remove()
slot.paste(Path("sha256-...abipkg"))
```

The slot re-hashes the installed file before each execution and fails closed if
the package is absent or changes after paste.

## What is not proven

R10 does not prove lossless source-model behavior. The extracted atomic
transition representation composes exactly in the canonical runtime, but the
source model's own decoder generalized to only 59.4-65.2% on the newly sampled
depth-4-to-7 prompts. That miss is why the overall preregistered R10 verdict is
negative even though every recipient runtime condition passed.

R10 also does not execute through the LayerCake product host, internalize the
capability in recipient neural computation, extract English or domains, match a
teacher's generation quality, establish minimum information, or outperform
LoRA/distillation. Those claims remain open.

## Verifier record

The frozen v1 verifier stopped on a per-host row-count arithmetic error. A
sealed one-line repair then exposed a float32 comparison tolerance below the
measured Pythia codec round-trip error. Both failures are preserved. The final
negative report uses the registered correct count and a bounded `1e-5`
comparison tolerance; the maximum observed error was
`8.225440979003906e-06`. Neither repair changes a scientific threshold, raw
row, exact output comparison, or the negative source verdict.

See `results/copy_paste_r10/revision_001/public_manifest.json` for the exact
hash-addressed evidence inventory.
