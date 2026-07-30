# LayerCake capability acquisition

ABI and LayerCake have separate responsibilities:

- ABI surveys frozen open-weight sources, labels capabilities, minimizes the
  tested imported-information budget, and produces provenance-bearing
  acquisition material.
- LayerCake hosts one English substrate and separately signed domain cakes,
  activates only the selected cake, and provides CPU/CUDA execution.

The bounded reference acquisition is certified. The general English product is
not: a post-certificate audit exposed 19 source-passing regressions on 28 novel
prompt forms. A new model or capability is not certified merely because the
pipeline can ingest it.

## Artifact lifecycle

```text
frozen source weights
        |
        v
labeled survey + provenance
        |
        v
non-deployable .abix search material
        |
        v
nested budget search + adjacent failure
        |
        v
teacher-free English host / signed domain cake
        |
        v
LayerCake installation, routing, and certification
```

`.abix` files may contain teacher outputs and are prohibited from deployment.
The final English host contains no source transformer blocks. Domain cakes are
immutable signed archives and can be installed independently.

## Reference release layout

The v2 certificate points to the exact English artifact, three package archives
and public keys, final evidence, and sealed sibling LayerCake commit. Do not
copy paths manually into a new certificate; use the fail-closed builder and
verifier.

```powershell
C:\Python310\python.exe -m abi.moonshot_release verify
C:\Python310\python.exe -m abi.moonshot_release inspect
```

Reproduce the bounded host:

```powershell
C:\Python310\python.exe -m abi.moonshot_release generate `
  --allow-bounded-reference `
  --prompt "Rewrite professionally in one sentence: Hey Asha, send file-212.txt now."
```

Run one explicitly selected cake:

```powershell
C:\Python310\python.exe -m abi.moonshot_release generate `
  --allow-bounded-reference `
  --domain chemistry `
  --device cuda `
  --prompt "What is the chemical symbol for gold?"
```

If CUDA is unavailable, use `--device cpu`. Uninstalled, unsigned, tampered,
duplicate, traversal-bearing, stale, or wrong-key packages fail closed.

## Adding a source

Use `python -m abi.moonshot --help` for source survey, composition, and bundle
verification commands. A production extension must additionally:

1. pin source model revision and weight identity;
2. declare user-selected capability labels and destination scopes;
3. count raw prompts, unique UTF-8 bytes, teacher outputs, runtime token IDs,
   logits, activations, copied/final parameters, disk, RAM, time, and hardware;
4. use disjoint search, validation, and final catalogs;
5. demonstrate the largest lower failing nested budget;
6. reproduce the promoted result across three seeds or fresh hosts;
7. show no unselected-domain material or inactive execution;
8. certify the same final teacher-free host for quality, routing, coherence,
   speed, TTFO, RSS, and package identity.

Source models may contribute to the same LayerCake only when record-level
provenance remains intact and conflicts resolve under the preregistered
fail-closed policy. Source agreement alone is not a semantic label.

## Portability meanings

Package portability is mathematically exact when archive, manifest, tensor
payload, and installed tensor hashes match. Behavioral portability is a
separate finite-suite claim. ABI never converts the latter into a universal
theorem by wording.

The original `LAYERCAKE_ACQUISITION_PROTOCOL.json` remains historical. Current
locked-suite truth lives in `ABI_MOONSHOT_CERTIFICATE_V2.json`; the later
product decision is `ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json`.
