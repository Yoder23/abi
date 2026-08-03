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

Since that audit, V87 has demonstrated a bounded causal transfer signal and V89
has passed pre-transfer labeling for a locked English-plus-four-domain
ontology. Neither result authorizes a new LayerCake training run.
Capability-compiler Phase 0 is complete; the immediate gate is the Phase 1
normalized, provenance-bound acquisition artifact. See
`CURRENT_PROJECT_STATUS.md` and `ROADMAP.md`.

The controlling successor sequence is
`ABI_CAPABILITY_COMPILER_CAMPAIGN_V1.md` plus its machine-readable contract.
`ABI_ENGLISH_CORE_DOMAIN_SEGREGATION_CONTRACT_V2.json` remains binding for the
artifact boundary. Foreign-teacher acquisition targets measured generation
quality, sparsity, and capability segregation—not lossless model compression.

## Artifact lifecycle

```text
frozen source weights
        |
        v
labeled survey + provenance
        |
        v
semantic destination labels + ambiguity quarantine
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

The versioned `abi.layercake_host_v3` and `abi.layercake_domains_v3` consumers
accept only verified v3 segregated training bundles. They independently
recheck the passing purity manifest and each record's semantic destination
before any training row is materialized, while reusing the frozen certified
v47 training implementation without modifying it. Historical v1/v2 bundles
and consumers remain auditable but cannot enter a new successor training run.

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
3. label every specialist record by domain, capability, and atomic claim or
   skill; keep English records free of specialist labels and declared
   unsupplied facts;
4. provide a user-governed ontology and pass the core/domain segregation
   manifest;
5. count raw prompts, unique UTF-8 bytes, teacher outputs, runtime token IDs,
   logits, activations, copied/final parameters, disk, RAM, time, and hardware;
6. use disjoint search, validation, and final catalogs;
7. demonstrate the largest lower failing nested information and core-size
   budgets;
8. reproduce the promoted result across three seeds or fresh hosts;
9. show core-only specialist abstention, no unselected-domain material, and no
   inactive execution;
10. certify the same final teacher-free host for quality, routing, coherence,
   speed, TTFO, RSS, and package identity.

Source models may contribute to the same LayerCake only when record-level
provenance remains intact and conflicts resolve under the preregistered
fail-closed policy. Source agreement alone is not a semantic label.

## Portability meanings

Package portability is mathematically exact when archive, manifest, tensor
payload, and installed tensor hashes match. Behavioral portability is a
separate finite-suite claim. ABI never converts the latter into a universal
theorem by wording.

The English-core purity claim is also finite and operational. The pipeline
rejects labeled facts, known ontology markers, core/domain payload overlap,
selection mismatches, and ambiguous material, then promotion adds held-out
closed-book leakage tests. This cannot prove literal zero knowledge in weights.

The original `LAYERCAKE_ACQUISITION_PROTOCOL.json` remains historical. Current
locked-suite truth lives in `ABI_MOONSHOT_CERTIFICATE_V2.json`; the later
product decision is `ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json`.
