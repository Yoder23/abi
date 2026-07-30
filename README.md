# ABI: capability extraction for LayerCake

ABI extracts a measured capability substrate from frozen open-weight language
models, labels it by destination, and packages it for a teacher-free LayerCake
runtime. LayerCake remains the execution, installation, composition, and
routing system. Source transformers and ABI extraction bundles are not part of
the deployed product.

## Current status

The bounded ABI-to-LayerCake reference release is **PASS**, but the broader
English-product moonshot is **OPEN**. Its locked-suite machine-readable claim is
[`ABI_MOONSHOT_CERTIFICATE_V2.json`](ABI_MOONSHOT_CERTIFICATE_V2.json).
The earlier v1 certificate and every failed experiment remain historical
evidence.

A post-certificate novel-prompt audit found that the exact candidate passed
0/28 checks while frozen Phi-3 passed 19/28, creating 19 source-passing
regressions. See
[`ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json`](ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json).
This falsifies any claim that v47 is already a generally fluent English core.
The CLI remains useful for reproducing the bounded result, not as a
general-purpose source-quality assistant.
The next campaign is preregistered in
[`ABI_ENGLISH_CORE_DOMAIN_SEGREGATION_CONTRACT_V2.json`](ABI_ENGLISH_CORE_DOMAIN_SEGREGATION_CONTRACT_V2.json).
The earlier successor contract remains historical.

### Corrected product target

Extraction from a foreign transformer is not called lossless. The successor
must retain the teacher's measured English generation quality while minimizing
the tested LayerCake core and keeping specialist acquisition material outside
that core. English records may contain linguistic-form tasks based on
abstract/nonce content, supplied non-domain context, interpersonal pragmatics,
or domain-free instructions. Specialist facts, procedures, calculations, and
code must be labeled and packaged as separately selectable domain material.

This is a bounded, falsifiable purity claim—not proof that neural weights
contain literally zero world knowledge. Exact losslessness remains available
afterward for LayerCake-to-LayerCake package transfer, where bytes, manifests,
tensors, and installed payload identity can all be verified.

The exact final candidate:

- passes 1,700/1,700 locked final observations across 17 capabilities;
- has zero regressions on the 1,692 observations passed by its frozen sources;
- contains no teacher and no source transformer blocks at inference;
- exposes English as the core and signed chemistry, civics, and Python cakes;
- executes inactive domain cakes zero times;
- reproduces across three fresh host initializations;
- preserves exact CPU/CUDA domain-package output identity on the certified
  suites;
- runs at a 2.801x median CPU byte-throughput ratio against the optimized
  comparator in the 120-observation headline benchmark (paired bootstrap 95%
  lower bound 2.672x);
- has 45 ms median time to first output, a 64,665,070-byte active runtime model,
  and 168,718,336-byte measured peak process RSS in that benchmark.

The deployed English artifact is 68,288,061 bytes. English budget 3 uses 1,381
selected records and 39,136 teacher tokens. It is the lowest passing budget
among four preregistered nested budgets; budget 2 failed with 19
source-passing regressions. This is not a proof of a global information
minimum.

Qualified domain scope is intentionally exact:

| Package | Certified capability | Teacher tokens | Parameters | Largest tested lower budget |
|---|---|---:|---:|---|
| chemistry | periodic-table queries | 1,500 | 124,766 | failed |
| civics | independence-day queries | 1,210 | 121,025 | failed |
| python | bounded arithmetic-function generation | 691 | 109,415 | failed |

Mathematics failed its locked closure gate and is not packaged. “Python,”
“chemistry,” and “civics” here do not mean exhaustive mastery of those fields.

## Verify and reproduce

From this directory:

```powershell
C:\Python310\python.exe -m abi.moonshot_release verify
C:\Python310\python.exe -m abi.moonshot_release inspect
C:\Python310\python.exe -m abi.moonshot_release generate `
  --allow-bounded-reference `
  --prompt "Rewrite professionally in one sentence: Hey Asha, send file-212.txt now."
C:\Python310\python.exe -m abi.moonshot_release generate `
  --allow-bounded-reference `
  --domain python `
  --device cpu `
  --prompt 'Write only Python code defining `calculate_100(a, b)` that returns the add result using `a + b`.'
```

Use `--device cuda` for an installed domain cake when CUDA is available. The
English runtime remains the same certified host. Every invocation first checks
the certificate, evidence, candidate components, implementation files, signed
packages, and sealed LayerCake checkout.

The large runtime artifact and experimental ledger are release assets, not
ordinary source-package files. Verification fails closed if a referenced asset
is absent or changed.

## Extract another source or capability

The certified release does not assert that every model or all of its knowledge
can be extracted automatically. A new source follows the governed pipeline:

1. Pin its immutable revision and hash every source weight.
2. Survey it with a labeled, user-extendable capability catalog.
3. Keep search, validation, and final prompts disjoint.
4. Compose only selected records into a non-deployable `.abix` acquisition
   bundle; require a domain ontology and passing core/domain segregation
   manifest; reject conflicts and provenance loss.
5. Search nested imported-information budgets and retain the adjacent failure.
6. Train or conform the smallest tested passing LayerCake core/cake.
7. Package domains separately, then test routing, isolation, identity,
   tampering, CPU/CUDA behavior, speed, memory, and fresh-host reproduction.
8. Remove the teacher and all extraction material before final testing.

Start with:

```powershell
C:\Python310\python.exe -m abi.moonshot --help
```

New compositions require `--domain-ontology
catalogs/domain_ontology_v1.json`. The checked-in ontology is a bounded starter
for the current catalog, not an exhaustive map of a source model's knowledge.
Generate or inspect the first fully segregated synthetic catalog with:

```powershell
C:\Python310\python.exe -m abi.certification_catalog `
  --version v6 `
  --output catalogs/english_and_first_domains_certification_v6.json
```

V6 has 100 search, 100 validation, and 100 sealed final-test probes for each of
14 English and four domain capabilities. It validates the extraction,
selection, labeling, and quarantine path; its disclosed synthetic templates
are not sufficient evidence of broad English generalization by themselves.

An `.abix` file contains teacher material. It is never a deployable cake.

## Proof boundaries

"Lossless" is reserved for exact LayerCake package transfer. Foreign-teacher
capability acquisition is source-relative generation-quality retention on
locked, independently separated suites.

Finite testing cannot establish semantic identity on every possible prompt or
discover every latent domain in arbitrary weights. ABI therefore issues
capability-specific certificates, preserves failures, and requires
recertification for every new model, catalog, host, or package selection. The
conditional theorem and impossibility boundary are in
[`FORMAL_UNIVERSAL_TRANSFER.md`](FORMAL_UNIVERSAL_TRANSFER.md).

## Documentation map

| Document | Purpose |
|---|---|
| [`ABI_MOONSHOT.md`](ABI_MOONSHOT.md) | Final campaign result and evidence map |
| [`LAYERCAKE_ACQUISITION.md`](LAYERCAKE_ACQUISITION.md) | Acquisition and deployment workflow |
| [`CLAIMS.md`](CLAIMS.md) | Canonical claim ledger |
| [`RESEARCH_STATUS_AND_GAPS.md`](RESEARCH_STATUS_AND_GAPS.md) | Remaining scientific gaps |
| [`ACTIVE_MISSION.md`](ACTIVE_MISSION.md) | Current sealed state and extension policy |
| [`PROOF_LAYERS.md`](PROOF_LAYERS.md) | Legacy NIB proof layers |

The earlier NIB cross-architecture experiments remain valid historical
research controls. They are not the certified deployed LayerCake release.
