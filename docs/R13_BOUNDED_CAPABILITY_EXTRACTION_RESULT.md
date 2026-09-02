# R13-B bounded enumerable capability extraction

Updated: 2026-09-02

## Verdict

`PASS: BOUNDED_ENUMERABLE_CAPABILITY_EXTRACTION`

This is a local held-out scientific result, not a new public release. R7
remains the controlling public release. Bulk source adapters and raw rows have
not yet been published at a durable public location.

## Result

Four capabilities were derived from a secret committed in Git before reveal.
For each capability, Qwen2.5-0.5B received ordinary rank-16 LoRA next-token
training on all 2,904 depth-1 through depth-5 programs. Checkpoint selection
used only the 24 atomic prompts and stopped after three consecutive 24/24
observations. No depth-6/7 evaluation prompt was evaluated before the source
state was frozen.

The fixed R12 zero-parameter compiler then queried all 24 atomic transitions
and emitted one unchanged 2,053-byte R11 package per capability.

| Measurement | Result |
|---|---:|
| Held-out capabilities | 4 |
| Source training/evaluation prompt overlap | 0 |
| Source BEFORE atomic accuracy | 0/96 |
| Source AFTER atomic accuracy | 96/96 |
| Source optimizer steps | 600, 500, 600, 600 |
| Source training wall-time sum | 405.37 seconds |
| Native source depth-6/7 accuracy | 372/2,048 (0.181640625) |
| Package/source disagreements | 1,676/2,048 |
| Package/oracle depth-6/7 accuracy | 2,048/2,048 |
| Package bytes | 2,053 each; 8,212 total |
| Source adapter bytes | 35,229,008 each; 140,916,032 total |
| Recipient AFTER rows | 6,144/6,144 exact |
| Recipient RESTORED rows | 6,144/6,144 exact |
| Maximum negative-control accuracy | 0.19140625 |
| Removal equality to BASE | Exact for every host/capability/row |
| Recipient optimization | 0 steps |
| Fresh replay worker processes | 7 distinct |
| Fresh replay observation files | 7/7 byte-exact |
| Repaired hostile controls | 9/9 expected outcomes |

The recipient matrix used frozen Pythia-160M, Qwen2.5-0.5B, and T5-base native
output heads through the unchanged R11 interpreter and capability-blind
codecs. Source capability adapters were not arguments to and were not loaded
by recipient workers.

## Exact interpretation

The 24 probes enumerate the complete `3 operators × 8 states` transition
table. R13-B proves that a conventional Qwen source can carry that complete
finite specification and that ABI can compile it into an exact portable R11
package for three frozen heterogeneous hosts.

R13-B does not prove behavioral transplantation. Native Qwen reproduced only
13.09% to 21.68% of the long compositions, while the package reproduced the
registered capability exactly. ABI canonicalized an exhaustively queried
finite rule; it did not copy Qwen's complete output function or its mistakes.

It also does not prove extraction of knowledge that existed in pretrained
Qwen, non-enumerable generalization, English, specialist domains, LayerCake
product acceptance, information minimality, or superiority to LoRA or
distillation.

## Verification lineage

- Preregistration commit: `6d5bb3c`
- Live-replay freeze commit: `846ba9e`
- Hardened verifier commit: `6eb7fce`
- Hostile-audit v1 failure and repair: `4b42008`
- Hostile-audit v2 harness repair: `cdebe42`
- Run evidence SHA-256:
  `1b4af58a938b02ffa59ae785a4a89fcfbdc827aadf3d55bc597e56cd8212c922`
- Strict verification evidence SHA-256:
  `7832ad8b9229407480cc9355a249673b0c3c63d9832eea8c78c497f18815aff6`
- Live replay evidence SHA-256:
  `4d050451c2ac1bacb726aedffdd4bc23d2ee5ead5adca2f19d4eeaf3a1ac8eaa`
- Hostile audit evidence SHA-256:
  `dba430055ac0c6198b2017a4b205f5f5cffda2ccd58dbeb9ca0a703a76346bd5`
- Final bounded certificate evidence SHA-256:
  `bf29eb3e5cecd1cd5293e5219d10a17f1172ec9f9c0f22b8b3faee43c4757fbf`
- Local artifact-manifest evidence SHA-256:
  `64702ee4e57e6793948abf4a8f80d7e90568821706e76b36c099eec4c80b1512`

Hostile-audit revisions 1 and 2 remain documented as failures. Revision 1
found a real missing per-row argmax check; revision 2 found an incorrect audit
expectation around evidence identities. No certificate was issued until both
were repaired and revision 3 passed.

## Open publication gate

The compact package, receipt, certificate, reveal, and manifest evidence is
tracked in Git. The 243,426,540 bytes of adapters and original/replay raw rows
required for complete independent replay remain local and hash-addressed by
the artifact manifest. R13-B must be described as local until those assets are
published and reconstructed independently.
