# R84 protocol: deep selector plus deployment-mixture pointer

R83 failed prospectively at 2/1,400, equal to its R47 parent. Its selective
training optimized a gate/attention surrogate but omitted the actual deployed
pointer-language mixture likelihood. R78 provides an orthogonal measured
signal: its frozen deep-adapter host selected the correct conditional answer on
683/1,400 rows, although autonomous realization passed only 83/1,400.

R84 composes those two mechanisms without altering either sealed ABI code or
the R78 parent. It attaches one 49,935-parameter rank-32 prompt pointer to the
six-block, fourteen-route R78 host. The parent and all 84 installed deep
adapters remain byte-identical. Unlike R83, the pointer is trained with the
exact likelihood of the mixture used at deployment, plus direct gate and
attention supervision. The teacher is absent during training and inference.

Immutable inputs:

- R78 parent checkpoint: `b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e`
- R78 parent metadata: `1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e`
- R76 ABI artifact: `292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc`
- broad anchor: `82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150`
- R81 catalog: `0d2b24417bf1ea925b97747d54ac7053ac7a38f517966431e8befb8f893d2c54`
- R81 source result: `474e2e027b7e2447f4a33174061ea1cf556dd0572b74f5730a8bcafe0f35078f`
- R83 raw failure evidence: `4d90c0b27c271fbf3656b89dc8e26f34b407393676dd8c250b9805e2ccc5ac2e`

One CUDA candidate is authorized: seed 84001, 6,000 successful steps,
main/anchor batches 4/4, pointer learning rate 3e-4, mixture/direct pointer and
prompt-overlap weights 1.0, zero classifier loss, no teacher or parent
distillation, recovery from step 400 every eight steps over 4/8/16 tokens,
256-token contexts with enumerated overlength exclusions, uniform main
sampling, and balanced broad anchoring. No nearby seed, weight, or step sweep
is authorized.

Before any candidate generation, checkpoint, metadata, protocol, and screen
hashes are bound. The 1,400-row R81 screen requires at least 1,330 autonomous
exact answers, at least 180/200 in every family, at least 95% retention of the
1,382 source successes, at least a 50-point paired gain over unchanged R78,
zero collapse, one physical capability cake, six physical deep adapters, and
one pointer only. Conditional selection is diagnostic, not a substitute for
generation. Failure closes R84.

Passing is bounded nonce relation/copy acquisition, not unrestricted English,
domain transfer, minimality, LoRA/distillation superiority, or full ABI
moonshot certification. The full ABI moonshot remains `OPEN`.

## Pre-observation amendment

`AMENDMENT_1.md` records a fail-closed finalization error after training but
before binding or generation: inherited R78 adapter topology was present in the
checkpoint but omitted from candidate metadata. The correction carries forward
the frozen parent ledger only. Candidate weights and all scientific gates are
unchanged.
