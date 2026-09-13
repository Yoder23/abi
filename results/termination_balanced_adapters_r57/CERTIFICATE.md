# R57 termination-balanced adapters: negative certificate

R57 is a preregistered negative result. It assigns half of each complete
training record's language-loss mass to EOS while preserving the exact R55
architecture, R47 parent boundary, training corpus, router, and disclosed
gates.

- checkpoint SHA-256:
  `7929586d30d427d79471e8259d02735ec5519be46d986a684ddaa5464d6652c2`
- metadata file SHA-256:
  `adb116e7ef97ce36febe28361ff341afdf529588c55d5f85d987318499b4d598`
- 16,134/24,419 main rows exposed EOS; 8,285 responses were truncated before
  EOS; all 1,438 anchor rows exposed EOS;
- 6,000 successful CUDA optimizer steps; 5,787,086 trainable and 81,923,342
  frozen parameters; frozen shared-state hashes match exactly;
- disclosed validation: 1,256/1,400 candidate versus 1,220/1,400 source and
  1,277/1,400 parent gate;
- source-pass retention: 0.940983606557377;
- exact external routing and physical sparse execution: 1,400/1,400 each;
- collapse: 7/1,400; per-capability floor: 61/100.

The functional, parent-nondegradation, per-capability, and zero-collapse gates
failed. Final-test was not run. Equal-mass terminal balancing is closed; no
nearby EOS fraction, step, rate, rank, or decoder-guard sweep is authorized.
The full ABI moonshot remains open.

Raw evaluation SHA-256:
`477cc03797b3513962fbdfc1c523a2da564c97dec6418dd7ea22c2ec80b9dbff`.
Result file SHA-256:
`ce21fad2c1f5596ac708bce69d7d7c294fc779320b6510dbed2bcb54788b52da`.
