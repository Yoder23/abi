# R95 direct paired teacher-transfer failure certificate

Verdict: `FAIL_R95_DIRECT_TEACHER_TRANSFER`

The unchanged R91 LayerCake candidate was evaluated prospectively on all 1,400
R95 rows after the catalog, live source evidence, and candidate binding were
separately committed. It scored 1,328/1,400 against the live frozen teacher's
1,392/1,400. The frozen parent scored 8 and the seeded randomized bridge scored
0. All 1,400 candidate rows physically executed the registered sparse path and
no collapse was detected.

The candidate missed the locked 1,330 quality threshold by two rows. Its family
scores were 326, 345, 310, and 347 of 350, so family 2 also missed the 315 floor.
It retained 1,320 of 1,392 teacher-correct rows (94.83%), below 95%. The paired
candidate-minus-source bootstrap interval was [-0.05857, -0.03357], below the
preregistered non-inferiority floor.

- Binding SHA-256: `f5fe4512b157a53f97688e9d9fc0f4d2fd836849956ad8e377e42d7cc0aefcde`
- Raw evaluation SHA-256: `5ce41125219a7f7cac2dcfce692e6aaea6d7404a1bab107ad16bab6e7d2158332`
- Evidence digest: `3eb47d306780130a750dedc6ec19036c208c0b130fd4d4ce801203f4db8bd241`

R95 is strong positive evidence for a compact teacher-derived reasoning
capability but is a correctly failed certification. It does not establish
teacher-equivalent quality or the full ABI moonshot. The measured limiting
factor is structural-family robustness, concentrated in family 2.
