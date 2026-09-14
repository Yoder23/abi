# R96 on R95 disclosed development certificate

Verdict: `FAIL_R96_R95_DEVELOPMENT_TARGETS` (development only)

The frozen R96 candidate scored 1,386/1,400 on the disclosed R95 surface,
versus R91's 1,328 and the live source's 1,392. Family scores were 346, 350,
340, and 350; source retention was 98.99%; parent was 8; random was 0; all
rows were sparse and none collapsed. The paired candidate-minus-source 95%
bootstrap interval was [-0.01071, 0.00214], satisfying the -0.02
non-inferiority margin.

The preregistered point gate nevertheless required the candidate to equal or
exceed the source. R96 was six rows lower, so the verdict remains failure. The
result validates the measured length/structure repair but cannot promote R96
because R95 was disclosed before R96 training. R96 is frozen without further
tuning and must face a new prospective surface.

- Candidate checkpoint: `9d3d6b38b1d437b5cbbed50c1470b4df1aa0371ffa51b1abd2adefbb645806e4`
- Raw evaluation: `416fe57d6497e451ddecf35729931f3a29754b015ace922dcb7f97946fe06233d`
- Evidence digest: `e196a01ad3fe160e1d3e60b1e10f142ab36f2669a606cfde10d577e54cc59d8f0`

The full ABI moonshot remains `OPEN`.
