# R89 frozen-package holdout certificate

R89 passed every preregistered gate with the unchanged R88 package.

- R88 candidate checkpoint: `15c28bdceea49e61e771092ce74be1cc30233e8bef53d87171d36f5a2487372f`
- Holdout binding claim: `71e7066a54428200bb6448c87a6e1bc40c909089574aa7872313ec1f8a3dc1d7`
- Holdout binding file: `a5f869fe51262f7c2b5b5e8f7aa7a6605e267bc98d03a86dbd9fac0d7c8acc67`
- Screen result file: `63b8b632184a021c33fe5fb72b7af322d20400eda8fc356e6589936316df5616`
- Raw 1,400-row evidence: `23dd0fd2b02a2526498255d72a2639eb2242a3d99a52816777a2c6550a9e7708`
- Recomputable evidence digest: `cf9cf6f44b9e3fd07ad4b3c384de64c39ca92a67ac6c34b30ef032ecfea016a0`

The unchanged package passed 1,400/1,400 and every family passed 200/200 on
new lexical triples, a new subject prefix, new numeric identities, and seven-
digit codes outside its training and R88 validation distribution. It retained
all 1,353 source-passing rows. The unchanged R78 parent passed 209/1,400 and a
fresh deterministic random bridge passed 0/1,400. Candidate-minus-parent was
0.85071 with 95% bootstrap interval `[0.83214, 0.86929]`; candidate-minus-
source was 0.03357 with interval `[0.02429, 0.04357]`. Collapse was zero and
all 1,400 rows passed the physical sparse-execution trace.

The candidate was not retrained or calibrated. This independently replicates
bounded extractive transfer across a tokenization-width shift. It does not
certify unrestricted English, domains, minimality, fair LoRA/distillation
superiority, or the full ABI moonshot. The full ABI moonshot remains `OPEN`.
