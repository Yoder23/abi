# Runtime performance

Twenty paired observations per declared environment yield these median
conformance-wrapper overhead fractions:

| Environment | Median overhead fraction |
| --- | ---: |
| LayerCake v25 | 0.026818 |
| Qwen2.5-0.5B | -0.007342 |
| Pythia-160M | 0.081364 |

All are below the registered 10% ceiling. These are wrapper/conformance
measurements, not full generation throughput, TTFT, energy, or product-quality
benchmarks. R7 makes no runtime superiority claim over LoRA or distillation.
