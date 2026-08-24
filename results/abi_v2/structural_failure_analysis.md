# ABI V2 structural failure analysis

The V1 failure is an executable-schema mismatch, not evidence that Qwen or
Pythia should learn LayerCake's residual basis. The frozen English archive is
a complete width-768 LayerCake decoder/router/residual machine carrying a
GPT-2 tokenizer. Each specialist archive is a separate width-64 pointer
decoder carrying its own lexeme vocabulary. Neither archive is a Qwen or
GPT-NeoX state dictionary.

The external boundary is already representation-neutral: every package takes
strict UTF-8 bytes and produces strict UTF-8 bytes. The internal execution
schema is intentionally package-owned. Qwen and Pythia rejected V1 because
they had no loader, lifecycle, state namespace, or output-fusion contract for
those machines.

## Ownership decision

| Mismatch | Owner | V2 treatment |
| --- | --- | --- |
| Hidden width and residual basis | Host/capability remain private | Never project one private basis into another |
| Tokenizer and vocabulary | Host adapter | Strict UTF-8 and exact anchor round trips |
| Position and normalization | Host adapter | Explicit canonical indices and normalized typed channels |
| Capability execution schema | Canonical ABI runtime | Versioned host-neutral runtime classes |
| Incremental state | Canonical ABI | Disjoint host, adapter, and capability namespaces |
| Output heads | Host adapter | Canonical output intent and authoritative byte anchors |
| Package lifecycle | Host adapter | Verify/load/register/activate/remove/reinstall |
| Domain routing | Capability artifact | Preserve direct selected-only execution |

## Bounded V2 hypothesis

One capability-blind adapter per architecture will certify the frozen host
checkpoint and tokenizer against a typed canonical state and lifecycle. A
host-neutral capability runtime—not the LayerCake v25 host—will execute the
unchanged package-owned machine. The adapter will translate host input and
canonical output through strict UTF-8 and its frozen native tokenizer.

This does not claim that capability tensors were transplanted into Qwen or
Pythia weights. It tests whether those independently frozen host environments
can conform to one package ABI without capability-specific fitting. A shell
that merely invokes the LayerCake provider remains non-qualifying.

The complete classified evidence is in `structural_failure_analysis.json`.
