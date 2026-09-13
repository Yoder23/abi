# R56 response-only four-gram guard — negative certificate

Status: **FAILED; BRANCH CLOSED**

R56 froze R55's weights and applied the preregistered response-only token
four-gram continuation mask.  It performed zero training steps and zero
teacher calls.

On disclosed validation it removed all detected collapse but reduced quality
from R55's 1,277/1,400 to 1,108/1,400 and source-pass retention to 81.23%.
Coherence fell to 0/100 and domain-independent reasoning to 6/100 because
legitimate repeated prompt-derived identifier token patterns were masked.
This is a causal failure of unconditional token banning, not a neural-weight
change.

- raw rows SHA-256:
  `e1e552d69a804d9c61a0e2e00cdde9445d41b028f80a8cce86f83cba746b8cc3`
- result file SHA-256:
  `24f5cf64fcc3069832fc7266eb73d2281c290edfc5630adcddafc1e63851a5a9`
- embedded evidence SHA-256:
  `b2e5847a0ffdcb1433d46eadeba8c27f5a703fd1a50d90d2dde541d4dbc8187b`

Per protocol, no n-gram-size sweep is authorized.  The next successor must
change sequence-termination supervision while leaving legitimate response
structure unconstrained.  R56 is not a candidate and the ABI moonshot remains
**OPEN**.
