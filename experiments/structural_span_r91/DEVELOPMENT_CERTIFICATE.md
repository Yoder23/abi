# R91 disclosed development certificate

R91 passed every preregistered disclosed-development gate. The frozen
530,050-parameter bridge passed 95/100 R60 reasoning rows versus 0/100 for the
old R60 endpoint and 0/100 for the deterministic random bridge. It retained
all 21 source-passing rows, had zero collapse, and physically executed one
reasoning cake, six reasoning adapters, and one bridge on every row.

- Candidate checkpoint: `04b350e0a7f22ac4facb380f6bf99edfc0022697a312248972491434234a4fba`
- Candidate metadata: `d4d413549b09b37869e51a1bb406971cdcfacd4116421d6d98b96385f78a82e3`
- Candidate binding: `4297089ea19a2acc39ea6ff2d941d927410d6a89a983b02ec14413abb5a416e1`
- Raw evidence: `3515848c16105be51ec046271a781ba4bb884f73deea41bc20fd66836a679659`
- Evidence digest: `53fb783e4c97e18e68df42ac5bc78cf0fa3ddbd055faaaad4743e293a6b3e8df`

Training used all 8,129 immutable teacher records, 65,716 teacher tokens, zero
new teacher calls, and 128,000 meaning-preserving normalized examples. Only
the bridge trained, for 518.18 seconds on the declared GPU. The teacher is
absent at inference and no source parameters, logits, or activations are in
the package.

This disclosed pass authorizes the single R92 prospective test. It is not a
prospective result, broad-English certification, or completion of the ABI
moonshot.
