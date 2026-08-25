# External reproduction

The definitive archive is for a genuinely independent operator on hardware
different from the development laptop. Follow `external_reproduction/README.md`
and run:

```text
abi-reproduce verify
abi-reproduce certify-hosts
abi-reproduce capability-matrix
abi-reproduce causality
abi-reproduce isolation
abi-reproduce performance
abi-reproduce hostile-audit
abi-reproduce report
```

First perform the same sequence in a brand-new clone using only the published
tag, manifest, and assets. That local reconstruction plus blind red-team is a
prerequisite, not independent reproduction.
