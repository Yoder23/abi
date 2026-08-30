# Hostile audit

The R7 verifier mutates only a marked disposable extraction. It covers missing
or corrupt packages, absent rows or hashes, altered transitive code, stale
receipts, incomplete inventories, missing adapters, archive hiding, and broken
release bindings. Every mutation must fail, and exact restoration must pass.

- Pre-public R7: 19/19 rejected, exact restore passed.
- Post-public R7: 19/19 rejected, exact restore passed.
- Blind scanner controls: 12/12 valid/invalid USTAR/V7 prefix cases passed.

R5 and R6 are retained because blind review exposed real defects in those
lineages. R7 is additive; it does not relabel those failures.
