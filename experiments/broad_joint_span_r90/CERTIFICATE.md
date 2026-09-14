# R90 disclosed diagnostic negative certificate

Verdict: `FAIL_R90_DISCLOSED_DIAGNOSTIC`

The unchanged R88 package passed 55/100 disclosed R60 reasoning rows, compared
with 0/100 for both the old R60 endpoint and a deterministic random bridge.
It ran physically sparse on all 100 rows with zero collapse. It failed the
locked 95/100 quality and 95% source-retention gates, retaining 0/21 source
successes.

The row-level failures showed a structural shortcut: without the source
corpus's candidate-list layout, the bridge often selected the intermediate
class, leading words, or a truncated final span. This negative evidence
motivated R91's materially different structural normalization; R90 itself is
not promoted.

- Raw evidence SHA-256: `46f7f0297cb0c333de08379f9b998b4e105632c23edf7a5fdd52d0443549fc12`
- Evidence digest: `3b482dc4e4127dfa4f2452e3bab8baa0c31f0c88864e1d290ea3d68fc76882d8`

The full ABI moonshot remains `OPEN`.
