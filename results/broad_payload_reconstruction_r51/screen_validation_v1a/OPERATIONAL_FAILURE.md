# R51 validation screen v1a operational failure

Verdict: `FAIL_CLOSED_ZERO_ROWS_SURPLUS_SOURCE_PREFLIGHT`

The disclosed validation screen stopped before candidate model loading or
generation.  All 1,400 required source rows were present, but the frozen
source archives also contain 200 non-selected rows and the new screen harness
incorrectly required exact set equality.

Candidate rows, model loads, generations, and training steps were all zero.
The additive v1b repair may only restore the already established R49 coverage
contract: fail on any missing selected ID, permit but enumerate and hash
surplus non-selected IDs, and evaluate only the selected 1,400 rows.

