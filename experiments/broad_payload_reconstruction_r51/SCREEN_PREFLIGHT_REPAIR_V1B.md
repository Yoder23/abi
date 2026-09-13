# R51-v1b validation-screen preflight repair

The v1a screen harness incorrectly rejected frozen source archives containing
surplus evidence even though no required validation row was missing.  R51-v1b
adopts the R49 source-inventory contract without changing model behavior:

- the selected 1,400 IDs must all be present;
- surplus rows are not evaluated;
- surplus count and sorted-ID SHA-256 are sealed in the result; and
- any missing selected row still fails closed.

No checkpoint, router, source row, prompt, evaluator, decoder, threshold, or
scientific gate changes.

