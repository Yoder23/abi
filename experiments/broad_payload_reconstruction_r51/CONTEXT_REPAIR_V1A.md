# R51-v1a additive context repair

The R51 v1 launch stopped before optimizer step 1 because exactly 2 of the
24,421 broad source rows cannot fit the frozen 256-token LayerCake context.

R51-v1a changes one command-line condition: enable
`--exclude-overlength-prompts`.  The acquisition implementation must preserve
the exclusion count, exact record identities, prompt hashes, and an aggregate
exclusion hash in immutable metadata.  The remaining 24,419 rows and all 1,438
anchor rows retain their original text and supervision.

No truncation, context increase, data replacement, learning-rate change,
step-count change, seed change, or gate change is authorized.  All other R51
protocol terms remain controlling.
