# Phase 2 independent-human handoff

This is the exact external procedure for completing the only unresolved Phase
2 gate. It does not authorize synthetic, model-generated, or research-agent
ratings.

1. A custodian gives exactly one of the sealed `rater_form_N.jsonl` files to
   each of three distinct human raters. The custodian withholds
   `blinding_key.jsonl` and every other rater's work.
2. Each rater fills every rating field in their assigned 7,000-row form. They
   must not change prompts, outputs, identifiers, capabilities, or schema.
3. Preserve the returned files as
   `results/abi_capability_compiler_phase2/human_ratings_v1/rater_form_N.completed.jsonl`.
   Never overwrite the sealed templates.
4. Copy `PHASE2_HUMAN_RATING_ATTESTATIONS_TEMPLATE_V1.json` into that directory
   as `attestations.json`, replace every placeholder, and obtain the stated
   declarations from the raters and custodian.
5. While the answer key is still withheld, lock the completed hashes:

   ```powershell
   abi-capability-compiler-phase2-human-ratings lock `
     --packet-dir results/abi_capability_compiler_phase2/human_rating_packet_v1 `
     --completed-dir results/abi_capability_compiler_phase2/human_ratings_v1 `
     --output results/abi_capability_compiler_phase2/human_ratings_v1/blind_lock.json
   ```

6. Only after the lock succeeds may the custodian release the answer key for
   scoring. Run:

   ```powershell
   abi-capability-compiler-phase2-human-ratings score `
     --packet-dir results/abi_capability_compiler_phase2/human_rating_packet_v1 `
     --completed-dir results/abi_capability_compiler_phase2/human_ratings_v1 `
     --lock results/abi_capability_compiler_phase2/human_ratings_v1/blind_lock.json `
     --output results/abi_capability_compiler_phase2/human_ratings_v1/manifest.json
   ```

7. Run the Phase 2 evidence verifier. It recomputes the score from the locked
   raw forms and rejects missing, malformed, modified, or non-reproducible
   evidence.

The score uses one candidate/prompt cluster as the sampling unit, averages its
three rater credits, resamples within each of the fourteen capability strata
10,000 times with seed 1729, and reports the empirical 95% interval. Candidate
wins receive 1, ties 0.5, and reference wins or both-unacceptable judgments 0.
The registered lower-bound threshold is 0.45.

Software cannot prove that an identifier belongs to a human or exclude
off-system collusion. That evidence is the named custodian's responsibility;
the tool verifies the declarations, file identity, completeness, blinded lock
ordering, and reproducible statistics.
