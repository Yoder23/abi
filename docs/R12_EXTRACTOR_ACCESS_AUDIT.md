# R12-A extractor access audit

Updated: 2026-09-02

## Finding

The R12-A extractor is a fixed behavioral compiler for an exhaustively
enumerable capability family. It is not a general latent-knowledge extractor.

The capability family has three opaque operators, eight input states, and
eight output states. The extractor sends one canonical prompt for every
operator/input pair:

`3 operators × 8 input states = 24 atomic probes`

Those 24 outputs are the complete transition table for the registered family.
Once they are exact, the frozen recurrent R11 executor can compose them at any
registered program depth. The 512/512 package score therefore validates
package construction and compositional execution, but it does not show that
the extractor inferred a non-enumerable capability from sparse observations.

## Exact access

The extractor receives:

- exactly 24 canonical atomic prompts;
- the frozen source model's full-vocabulary next-token logits for each prompt;
- the eight already registered canonical digit token IDs; and
- the frozen R11 package schema.

The extractor does not receive:

- source-training rows;
- source-training labels;
- public evaluation rows or labels;
- capability offsets, rule seed, or capability ID;
- held-out data;
- learned parameters; or
- a capability-specific recipient optimizer.

For each probe, it requires the source's native full-vocabulary argmax to be
one of the canonical digit tokens, selects the largest canonical digit logit,
and writes that result as one row of a one-hot `3 × 8 × 8` transition tensor.

## Scientific interpretation

R12-A supports:

> A conventionally trained Qwen source exposed a complete finite capability
> specification through 24 behavioral queries, and a zero-parameter compiler
> converted that specification into an exact frozen R11 package.

R12-A does not support:

> ABI copied Qwen's complete learned function losslessly.

The distinction is observable: native Qwen peaked at 510/512 compositions,
while the canonical package executed 512/512. The package recovered the finite
transition rule and removed native source execution errors. That is capability
canonicalization, not equality to all source outputs.

This limitation is permanent context for every R12-A citation.
