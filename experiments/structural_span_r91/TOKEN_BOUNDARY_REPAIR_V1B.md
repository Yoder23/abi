# R91 token-boundary preflight repair

The post-repair 2,000-example parser preflight rejected a structural form that
placed opaque codes directly against parentheses and equals signs. No training
launch or optimizer step occurred. GPT-2 byte-pair tokens crossing those
boundaries could not decode to the exact answer span.

The affected rendering forms now put spaces around symbolic punctuation. This
changes no relation, answer, model, optimization setting, source evidence, or
gate. The full synthetic preflight must pass before the training relaunch.
