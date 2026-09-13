# R36 v2 duplicate-field preflight repair

The v2 training-only preflight found zero eligible abstention and reasoning
rows because their immutable identifier is intentionally repeated in more than
one supplied field.  The field-addressed representation can safely point to
the canonical first occurrence: every occurrence has identical bytes and the
fixed field order makes that action stable.  Requiring global uniqueness was
an unintended holdover from the position-pointer tokenizer.

The pointer resolver now selects the first identical supplied-field occurrence
instead of rejecting duplicates.  This ran before v2 selection, training,
evaluation, or result creation.  No data, output, action bytes, architecture,
seed, compute, fixture, metric, or threshold changed.
