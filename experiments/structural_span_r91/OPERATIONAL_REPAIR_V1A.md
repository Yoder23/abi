# R91 pre-step span-parser repair

The first frozen R91 launch failed before optimizer step 1 and before creating
an output directory. The imported R85 `_span_example` helper required the gold
response to occur exactly twice. R91's preregistered candidate-list placement
variation intentionally creates prompts with one or more occurrences.

This repair replaces only that inherited cardinality assumption with a local
offset-mapped parser requiring at least one exact, token-decodable occurrence,
equal token width across all occurrences, at most 16 response tokens, and at
most 256 prompt tokens. Every exact occurrence remains a valid gold span.

No teacher row, rendered prompt, answer, structural form, interface, candidate
placement, model architecture, seed, optimization setting, evaluator, gate,
or planned prospective boundary changes. The failed launch produced no
scientific candidate and is not counted as a training run.
