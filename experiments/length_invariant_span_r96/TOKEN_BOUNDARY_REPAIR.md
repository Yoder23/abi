# R96 preflight token-boundary repair

After preregistration commit `3688c43`, a 20,000-example CPU tokenizer
preflight failed before training on a form using compact `--` separators. The
response span shared tokenizer boundaries with adjacent punctuation, violating
the exact extractive-span contract. No optimizer, GPU training, R96 checkpoint,
or R96 candidate observation existed.

The bounded repair removes the compact punctuation-only relation form and
replaces it with a whitespace-delimited natural-language form carrying the
same two-hop semantics. All registered counts, training steps, inputs, and
scientific gates remain unchanged. The preflight must pass before training.
