# R21 label-interface repair

The frozen R21 source run produced 583/600 exact registered labels. All 17
failures were one repeated prose instruction and all returned `Analyst case-`,
the beginning of the requested output, rather than any competing label. The
other five tasks and the other five prose instructions were exact.

This one repair changes only label observation. It reuses all 600 immutable
teacher responses byte-for-byte and replaces free label generation with one
prompt-end forward pass. The six registered labels are mapped to single-token
letters A through F; prediction is the argmax over only those six source logits.
All six restricted probabilities and the selected label are preserved per row.

The repair passes only if at least 594/600 labels match the already registered
ontology and all 600 predictions are in the ontology. No response regeneration,
prompt wording change, generator change, model-size change, quality-gate change,
or evaluation access is authorized. If it passes, the original R21 training and
evaluation settings may execute through an additive wrapper bound to both the
failed source run and this repair.
