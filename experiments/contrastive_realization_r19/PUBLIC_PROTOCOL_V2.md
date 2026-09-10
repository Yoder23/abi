# R19 disclosed-development protocol v2: fail-closed control handling

R19 v1 is preserved as `FAIL_CONTROL_COMPILATION`. Its primary physical
compiler completed, but the mood-permuted evidence contained no compatible
positive/negative structural pair, so the compiler correctly rejected it.
The v1 runner had not registered that rejection as a control outcome.

V2 changes only control handling. If and only if the isolated control compiler
fails with `no structure-compatible polarity contrast`, V2 records a physical
compiler rejection, emits no control package, and treats package execution as
absent/abstaining on all 48 control rows. Any other physical failure aborts.
The primary worker, representation, source evidence, package schema, semantic
evaluator, row counts, 48/48 quality gate, zero-regression gate, modal baseline,
removal gate, and 10% maximum control exactness are unchanged.

Both evidence sets are disclosed development data. A pass authorizes only a
new preregistered hidden lexical replication and proves no unrestricted English,
LayerCake acceptance, minimality, autonomous labeling, domain extraction, or
LoRA/distillation superiority.
