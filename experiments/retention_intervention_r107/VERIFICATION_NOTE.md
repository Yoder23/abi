# R107 post-run verification notes

The frozen execution and scalar verifier are unchanged. After all 18 model
processes and exact raw replay completed, the additive `complete.py` verifier
also reconstructs every recipient active/neutral state and every intervention
digest from the original published safetensors. This strengthens custody; it
does not change an experimental gate or response.

An initially running completion check was slow. A nonblocking `py-spy 0.4.3`
stack read on the owned verifier process showed the main thread in
`transformers/tokenization_utils_fast.py::__len__`, called from the per-token
range check in frozen `core.py::validate`. Repeated vocabulary-size queries
were pure validation overhead. The first process was not stopped.

The additive completion wrapper now caches the tokenizer's immutable vocabulary
size once and delegates all other operations unchanged. Source/tokenizer files
are still hash-checked. Equivalence/rejection tests cover the proxy; all raw
rows, hashes, state reconstruction and thresholds remain mandatory. This is
not an inference, acquisition or model-speed optimization. The diagnostic
profiler was installed in a separate workspace tools directory, not either repo.
