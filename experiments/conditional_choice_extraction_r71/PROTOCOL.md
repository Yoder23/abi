# R71 preregistration: conditional-choice weight interrogation

R67-R70 show that source free generation is the bottleneck: Phi often reasons
verbosely until the generation limit, so valid weight knowledge is not reliably
converted into a concise training record. R71 stops prompt iteration and tests a
different ABI primitive.

For every frozen R70 prompt, the source weights score exactly three continuations:
the supplied A, B, and C category codes. R71 records the mean conditional log
probability of each complete candidate code and selects the argmax without using
the evaluator or expected answer. The evaluator is consulted only afterward to
measure whether the source weights rank the derived C-code highest.

The source is pinned Phi revision
`f39ac1d28e925b323eae81227eaba4464caced4e`, loaded locally on CUDA in bfloat16.
The code, R70 catalog hash, candidate construction, scoring rule, tie policy,
batch limit, and 2,000-row matrix are committed before model access. LayerCake
and every candidate output remain inaccessible.

R71 accounts 6,000 scalar candidate scores, their storage bytes, source tokens
scored, source parameters read, wall time, GPU peak memory, prompt hashes, and
selected-output hashes. It stores no hidden activations or full-vocabulary
logit vectors.

At least 1,800/2,000 rows, every premise family, and every instruction family
must select the expected terminal code at 85% or higher, with no non-finite
score or tie. Passing authorizes packaging only the source-selected correct
search rows as a segregated reasoning acquisition artifact. R71 alone cannot
certify a LayerCake host or the full ABI moonshot.

