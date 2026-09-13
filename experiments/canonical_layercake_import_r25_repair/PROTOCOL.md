# R25 semantic agreement scorer repair

R25 remains an immutable failed preregistered run. Its direct LayerCake outputs
were 144/144 exact against the canonical factual answers, while teacher byte
agreement was 135/144. All nine disagreements were the predeclared answer
`Reykjavik` versus Qwen's Unicode spelling `Reykjavík`.

R16 had already frozen and used NFKD accent-insensitive, punctuation-insensitive,
case-insensitive answer normalization. This additive repair changes only the
R25 teacher-agreement scorer to that pre-existing R16 function, then performs
a fresh full live replay. Candidate packages, source rows, prompts, host code,
and every other gate remain unchanged.

Because the mismatch was inspected before this repair, a repaired pass is
post-hoc diagnostic evidence and requires a new preregistered hidden
replication before promotion.
