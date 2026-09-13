# R68 preregistration: answer-constrained reasoning acquisition

R67 failed source eligibility: 1,496/2,000 answers reached its 64-token ceiling,
and every ceiling-hit failed because the long explanation ended before the
terminal class. R68 changes the measured bottleneck, not a nearby model knob:

* compact, unique nonce labels replace long hyphenated labels;
* prompts explicitly request one concise terminal class;
* the reasoning ceiling is 96 tokens, matching the earlier v2 repair; and
* evaluators accept the correct conclusion anywhere in a response so a valid
  concise explanation is not mislabeled.

R68 again contains 2,000 unique search-only prompts, eight rule phrasings, four
request phrasings, empty domain labels/claims, and no real-world facts. Its code
and catalog must be committed before the pinned Phi source is loaded. The same
source revision, CUDA capture, provenance, imported-information accounting,
segregation, and candidate-inaccessibility rules as R67 apply.

At least 1,800/2,000 source rows must pass and no source output may trigger the
locked collapse detector. Only passing search rows may be composed into a v3
segregated reasoning-training artifact. This acquisition does not certify a
host or the full ABI moonshot.

