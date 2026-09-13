# R33 counterfactual normalization

R31's frozen-package replication failed at 130/144 because instruction wording
and generated content were correlated in the acquired source.  R32 attempted a
direct 3x3 teacher acquisition repair twice and failed closed at 21 and 22
accepted rows.  The second immutable receipt SHA-256 is
`6c9972ba6e752f07e17fb3b30746bd6dcca94d53b332441cec57262a66a80a29`.
No further teacher-prompt retries are authorized in this branch.

R33 tests a token-free normalization mechanism.  The three instruction
wordings inside each registered contract are prospectively defined semantic
paraphrases.  For each of the 576 accepted R31 training examples, construct
three rows by pairing its unchanged supplied material and unchanged accepted
teacher output with every registered instruction paraphrase for that contract.
All 1,728 constructed rows must independently pass the frozen functional
validator.  Preserve provenance from each constructed row to its original
teacher observation.  This is explicitly synthetic counterfactual
augmentation, not a claim that the teacher generated 1,728 independent answers.

Train exactly one grounded-pointer package per contract.  Architecture,
optimizer, batch size, 384-action limit, and 48-row R31 compute budget remain
fixed.  Each contract receives 25,584 sampled example exposures (48 x 533), or
1,066 batches of 24.  Thus the augmentation consumes zero additional teacher
tokens and no more learner steps than the failed R31 48-row package.

The sole development rescreen is the exact R31 v4 fixture.  Pass requires
132/144 functional, at least 11/12 in every contract, 12/12 abstention, exact
routes and contract audits, exact signed installs, and 12/12 removal rejection.
A pass authorizes one separately committed new-fixture replication.  A failure
closes this normalization mechanism.  Neither outcome establishes unrestricted
English, autonomous labeling, global minimality, arbitrary-domain transfer, or
superiority to LoRA/distillation.
