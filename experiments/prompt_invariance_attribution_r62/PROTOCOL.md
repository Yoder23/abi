# R62 preregistration: prompt-interface invariance attribution

R61 proved that correcting all 90 R60 route errors recovers zero functional
passes. R62 therefore fixes the route to the sealed capability label and tests
two prompt-only counterfactuals on all 1,400 R60 tasks:

1. remove only the capability-neutral `Evaluation item N.` ordinal while
   retaining the unseen R60 request wrapper; and
2. remove both the ordinal and the unseen wrapper, leaving the task body.

The semantic task body, evaluator, content, checkpoint, adapters, decoder, and
thresholds remain unchanged. The transformation code must prove that exactly
one registered wrapper was removed and cannot inspect answers or evaluator
values. Every output is generated live. No training is authorized.

This is a post-failure diagnostic and cannot earn promotion. A large recovery
from wrapper removal attributes R60 primarily to prompt-interface brittleness;
weak body-only results attribute it to the learned capability payload. The
full ABI moonshot remains OPEN.

