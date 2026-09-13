# R52 preregistration: sparse payload isolation

R51 proved that the broad English archive carries useful reasoning signal but
that updating all shared weights creates cross-capability interference.  R52
makes the measured architectural correction: restart from the proven R47
parent, keep the shared English transformer, embeddings, language head, and
canonical ABI bit-exact, and train only the ten already installed rank-64 task
cakes plus their existing classifier.  The external R49 router remains the
deployment router.

No R49/R50 output or evaluator outcome is training material.  R51 weights are
not a parent and are not copied.

## Frozen inputs

- R47 parent checkpoint:
  `65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b`;
- R47 metadata:
  `9590374b0afd3184dd75bcf08d8b9f0876ed7bcc0db7f7ec1f4abf147093d1d6`;
- broad archive:
  `82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150`;
- anchor archive:
  `f6a27cae529d990ba67f1a26bb9cd79f97ba5ca0965c9debd0fc98dab8dba820`;
- canonical ABI:
  `d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761`;
- external router:
  `20ae6562432cc59ece5b9cb552caaa16da4824323e62460c6c33feb0579df863`.

## Frozen training

- seed 52,001;
- `task_cakes_classifier` scope on CUDA;
- 6,000 successful steps, main microbatch 8 and anchor microbatch 4;
- complete broad and anchor budgets with balanced capability sampling;
- shared learning rate 2e-5, cake learning rate 1e-4;
- classifier loss 0.25, prompt-overlap loss 1.0, weight decay 0.01;
- frozen-parent top-64 preservation weight 0.5;
- max tokens 256, with only the same two enumerated overlength broad rows
  excluded; and
- autonomous-prefix recovery at the unchanged 400/every-8/[8,16,32]
  schedule.

The source teacher is absent.  The deployed graph retains the existing ten
physical task-cake paths and activates only the externally selected route.

## Development decision

On disclosed validation, R52 must achieve at least 1,277/1,400, at least
65/100 per capability, at least 94% source-pass retention, zero collapse and
generation errors, exact routing, and 1,400/1,400 physical single-cake rows.
Only then may it be screened on disclosed R50 final-test, where it must achieve
at least 1,261/1,400 and satisfy the same remaining gates.  These disclosed
screens cannot promote R52.

If both pass, freeze R52 and earn a new preregistered catalog lineage with a
live pinned-source comparator.  If either fails, close this ten-route sparse
branch; a successor must remove the measured route-collision bottleneck rather
than sweep learning rate, step count, or nearby loss weights.

R52 alone cannot establish unrestricted English, global minimality, human
quality, independent hardware, or LoRA/distillation superiority.  The ABI
moonshot remains open.
