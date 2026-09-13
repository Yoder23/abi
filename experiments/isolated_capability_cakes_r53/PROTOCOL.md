# R53 preregistration: six-block isolated capability cakes

R52 proves the shared core can remain immutable during ABI payload training,
but its ten canonical routes collide across fourteen labeled capabilities.
R53 removes exactly that measured bottleneck: expand the R47 six-block host
from ten rank-64 task cakes to fourteen rank-64 capability cakes, copying each
new cake and classifier row bit-exactly from its canonical parent route before
training.  Only those cakes and the 14-way classifier may train.

## Implementation contract

- Add a distinct versioned six-block/14-cake ABI configuration; do not alter
  the existing three-block architecture identity.
- The expansion must accept only a six-block, ten-route, rank-64 parent.
- Before training, every capability cake and classifier row must equal its
  declared parent route exactly, so expansion alone is function-preserving.
- Embeddings, all six transformer blocks, final normalization, and output head
  remain frozen and bit-exact.
- At inference, exactly one of fourteen cakes is physically selected.
- Unit tests must cover valid six-block construction, wrong-version rejection,
  exact copied initialization, and unchanged three-block behavior.

The external route package is also expanded from 10 to 14 outputs.  It uses
the unchanged lowercase unigram/bigram SHA-256 feature map with 1,024 bins and
trains only on the catalog's 1,400 search capability labels—never responses,
teacher outputs, validation outcomes, or final-test outcomes.  It must achieve
exact search, validation, and final-test label routing with zero rotated-label
matches before a behavioral screen.

## Frozen acquisition

- parent, broad archive, anchor archive, and canonical ABI hashes are exactly
  those registered for R52;
- seed 53,001;
- `capability_cakes_classifier`, CUDA, 6,000 successful steps;
- main microbatch 8, anchor microbatch 4, balanced capability sampling;
- shared/cake rates 2e-5/1e-4, classifier 0.25, prompt overlap 1.0,
  parent preservation 0.5, weight decay 0.01;
- max tokens 256 with the same two enumerated broad exclusions; and
- autonomous recovery 400/every-8/[8,16,32].

## Decision rule

Disclosed validation must reach at least 1,277/1,400, at least 65/100 for
every capability, at least 94% source-pass retention, zero collapse/errors,
exact routing, and 1,400/1,400 physical one-cake execution.  If it passes,
disclosed R50 final-test must reach at least 1,261/1,400 under the same gates.
Neither disclosed split can promote R53.

Passing both authorizes one new prospective catalog/source capture after the
checkpoint and router are frozen.  Failure closes the shallow post-transformer
capability-cake representation; nearby hyperparameter sweeps are prohibited.

R53 cannot by itself establish unrestricted English, global minimality,
human quality, independent hardware, or LoRA/distillation superiority.  The
ABI moonshot remains open.
