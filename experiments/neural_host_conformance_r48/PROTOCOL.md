# R48 preregistration: labeled router and termination conformance

R47 exceeded the stored source on the complete 1,400-row functional matrix but
failed exactly two registered requirements: 25 instruction-following prompts
routed to cake 8 instead of cake 4, and three abstention generations repeated
the word `generation` until the collapse threshold fired.  R48 performs no
new full-core training.  It may update only the 7,690-parameter task classifier
from already public capability labels and may tighten only the generic lexical
termination guard.

## Frozen candidate and data

- R47 checkpoint SHA-256:
  `65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b`.
- R47 metadata SHA-256:
  `9590374b0afd3184dd75bcf08d8b9f0876ed7bcc0db7f7ec1f4abf147093d1d6`.
- R47 evaluation SHA-256:
  `5532b36cc0f12883f3d6da30b6aa428d4d23d611f59d2782fd16d5020d2d481c`.
- catalog SHA-256:
  `8992c7de94d3733d66f2083f96ec8d3cba31be3943afe1849f42220f33ff8d08`.
- router fitting may use only the catalog's 1,400 `search` prompts and the
  fixed public `CAPABILITY_TO_ROUTE` labels.  Validation prompts and outputs
  are forbidden during fitting.

## Frozen router fitting

- seed 48,001;
- frozen transformer, embeddings, output head, and all ten task cakes;
- task classifier weight and bias are the only trainable tensors;
- AdamW, learning rate 0.002, weight decay 0.01;
- 20 epochs, batch size 64, deterministic seed-shuffled batches;
- mean prompt-hidden summary using the unchanged LayerCake classifier input;
- no teacher calls, outputs, logits, activations, or response targets.

The fitted router must reach 100% on the search prompts and at least 99% on
the separately held validation prompts.  Every non-classifier tensor must be
byte-identical before and after fitting.  A deterministically rotated route
mapping must score no more than 20% under the fitted classifier.

## Integrated rescreen

The same checkpoint is regenerated on all 1,400 validation prompts using
automatic routing, neural greedy decoding, and lexical truncation at the first
repeated lexical four-gram (threshold zero).  No output template, planner,
post-hoc answer replacement, or symbolic surface handler is permitted.

The integrated screen passes only with at least 0.90 functional accuracy,
at least 0.65 in every capability, no negative aggregate difference from the
stored source, at least 94% retention on source-passing rows, zero collapse,
zero generation error, at least 99% automatic route accuracy, one physically
active cake per sequence, and unchanged source/candidate inputs.

This is a development conformance repair.  A pass authorizes a new prospective
teacher-relative replication and minimization ladder; it is not itself the ABI
moonshot or proof of unrestricted English or superiority to other methods.
