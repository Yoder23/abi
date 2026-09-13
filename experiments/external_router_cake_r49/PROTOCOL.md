# R49 preregistration: external sparse label-router cake

R48 showed that capability routing labels are perfectly learnable on search
but that the host's mean-hidden linear classifier misses one recurring
instruction subtype on validation.  R49 changes the interface, not the English
model: it builds a separately packaged, sparse hashed-text router from the same
public search labels and passes the selected canonical route through the
LayerCake model's existing `task_routes` input.

## Frozen inputs

- R47 checkpoint:
  `65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b`.
- R47 metadata:
  `9590374b0afd3184dd75bcf08d8b9f0876ed7bcc0db7f7ec1f4abf147093d1d6`.
- R47 evaluation:
  `5532b36cc0f12883f3d6da30b6aa428d4d23d611f59d2782fd16d5020d2d481c`.
- catalog:
  `8992c7de94d3733d66f2083f96ec8d3cba31be3943afe1849f42220f33ff8d08`.
- immutable source bundles:
  `85abf114ffd6455d589d8b55b740becd920bc2e4dd4f3f62fddffe4b1708af2c`
  and `fe9af246e28efdf57e234e9384fa04c21974f997fe753ac0975d67b1d0c6ecea`.

## Frozen router

- input: Unicode-casefolded word unigrams and adjacent bigrams; digit runs are
  normalized to `<num>`;
- feature hashing: SHA-256 into 1,024 signed float32 bins;
- head: one linear 1,024-by-10 classifier (10,250 parameters);
- fit data: exactly 1,400 catalog `search` prompts and public
  `CAPABILITY_TO_ROUTE` labels;
- seed 49,001; full-batch AdamW; 300 steps; learning rate 0.05; weight decay
  0.001;
- no teacher calls, responses, source logits, hidden activations, or validation
  examples.

The router gate is 100% search and validation route accuracy, at most 20%
accuracy against the one-slot rotated label control, and byte-identical R47
checkpoint/metadata before and after fitting.

## Integrated screen

All 1,400 validation prompts are regenerated from the untouched R47 neural
checkpoint.  The external router supplies one canonical route to the existing
LayerCake `task_routes` interface.  Greedy decoding is unchanged except that a
generic lexical guard truncates at the first repeated novel lexical four-gram
(runtime threshold 1).  No symbolic output, template, planner, retrieval, or
answer replacement is permitted.

Pass requires:

- at least 1,260/1,400 functional and at least 65/100 per capability;
- candidate functional count at least the stored source count;
- at least 94% retention on source-passing rows;
- zero collapse and zero generation error;
- 1,400/1,400 route accuracy;
- one physically called task cake per model invocation;
- exact frozen core identity; and
- fail-closed recomputation from raw rows.

A pass establishes a bounded label-router plus neural-host construction and
authorizes a fresh prospective replication.  It does not establish
unrestricted English, autonomous ontology discovery, global minimality,
LoRA/distillation superiority, or the full ABI moonshot.
