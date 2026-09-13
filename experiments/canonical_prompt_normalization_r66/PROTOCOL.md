# R66 preregistration: complete canonical host-prompt normalization

R65 completed training but its fail-closed receipt rejected the run because the
broad archive contains two policy suffix variants rather than one. The invalid
checkpoint and its hashes are preserved; it will not be screened or reused.

R66 repairs only that enumerated inventory defect. Before LayerCake tokenization
it removes either of the two exact source-extraction suffixes:

* 19,337 rows: broad English-form acquisition policy;
* 4,769 rows: supplied-text linguistic policy; and
* 315 rows: already canonical and byte-unchanged.

Both suffix strings and counts are fail-closed. The output, source provenance,
capability label, record ID, and immutable archive remain unchanged.

R66 restarts from untouched R47 and otherwise copies the frozen R65/R55 run:
CUDA, seed 66,001, 6,000 successful steps, six-block rank-32 sparse adapters,
batch 8, anchor batch 4, balanced sampling, rates 2e-5/1e-4, classifier 0.25,
prompt overlap 1.0, parent preservation 0.5, weight decay 0.01, max 256, and
autonomous recovery 400/every-8/[8,16,32]. There are zero new teacher calls,
teacher outputs, R60 outputs, logits, activations, or evaluator outcomes.

The R65 development decision rule is unchanged. R60 remains non-promotional;
coherence and format remain non-rescorable. A new prospective catalog is
authorized only if R66 preserves at least 94% of source-passing adjudicable
rows, reaches 65/100 on every unchanged capability, materially exceeds R59 on
the 1,000 unchanged rows, has zero final collapse, and physically executes only
one selected route on all 1,400 rows. The full ABI moonshot remains OPEN.

