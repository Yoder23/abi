# R52 sparse-payload isolation certificate

Verdict: `FAIL_R52_DISCLOSED_SCREEN`

R52 is preserved negative evidence and is not a promoted artifact.

R52 restarted from the R47 parent and trained only 1,006,090 parameters in the
ten existing rank-64 task cakes and classifier.  All 81,912,576 shared-core
parameters remained bit-exact, graph topology did not change, and at most one
cake was active.  The fixed 6,000-step endpoint completed in 1,135.293 seconds
with 2,478,569,472 peak allocated CUDA bytes.  Its checkpoint SHA-256 is
`5ef352f6638f17d3df96e8dae3be160648726f371caf54d29afa1ec835edd58b`.

On disclosed validation it scored 1,240/1,400 versus 1,277 for its parent and
1,220 for the stored source.  Source-pass retention was 92.2131%.  Tone
control was 64/100, below the 65 minimum, and one clarification output met the
collapse criterion.  Functional, parent nondegradation, per-capability,
source-retention, and zero-collapse gates failed.  Routing and physical
single-cake execution remained exact on all 1,400 rows.  The disclosed
final-test screen was not run.

This closes nearby training of the existing ten-route cake topology.  Its
capability collisions are now a measured bottleneck: prompt grounding,
instruction following, tone control, and format control share route 4;
conversation and rewriting share route 8; clarification and abstention share
route 7.  A successor must separate these capability payloads rather than
sweep the R52 learning rate, loss weights, or step count.

Evidence:

- metadata SHA-256:
  `b18882aaba781e3bef0f5b7a29d6b1e0668c7c9fdf900f34b30fcf7e39687e4c`;
- validation rows SHA-256:
  `d4490ddcdc1d810a8ad9829325357bba9336da5e881d6ba2e186e6bd688f0cbe`;
- validation result SHA-256:
  `ad10b83bd046aa3699c413973b7e74583af7cf763e00d2ad58346d63f93741e7`.

The ABI moonshot remains `OPEN`.

