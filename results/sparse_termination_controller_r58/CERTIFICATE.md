# R58 sparse termination controller: negative certificate

R58 is a preregistered negative result. A 172,494-parameter route-selected
rank-16 controller was trained on every terminal-observable broad and anchor
record while the exact R55 host remained frozen. The controller could stop but
could not rescore, select, replace, or rewrite tokens.

- controller SHA-256:
  `3c9a7f5bc45b3a5eb5183ec29cd994c083b3d4fbfdfcb3898c17f82848b9ee22`;
- metadata SHA-256:
  `0de887077a3feee10e01f27e6d98e253e0ae87062bece6bce991ee43c92cc5b8`;
- 1,200 successful CUDA steps in 126.1 seconds; all 16,134 broad and
  1,438 anchor records observed;
- disclosed validation: 1,236/1,400 candidate versus 1,220/1,400 source and
  1,277/1,400 parent gate;
- source-pass retention: 0.9229508196721311;
- exact routing and model sparse execution: 1,400/1,400;
- termination decisions: 36,891; controller stops: 1,364; natural model EOS
  stops: 24; collapse: 12/1,400.

Functional, parent-nondegradation, source-retention, and zero-collapse gates
failed. Final-test was not run. The teacher-forced rank-16 hidden-state
controller and threshold/rank/step/rate variants are closed. The full ABI
moonshot remains open.

Raw evaluation SHA-256:
`902a82d3c8ba2107a0f874ff03d3cbea6bdaaa23099e7b41531c2b14e5ac5511`.
Result file SHA-256:
`e835aed6adce3312828262b59ce3579c9d1da879d56248c8a6bd3672bf595a6c`.
