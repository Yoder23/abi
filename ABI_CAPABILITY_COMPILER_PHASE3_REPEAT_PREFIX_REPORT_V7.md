# Phase 3 V18 recent-repeat recovery result

Status date: 2026-08-06

V18 is complete and failed. It changed only V17's error selector: 99 frozen-C0
events were eligible because the wrong prediction repeated a teacher response
token from the preceding eight positions. Every ordinary teacher-token
mismatch was ignored. S0 and S1 otherwise used the same 4,000-record sequence,
1,262 recovery targets, compute topology, immutable policy, and frozen host.

| System | Functional passes | Rate | Collapses |
| --- | ---: | ---: | ---: |
| S0 recent-repeat recovery | 1,213/1,400 | 86.64% | 52 |
| S1 compute-matched continuation | 1,208/1,400 | 86.29% | 56 |
| Prior V11 C0 | 1,207/1,400 | 86.21% | 51 |

S0-S1 is +0.36 points with a paired 95% interval from -0.29 to +1.00. S0-C0
is +0.43 points with an interval from -1.00 to +1.86. Neither causal quality
gate passes. Teacher aggregate noninferiority passes, but tail capability and
zero-collapse gates fail; S0 has one more collapse than its parent.

This branch shows that objective repeat selection is safer than V17's arbitrary
mismatch selection, but 99 sparse recovery events are not the missing Phase 3
mechanism. V18 is closed, remaining seeds are prohibited, and no nearby
recovery horizon, weight, step, or selection sweep is authorized.

The next diagnostic is an oracle-fit capacity control on development material.
It is permanently non-promotional and will determine whether the frozen host
plus registered bridge can express the missing tail behaviors when acquisition
generalization is removed from the problem. It may not support a quality claim.
