# Phase 3 V17 self-prefix result

Status date: 2026-08-06

V17 is complete and failed. Both candidates continued from the exact V11 C0
checkpoint, used the same 4,000-record sequence, executed the same frozen-policy
corruption forwards, and exposed exactly 1,162 recovery events covering 15,199
continuation tokens. Every LayerCake tensor and the corruption policy remained
unchanged; only the registered 1,057,798 bridge parameters changed.

| System | Functional passes | Rate | Collapses |
| --- | ---: | ---: | ---: |
| S0 self-prefix recovery | 1,185/1,400 | 84.64% | 64 |
| S1 compute-matched continuation | 1,208/1,400 | 86.29% | 56 |
| Prior V11 C0 | 1,207/1,400 | 86.21% | 51 |

S0 is significantly worse than S1: -1.64 points with a paired stratified
bootstrap 95% interval of -2.79 to -0.43 points. It is also significantly worse
than its V11 C0 parent: -1.57 points, interval -3.07 to -0.07. Its 64 collapses
fail the zero-collapse gate, and all remaining quality gates fail.

The failure refines the mechanism. A mismatch against one cached teacher token
is not necessarily a behavioral error; fluent alternatives are common. V17
forced valid alternative prefixes back onto one reference continuation and
damaged fluent realization, conversation, and coherence. This does not negate
the V16 teacher-payload signal or identify a LayerCake regression.

V17 is closed and remaining seeds are prohibited. If another recovery branch
is preregistered, it must select only an objectively invalid event supported by
the collapse metric—such as a wrong token that repeats a recent teacher token—
and retain a compute-matched control. No loss-weight, step-count, horizon, or
nearby arbitrary-mismatch sweep is supported.
