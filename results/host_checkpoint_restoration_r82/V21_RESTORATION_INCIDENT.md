# R82 v21 restoration incident

The intact v6 checkpoint was reproduced and restored exactly. The original
v21 runner was never committed as an independently recoverable source tree;
only its SHA-256 (`e4f85e28208b38278c988d83acc97767cbe5da02cd433bfc973e23fb5116157`)
survives in the preregistration and training result.

The first surviving post-interval runner stopped before training because a
later accounting path calls `int(None)` on the unchanged preservation corpus.
A diagnostic reconstruction restored the earlier zero-accounting behavior.
It matched the historical step-1 losses exactly, proving input, seed, parent,
and initial execution alignment, then diverged:

| step | historical total loss | reconstructed total loss |
|---:|---:|---:|
| 1 | 4.594590187072754 | 4.594590187072754 |
| 100 | 2.8020660877227783 | 2.802673578262329 |
| 200 | 2.3564374446868896 | 2.35526704788208 |

The process was interrupted at step 300. It created no output directory or
checkpoint. Approximate reconstruction is rejected; downstream v51 restoration
is closed unless its exact bytes are recovered independently.
