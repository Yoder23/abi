# R29 validator-backed multi-label capability acquisition

R28 proved that quorum alone can preserve a confidently wrong teacher answer
and that one forced label cannot describe overlapping mathematics/Python
knowledge. R29 therefore changes both mechanisms.

The isolated compiler canonicalizes a small declared taxonomy (`math`,
`arithmetic`, and `geometry` to `mathematics`; `programming` and `python` to
`python`) while preserving an unknown label only when at least two views agree.
Every recognized label proposed by any view is retained, so a fact may carry
multiple domain tags. No tag is ever `english`. Facts may consequently appear
in multiple domain packages; the duplication is fully accounted.
For a safely validated expression, the compiler also adds `python` when the
source questions explicitly say Python and `mathematics` otherwise. This
declared content classifier is not presented as autonomous ontology discovery.

Before answer quorum, a pure-stdlib AST validator executes only a tightly
whitelisted expression grammar with no builtins, imports, attributes other than
the registered string methods and `__name__`, or names beyond deterministic
math/Python primitives. The physical pivot-root, mount, and network isolation
still apply. When validation succeeds it supersedes source voting; otherwise a
unique 2-of-3 semantic answer quorum is required. Validation is recorded per
fact. This is normalization and verification, not pure teacher imitation.

The disclosed prerequisite replays the failed R28 hidden rows and must correct
`bool([0])`, retain both mathematics and Python tags for `2 ** 10`, and answer
36/36 evaluation rows. The fresh held-out universe is disjoint from R27/R28.
Its source, compiler, package, LayerCake CPU/GPU, English-immutability,
lifecycle, corruption, physical-isolation, and negative-control gates are
frozen before a fresh selection.

A pass proves a bounded teacher-probe, semantic-label, independent-validation,
and LayerCake-import pipeline, including bounded output better than the source
where validation corrects it. It does not prove exhaustive source diagnosis,
arbitrary-domain validation, fluent English extraction, global minimality, or
superiority to LoRA/distillation.
