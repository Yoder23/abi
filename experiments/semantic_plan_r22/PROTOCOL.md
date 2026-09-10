# R22 semantic-plan normalization prerequisite

R21 proved a bounded public generative transfer mechanism but failed hidden
summary retention. Inspection shows the seven misses are fluent paraphrases of
the supplied propositions. Four analogous teacher outputs also fail the same
lexical retention scorer. R21 remains failed and is not rescored.

R22 tests a materially different acquisition representation: ABI preserves the
raw teacher response, constructs an explicit task-and-field semantic plan, and
asks the same pinned source for one canonical transfer target that retains each
supplied value exactly while satisfying the task. No teacher label is requested;
the already frozen registered-ontology labels are reused.

The public normalization stage reuses all 600 R21 prompts and raw responses.
It makes exactly 600 new source calls and stores every normalized target, raw
target, prompt hash, token count, byte count, quality score, and source cost.
It passes only at:

- at least 594/600 functional normalized targets;
- 600/600 non-hallucinating normalized targets;
- at least 594/600 non-collapsed normalized targets; and
- 600/600 targets containing every DATA field value exactly once, verbatim, and
  in field order; and
- 600/600 unchanged registered labels and raw-response identities.

No row deletion, retry, sampling sweep, prompt repair, model change, gate
change, evaluation access, or training is permitted if this prerequisite
fails. If it passes, the unchanged R21 three-method/three-seed bakeoff may train
on the normalized targets, with the corrected LayerCake package manifest. A
positive result still requires fresh live replay and a new committed hidden
replication. General English and the ABI moonshot remain open.
