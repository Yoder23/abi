# R30 v2 capability-set diagnosis repair

The frozen R30 v1 source run failed its per-row free-label gate. Inspection of
the complete immutable rows shows that the source usually named subject matter
(`completion-date`, `surprise-resolution`, `glim`) rather than the reusable
language operation. No response row is deleted, regenerated, or relabeled.

V2 tests one materially different diagnosis interface. The source receives the
36 distinct public instructions in a deterministic shuffled order under opaque
content-derived IDs. It is asked to group behaviorally equivalent instructions
and invent one abstract capability name per group. It receives no task names,
label choices, expected grouping, facts, responses, or success IDs.

The raw diagnosis must be strict JSON, assign every opaque instruction ID
exactly once, create exactly twelve groups of three, and yield twelve distinct
ASCII lowercase slugs. Oracle purity is measured only after the diagnosis has
been frozen. A failure closes this public interface; it does not authorize
manual regrouping or another prompt retry.

