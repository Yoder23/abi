# Phase 3 Unicode-safe BPE pointer result

Status date: 2026-08-07

V54 is complete and failed. Its training-only preflight exactly reconstructed
all 7,000 targets and replaced 2,680 eligible fixed BPE actions across 2,090
records with source-position actions. Teacher data, qualified router, generator
topology, optimizer, steps, batch size, learning rate, and V50 seed remained
fixed. LayerCake was not changed.

The candidate completed 4,000 successful GPU updates in 97.15 seconds with no
skipped AMP steps. It scored 785/1,400 with 40 repetition collapses and zero
generation errors. That exactly matches V50's aggregate score, adds four
collapses, and remains 32.29 points below the teacher; the paired 95% interval
is [-34.57, -29.93] points. Training fit was also slightly worse than V50 at
96.22% action accuracy and 78.11% exact sequences.

The limited unique-identity pointer policy is closed, including remaining seeds
and nearby variants. Together, V50, V52, and V54 show that route isolation and
fixed-target representation changes do not supply enough teacher information.
The next valid Phase 3 gate is a no-training feasibility and imported-
information accounting study for richer frozen-teacher signals such as logits
or hidden capability representations. Extraction or training must be separately
preregistered after that study.

Phase 3 is not certified and Phase 4 remains locked.
