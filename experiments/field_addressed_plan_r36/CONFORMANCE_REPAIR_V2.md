# R36 v2 fixed-head leakage repair

R36 v1 is preserved at result SHA-256
`0da58e99d783aa16b3ff65008725439abd2cbd442407039510f927fb0a19f9dc`.
Its raw outputs repeatedly emitted training-set numbers through the fixed
literal head even though numeric training targets used field pointers.  The
implementation had incorrectly added every prompt number to the fixed
vocabulary, contradicting the registered requirement that supplied numbers be
emitted as field-position actions.

V2 removes source numbers from the fixed vocabulary, maps every uniquely
supplied target number to its field pointer, and permits only structural output
digits one through three without a source pointer.  A non-structural target
number without exactly one source position is quarantined.  Nothing else is
changed: source rows, shortest-representable selection rule, model, optimizer,
seed, batches, fixture, metrics, and thresholds remain fixed.  V1 remains
negative evidence.
