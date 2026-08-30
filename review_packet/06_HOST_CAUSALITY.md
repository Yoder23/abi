# Host causality

R7 performs 1,024 fresh executions per host, 3,072 total, under real-host,
neutral-host, zero-state, random-state, shuffled-state, host-removed,
adapter-removed, and capability-removed conditions.

The executions use 24 distinct condition processes and bind 733 transitive
source files. Positive conformance states preserve canonical capability bytes;
host/adapter/capability removal fails in the registered way. The verifier reads
live raw rows rather than trusting a replayed pass flag.

This establishes causal dependence for the registered runtime interface. It is
not a claim that foreign hidden states independently contain the capability.
