# R19 disclosed-development protocol v4: output publication completeness

R19 v3 preserved raw control-rejection evidence, but its physical wrapper
returned from the second primary extraction before the package file became
visible across the WSL/Windows filesystem boundary. The result and package
subsequently appeared complete. The partial run is `NOT_RUN` rather than a
scientific failure.

V4 changes only the launcher boundary. After WSL exits successfully, it waits
for the declared package, then validates its byte count and SHA-256 before
returning. No compiler, representation, package, input, evaluator, control, or
gate changes. All v3 evidence-binding requirements remain.

Both evidence sets remain disclosed development data. Passing authorizes only
a fresh preregistered hidden lexical replication. All broader claims remain
open.
