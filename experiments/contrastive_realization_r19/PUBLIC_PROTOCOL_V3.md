# R19 disclosed-development protocol v3: bind control rejection evidence

R19 v2 passed both disclosed development quality matrices, but is not promoted
because its receipt's control-rejection status was not backed by a separate
raw, content-hashed rejection record.

V3 changes only evidence preservation. On the already authorized exact
structural rejection, the runner writes a fail-closed record containing the
input bundle hash, isolated-worker hash, complete exception text, zero package
count, and its own evidence hash. The receipt binds that file by SHA-256. Any
other exception aborts. Primary compiler, package, source data, evaluator,
factorization, row counts, and numerical gates remain unchanged.

A v3 pass still uses disclosed development data and authorizes only a new
preregistered hidden lexical replication. All broader ABI claims remain open.
