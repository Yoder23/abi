# R41 preflight repair V1A

The first invocation stopped before creating the immutable output directory.
The scientific intervention did not start. The sandbox process has a different
Windows security identifier from the owner of the sibling LayerCake repository,
so Git rejected the read-only `rev-parse HEAD` commit check as dubious ownership.

The only repair is to pass Git's command-local
`-c safe.directory=<exact resolved LayerCake root>` option for the two read-only
commit checks. This does not change global Git configuration. No prerequisite,
component, package, fixture, model, runtime, device, scorer, gate, or threshold
changes. R41 remains bound to LayerCake commit `f8a4569`.

