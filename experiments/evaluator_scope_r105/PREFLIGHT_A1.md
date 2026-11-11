# R105 pre-inference custody repair

2026-11-11, before any model forward. The first `freeze` command failed with
FileNotFoundError while hashing `experiments/__init__.py`: this is a namespace
package and that file does not exist. Preparation completed and wrote the full
catalog, but no freeze or inference receipt was created. Preserve that catalog.

The additive implementation repair binds optional initializer files only when
they actually exist (as older runners already do). A repeated pre-inference
freeze may reuse an existing catalog only after exact equality with the source-
derived population; it cannot overwrite or change the catalog. The successful
freeze also binds this note. No model, labels, populations, gates or inference
settings change. Do not create a new shared package initializer or alter any
historical source. A committed successful freeze is still mandatory before GPU
inference. The first operational failure is not a scientific result.

Metadata discovery also encountered one corrected upstream 404 (test files are
under `scone_nli/test/`, not its parent) and a sandbox DNS failure reading model
configuration; the same public configuration read succeeded with approved network
access. These occurred before any model predictions and changed no selection.
