# R6 blind red-team: failed

Date: 2026-08-30

Target:
`abi-final-validation-v2-repaired-r6-2026-08-30` at
`415474d660a3f4d433546a23728693cadf542405`.

The fresh blind Codex review independently cloned the public tag, queried the
GitHub release, verified the five published payload hashes, reconstructed the
archive in a short clean directory, and recomputed the strict certificate.
Those checks passed. The reviewer also independently confirmed that deletion
of the ordinary `/usr/bin/NF` reachable-filesystem inventory row was rejected
by the release-source inventory commitment.

R6 is nevertheless rejected. A benign synthetic USTAR stream containing
`manifest.json`, `tensors.safetensors`, and `signature.json` was placed after
513 arbitrary prefix bytes. Python's tar reader enumerated the three members,
while the R6 content scanner reported zero capability signatures, zero archive
members, and zero unsupported-container findings. The same bypass reproduced
at offsets 1,024 and 8,388,608. R6 therefore did not establish that capability
archives were physically absent from every reachable file.

The official public-reconstruction command also failed when its scratch path
made one bundled historical member exceed the default Windows path limit. The
same exact assets reconstructed from a shorter directory, so this is a
workflow portability defect rather than missing public bytes.

The review continued through additional inspection and began an independent
19-mutation hostile run, but the Codex service safety filter terminated the
session before it could emit its requested final verdict file. That
termination is not treated as a pass or as completion of the hostile run. The
independently reproduced tar bypass is sufficient to classify R6 as failed.

R6 remains immutable historical negative evidence. R7 must add arbitrary-offset
tar checksum detection, Windows extended-length extraction, new physical
certification evidence, new strict and hostile receipts, a new public release,
and a fresh blind review.
