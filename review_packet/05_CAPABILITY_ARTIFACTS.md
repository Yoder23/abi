# Capability artifacts

The public R7 release publishes immutable English, Python, civics, and chemistry
packages. Their exact names, byte counts, hashes, runtime class, and evaluation
locks are bound in the public manifest and R7 candidate.

The same package bytes are used in all declared host cells. No package is
silently regenerated during certification. See `docs/R7_PUBLIC_VALIDATION.md`
for authoritative public hashes.

These artifacts validate the capability ABI execution boundary; they do not by
themselves prove a general teacher-to-artifact extraction method.
