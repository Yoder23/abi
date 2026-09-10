# R16 post-reveal assurance amendment

This amendment records repairs prompted by the blind review of commit
`b63bd55301d41f4bc352d250cef065f1450f0ef9`. It changes no source prompts,
facts, candidates, packages, metrics, gates, or original evidence.

## Accounting repair

The original v1 accounting value `unique_prompt_utf8_bytes=4,126` was the
question-only byte count. The additive v2 accounting reconstructs the exact
Qwen chat template and reports:

- 96 raw and 96 unique rendered prompts;
- 18,334 rendered-prompt UTF-8 bytes;
- 3,206 rendered-prompt token instances;
- 576 candidate-string instances and 1,146 candidate-token instances scored;
- 276 generated output tokens and 372 output bytes; and
- 576 scalar candidate scores imported by the compiler.

The old accounting receipt remains preserved.

## Public-protocol provenance repair

The historical public-v2 receipt declares a `PUBLIC_PROTOCOL.md` digest of
`4e5a60e1f3c2f5a5208f91b1f565a228b4a24fcbbe8957ab64fb4894d787156b`,
but that exact file is not present in committed history. This is not silently
treated as valid. A post-reveal assurance rerun under the committed protocol
digest `4b62aba12deb2fd5622e22e427f73b3568d532673c9ab11b230d9c77d10e3ea6`
reproduced the same public observations, extracted facts, residual tensor, fact
registry, and 48/48 gates byte-for-byte. Because this rerun occurred after the
held-out reveal, it is labeled requalification evidence and cannot repair the
original chronology by assertion.

A new held-out replication must bind this preserved public requalification
before its secret is revealed. That is the clean promotion route.

## Candidate disclosure

The compiler capsule contains no field labeled as an answer or oracle. It does,
however, receive the 12 registered candidate strings for each question, with
the correct string present exactly once, plus the teacher score for each
candidate. All documentation and certificates must describe this as
closed-candidate selection, not answer-free or answer-absent compilation.

## Verifier repair

Strict-v3 now:

- opens and hashes all seven actual live replay files;
- validates all four original/live physical extraction trees and their package
  files;
- directly rehashes the complete 10-file, 15.24 GB pinned source snapshot;
- validates the post-reveal public requalification and its committed protocol;
  and
- records both the correct-candidate disclosure and the historical protocol
  mismatch.

Hostile-v3 inherits all 15 prior mutations and rejects six new mutations of
actual live files, physical extraction results, and public requalification
evidence. The expanded total is 21/21.

These are assurance repairs. They do not broaden R16 beyond bounded,
registered, structured factual extraction and semantic segregation.
