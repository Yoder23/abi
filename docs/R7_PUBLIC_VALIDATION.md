# R7 public validation record

Updated: 2026-08-30

## Immutable public release

- Tag: `abi-final-validation-v2-repaired-r7-2026-08-30`
- Commit: `3f82a9f4d67dda5c8ea13bd59b2d8f1bbd3dd128`
- Release archive: `abi-final-validation-v2-r7-2026-08-30.zip`
- Archive size: 844,018,841 bytes
- Archive SHA-256:
  `fc50f423986149b5d4670ec9e28698540f64be96034efa26e5704c4469921e88`
- Embedded manifest SHA-256:
  `c44a5a57185042006570010d0bfd3d6600daebcf7ca096331f3161705992b8b9`
- ZIP inventory: 1,192 payload files plus the embedded manifest

## Public packages

| Package | Bytes | SHA-256 |
| --- | ---: | --- |
| English | 253,216,208 | `acb787b3ffa0153c57d88cd37ba81c3f00b370d4ca4937e659cd4c775851f25d` |
| Python | 448,404 | `f1defaef2771ced336a332572a2d2f0e1e542399c877d182c48a6cd2e199231d` |
| Civics | 495,919 | `634ce66958859ec36dc1fbdf5ef34d6d2a9949d10cf2348a68c245d8c325d604` |
| Chemistry | 510,981 | `f9c9b2668fda5ef6b92844c1b7097fbdf8ff0daaae51f5b86f72d4a49000abeb` |

## Clean reconstruction

The official reconstruction cloned the exact public tag, created a clean
Python 3.10 environment, installed the published lock, downloaded only the
manifest-listed assets, verified every hash, extracted the release, reproduced
the strict certificate, and passed 17/17 focused tests. A deep extracted path
exceeded 260 characters on Windows and passed with long paths disabled.

## Fresh blind review

An ephemeral Codex reviewer independently repeated public reconstruction,
audited every archive entry and all 301,543 physical rows, reran the complete
19-case hostile suite, verified tag/archive/certificate bindings, and exercised
12 valid/invalid USTAR/V7 arbitrary-prefix scanner controls. Its final bounded
verdict was `PASS`.

Two earlier reviewer attempts failed closed because the reviewer-created
environment omitted published dependencies. Those setup failures are preserved;
the complete-lock run passed. R5 and R6 remain preserved because blind review
found substantive archive-detection and Windows-path defects in those lineages.

Machine-readable receipts are in
`results/abi_final_validation_v2/blind_redteam_r7/` and beside the strict
certificate. This record opens human and independent-hardware review; it does
not claim either is complete.
