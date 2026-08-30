# Fresh blind Codex R7 review

This directory preserves the final immutable outputs copied from the isolated
blind-review workspace. The reviewer used only the public repository, R7 tag,
GitHub Release metadata, and published assets.

| File | Purpose |
| --- | --- |
| `blind_codex_r7_report.md` | Human-readable bounded verdict and methodology |
| `blind_codex_r7_evidence.json` | Machine-readable evidence index |
| `official_public_reconstruction_receipt_complete_lock.json` | Complete published-lock reconstruction |
| `independent_archive_audit.json` | Exact 1,193-entry archive audit |
| `independent_evidence_audit.json` | Full raw evidence recomputation |
| `scanner_prefix_controls.json` | USTAR/V7 arbitrary-prefix controls |
| `full_disposable_hostile_replay_receipt.json` | Independent 19-case hostile replay |
| `tag_and_certificate_binding_audit.json` | Tag/archive/certificate binding audit |
| `strict_stdout_identity_receipt.json` | Strict stdout identity check |
| `21_final_report_validation.txt` | Final hash/status validation transcript |

The first two reconstruction attempts omitted dependencies while the reviewer
was constructing its clean environment and therefore failed closed. The third
attempt installed the complete published lock and passed. The report preserves
that sequence rather than hiding the setup failures.

The verdict is `PASS` only for the bounded R7 capability-runtime/conformance
scope. Human quality, independent hardware, minimum-information certification,
and general teacher extraction remain open.
