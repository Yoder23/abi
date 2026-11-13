# R107 preparation custody

Before registration/freeze or model execution on 2026-11-11, two preparation
attempts failed on Hugging Face's `model_info(..., files_metadata=True)` endpoint:
HTTP 429, `maximum queue size reached`. Neither attempt created the results
directory or performed model inference. Request IDs ended in
`5a669d60-61d1-4032-b16b-603cad51b47a` and
`227aef95-2c83-4d2e-8aa4-b427108be5c7`.

The pre-freeze preparation implementation now reads the small public Git-LFS
pointers at the exact same revisions and requires exact local SHA-256 AND byte
size matches. Both pointers/URLs are retained in provenance.json. No source,
dataset, selection rule, scoring rule or scientific gate changed.
