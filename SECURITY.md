# Security Policy

## Supported Versions

This is a research preview (`v0.1.0`). Only the current release is supported.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅         |

## Scope

ABI is a research prototype for academic and experimental use. It is not intended for production deployment.

Known limitations relevant to security:

- `wikitext_cache.py` downloads data from Hugging Face datasets — use only in trusted network environments
- Model weights are downloaded from Hugging Face Hub (`t5-large`, `gpt2-medium`, etc.) — verify checksums independently if required
- No authentication, authorization, or input sanitization for inference — this is expected for a research prototype

## Reporting a Vulnerability

If you find a security vulnerability (e.g., a dependency with a known CVE, unsafe deserialization, or path traversal in a script):

1. **Do not** open a public GitHub issue.
2. Email `samyoder23@gmail.com` with subject line: `[ABI Security] <brief description>`.
3. Include: affected file(s), description of the issue, and any known exploit or proof-of-concept.

Expected response time: within 7 days.

This project uses `torch`, `transformers`, and `sentencepiece`. Keep these up to date to avoid known CVEs in those dependencies.
