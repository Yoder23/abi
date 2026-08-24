# Security policy

ABI is alpha research software. Only the latest commit on `master` receives
security fixes.

## Report privately

Do not open a public issue for a suspected vulnerability. Email
`samyoder23@gmail.com` with subject `[ABI Security]`, the affected revision,
impact, reproduction steps, and any proposed mitigation. Expect an initial
response within seven days.

## Security boundaries

- Treat teacher models, generated text, manifests, archives, and capability
  packages as untrusted input.
- Pin model revisions and verify weight hashes before extraction.
- Do not enable `trust_remote_code` for promotion-eligible sources.
- Never load pickle-based weights from an untrusted party. Prefer
  `safetensors` and verify the recorded digest.
- Extraction may execute on GPUs and access model credentials. Use a dedicated
  environment with least-privilege tokens and no unrelated secrets.
- ABI artifacts are not authenticated deployment packages. LayerCake owns its
  package signature and installation policy.
- Historical campaign packages use reproducible research signing custody and
  must not be treated as production-authenticated merely because their inner
  signatures verify. The final-mile failed-candidate bundle adds an independent
  outer Ed25519 signature, but current LayerCake installation does not yet
  enforce that outer signature. Production promotion is prohibited until it
  does.

The supported compiler surface validates paths, canonical hashes, provenance,
labels, selections, and bundle membership. Historical experiment modules have
not all received the same security review and should not be exposed directly
to untrusted network input.
