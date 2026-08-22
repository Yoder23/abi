# GitHub LFS publication boundary — 2026-08-21

The ABI working tree was clean, but `master` was 717 commits ahead of
`origin/master`. That unpublished range contained raw `*.safetensors` Git
blobs above GitHub's 100 MB ordinary-file limit, so a normal push could not
succeed.

Only the unpublished range was converted to Git LFS. The published
`origin/master` boundary was explicitly excluded. Hydrated checkpoint bytes
are unchanged; Git stores LFS pointer records while LFS stores the checkpoint
payloads.

- Pre-conversion head: `82ab8a589c7b242a932ef68b1e1a6ea89b1d6fe7`
- Post-conversion equivalent head: `ae21cba94a0e0d073eeb99cbe357e886e2f3f7d2`
- Local recovery ref: `refs/archive/pre-github-lfs-20260821`
- Converted pattern: `*.safetensors`
- LFS-managed files at the converted head: 88
- Oversized ordinary Git blobs remaining in `origin/master..master`: 0

The local recovery ref preserves the exact pre-conversion research commit
lineage. It is intentionally outside `refs/heads` and `refs/tags`; do not push
it to GitHub. Git commit and tag identifiers in the publication lineage differ
where conversion changed a tree. Scientific payload identities recorded by
the evidence files remain payload identities, not claims that the Git commit
IDs are unchanged.

Before publication, verify:

```powershell
git status --short
git lfs fsck
git fsck --connectivity-only
git lfs push --dry-run origin master
git push --dry-run origin master
```

Publish the branch with:

```powershell
git push origin master
```

Do not bypass LFS or replace checkpoints with placeholder files. A fresh clone
must have Git LFS installed to hydrate the exact checkpoint payloads.
