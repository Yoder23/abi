# Resumable blinded rater session

This optional tool implements the same frozen V554 handoff without requiring a
human to edit JSONL. It does not change the 7,000 judgments per form, the
three-human requirement, the rubric, or the scoring rule.

The custodian—not the rater—initializes each session on the machine that holds
the sealed packet. For form 1:

```powershell
python -m abi.capability_compiler_phase2_rater_session init `
  --packet-dir results/abi_capability_compiler_phase2/human_rating_packet_v1 `
  --form-index 1 `
  --session-dir C:\path\for\rater-1-session `
  --rater-id REPLACE_WITH_DISTINCT_RATER_ID_1
```

Repeat for forms 2 and 3 with different session directories and distinct
rater identifiers. Give each person only their session directory. Never give a
rater the packet directory, `blinding_key.jsonl`, another rater's session, or
another rater's completed output.

Each rater starts or resumes with:

```powershell
python -m abi.capability_compiler_phase2_rater_session rate `
  --session-dir C:\path\for\rater-1-session
```

Entering `Q` at a preference prompt stops safely. Progress is append-only and
hash-chained. Check status at any time:

```powershell
python -m abi.capability_compiler_phase2_rater_session status `
  --session-dir C:\path\for\rater-1-session
```

After all 7,000 rows are complete, export the exact completed form:

```powershell
python -m abi.capability_compiler_phase2_rater_session export `
  --session-dir C:\path\for\rater-1-session `
  --output C:\path\for\rater-1-session\rater_form_1.completed.jsonl
```

The tool refuses incomplete export and never overwrites an existing completed
form or receipt. The custodian collects the three exported forms and their
completion receipts, places the forms under
`results/abi_capability_compiler_phase2/human_ratings_v1`, completes the V554
attestation file, and resumes the lock/unblind/score procedure in
`PHASE2_HUMAN_RATING_HANDOFF_V1.md`.

The session contains no answer key or system identities. Software still cannot
prove that a person is human, prove independence, or exclude off-system
collusion; the named custodian remains responsible for those facts.
