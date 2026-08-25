# Human evaluation

The frozen packet has three blinded, counterbalanced 7,000-row forms (21,000
judgments total). Raters use unique identities and isolated append-only sessions:

```text
abi human-rate --rater R1
abi human-rate --rater R2
abi human-rate --rater R3
```

Codex completed zero ratings. Unblinding is forbidden until all three signed
forms are locked. See `human_packet_validation.json` and the Phase 2 handoff.
