# R51 launch v1 operational failure

Verdict: `FAIL_CLOSED_ZERO_STEPS_CONTEXT_INCOMPATIBILITY`

The frozen R51 command verified and loaded its input artifacts, then stopped
before training because 2 of 24,421 source prompts exceed the LayerCake host's
locked 256-token context and prompt exclusion had not been authorized.

- successful optimizer steps: 0;
- checkpoint emitted: no;
- output directory emitted: no;
- source or parent artifacts changed: no; and
- scientific candidate verdict: unavailable.

The overlength rows may not be truncated silently and the LayerCake context
contract may not be changed.  The additive v1a repair permits only the
existing fail-closed `--exclude-overlength-prompts` path, which must enumerate
and hash every excluded prompt in the candidate metadata.

