# R104 research review: control gate failed, recipient matrix not run

Research tag: `abi-research-r104-2026-11-10`. Package API remains `0.4.0a1`.
This is a bounded negative research prerelease, not a production English core.

## Result

The [full packet](../R104_EVALUATOR_CONTROL.md) preserves the prospective protocol,
729 input bindings, 155 evaluator cases and 48 raw Qwen generations. The control
scored 29/48 against 44/48 and failed four task floors. The legacy screen scored
44/48 but also missed the clarification floor. No recipients or training ran.

Two additional approval-scope limitations are disclosed without changing the
frozen scorer. Removing every approval flag yields at most 30/48, still failing.
Do not claim general semantic evaluator validation, a qualified English control,
ABI-transfer failure, native LayerCake failure or moonshot completion.

## Review contents

- `results/review_2026_11_10/r104_manifest.json`: complete required asset inventory,
  four inherited LFS checkpoint paths, source tokenizer identity and host binding.
- `scripts/review_r104_public.py`: strict public-checkout recomputation helper.
- `results/review_2026_11_10/r104_preflight_operations.md`: preserved initial
  direct-harness failure and successful supported exact-fixture rerun.
- `results/evaluator_control_r104/`: raw outputs, input freeze, evaluator audit,
  registered decision, resource receipt and post-run descriptive inventory.

The helper requires pinned sibling checkouts and the source tokenizer/config
files. It checks custody, recomputes the decision and error inventory, and runs
43 focused checks. It does not run a model or claim independent hardware.
It must write its receipt outside both checkouts and never overwrite a receipt.

## Published reconstruction

Source commit: `f9bff9ae52499145f083b526b32aa2ac59eeb4ea`.
Manifest SHA-256: `4be7dfc63be3204876da8aae6ceed8a9f90860f085da467e26f174a8149b7af4`.
The additive `public_r104_reconstruction.json` receipt matches 566 assets,
four inherited LFS payload paths and 729 frozen bindings; the same decision and
error inventory recompute and 43 focused checks pass. Receipt SHA-256:
`017b84072a2d0e04c79426c12e72d9f002099d06785b660ab2de656d62e9d696`.

Public-only checkouts and public tokenizer cache were reused on the same machine.
This is not a fresh clone/environment, live model rerun, independent hardware or
human review. Sealed-tag GitHub CI passes Python3.10-3.12 plus lint. The receipt
and updated notes are additive; the scientific tag is unchanged.

The [weight-space research assessment](../WEIGHT_SPACE_RESEARCH_2026_11.md) treats
direct editing as a future competing acquisition mechanism. No backend or
LayerCake package-contract change is bundled here. R7 and all older tags remain
unchanged; `FULL_MOONSHOT_NOT_PROVEN` remains authoritative.
