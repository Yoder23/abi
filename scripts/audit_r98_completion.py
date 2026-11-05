"""Read-only C1 audit repair: resolve the pinned tokenizer by local snapshot path.

The frozen C1 verifier passed a Hub identifier to a recent tokenizer version;
its metadata probe attempted a forbidden offline network lookup. This additive
audit does not change C1's code, rows, gates, or failed orchestration receipt.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.plan_english_r98_completion import core as c  # noqa: E402


def recompute(catalog, old, completed, merged, result):
    if c.merge_completions(catalog, old, completed) != merged:
        raise RuntimeError("Merged evidence not reconstructible")
    summary = c.aggregate(merged)
    gate = c.source_gate(merged)
    if result["summary"] != summary or result["source_gate"] != gate:
        raise RuntimeError("Unrecomputable summary or trusted gate")
    original = c.read(c.original.OUT / "source/result.json")["accounting"]
    accounting = result["accounting"]
    if accounting["original_acquisition"] != original:
        raise RuntimeError("Original acquisition accounting changed")
    completion_costs = accounting["completion_acquisition"]
    checks = {
        "total_teacher_generated_tokens": original["teacher_generated_tokens"]
        + sum(len(r["token_ids"]) for r in completed),
        "total_teacher_input_tokens": original["teacher_input_tokens"]
        + sum(r["input_tokens"] for r in completed),
        "raw_unique_prompts": len({r["prompt"] for r in merged}),
        "total_source_requests": len(old) + len(completed),
        "unique_prompt_utf8_bytes": sum(len(s.encode()) for s in {r["prompt"] for r in merged}),
        "imported_completed_output_tokens": sum(len(r["token_ids"]) for r in merged),
        "imported_completed_output_utf8_bytes": sum(
            len(r["teacher_output"].encode()) for r in merged
        ),
    }
    if any(accounting[key] != value for key, value in checks.items()):
        raise RuntimeError("Unrecomputable imported-information accounting")
    if completion_costs["teacher_generated_tokens"] != sum(len(r["token_ids"]) for r in completed):
        raise RuntimeError("Completion token accounting mismatch")
    if completion_costs["teacher_input_tokens"] != sum(r["input_tokens"] for r in completed):
        raise RuntimeError("Completion input accounting mismatch")
    if completion_costs["source_inference_seconds"] != sum(r["wall_seconds"] for r in completed):
        raise RuntimeError("Completion timing mismatch")
    return summary, gate, checks


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    c.verify_binding()
    catalog, old = c.original_evidence()
    destination = c.OUT / "source"
    result = c.read(destination / "result.json")
    for file, key in (("rows.jsonl", "rows_sha256"), ("completions.jsonl", "completions_sha256")):
        if c.sha(destination / file) != result[key]:
            raise RuntimeError("Source evidence hash mismatch")
    completed = c.rows(destination / "completions.jsonl")
    merged = c.rows(destination / "rows.jsonl")
    summary, gate, accounting = recompute(catalog, old, completed, merged, result)
    snapshot = Path(
        snapshot_download(c.SOURCE, revision=c.REVISION, local_files_only=True)
    ).resolve()
    if snapshot.name != c.REVISION:
        raise RuntimeError("Source revision mismatch")
    inventory = result["accounting"]["original_acquisition"]["source_files"]
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "config.json",
    ):
        if c.sha(snapshot / name) != inventory[name]["sha256"]:
            raise RuntimeError(f"Pinned tokenizer file changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    for row in merged:
        if (
            tokenizer.decode(row["token_ids"], skip_special_tokens=True).strip()
            != row["teacher_output"]
        ):
            raise RuntimeError("Token/text evidence mismatch")
    value = {
        "format": "abi-r98-completion-audit/1",
        "freeze_sha256": c.sha(c.OUT / "freeze.json"),
        "source_rows_sha256": c.sha(destination / "rows.jsonl"),
        "completions_sha256": c.sha(destination / "completions.jsonl"),
        "source_result_sha256": c.sha(destination / "result.json"),
        "preserved_stop_sha256": c.sha(c.OUT / "stop.json"),
        "audit_code_sha256": c.sha(Path(__file__)),
        "summary": summary,
        "source_gate_recomputed": gate,
        "verdict": "SOURCE_PREREQUISITE_PASSED" if gate else "ABI_SOURCE_PREREQUISITE_FAILED",
        "complete_token_prefix_matches": len(completed),
        "truncated_rows": sum(r["truncated"] for r in merged),
        "email_failures_over_length": sum(
            r["task"] == "email_drafting" and not r["screen"]["nonempty_bounded"] for r in merged
        ),
        "email_confirmation_flags": sum(
            r["task"] == "email_drafting" and r["screen"]["explicit_approval_contradiction"]
            for r in merged
        ),
        "accounting_recomputed": accounting,
        "training_artifacts_present": [arm for arm in c.ARMS if (c.OUT / arm).exists()],
        "human_fluency": "UNASSESSED",
        "full_moonshot": "NOT_PROVEN",
        "attribution": "ABI acquisition/normalization boundary; LayerCake quality untested",
    }
    output = ROOT / "results/plan_english_r98_completion_review/audit.json"
    if output.exists():
        if c.read(output) != value:
            raise RuntimeError("Existing C1 audit is stale")
    else:
        c.write_once(output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
