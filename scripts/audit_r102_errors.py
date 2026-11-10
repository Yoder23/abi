"""Post-run descriptive error inventory; no new inference or changed R102 gates."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.coverage_exposure_r102 import core as c  # noqa: E402

OUT = ROOT / "results/coverage_exposure_r102_error_audit_v1"
DECISION_SHA = "177117d4cb59e79139366cb09eda856c82dbf257c50fd973d5758effb7de9a4e"


def classify(reference, output, training_ids):
    screen = c.screen(reference, output)
    missing = [value for value in reference["required"]
               if not re.search(r"\b" + re.escape(value) + r"\b", output, re.I)]
    supplied_ids = set(c.entities(reference))
    missing_ids = [value for value in missing if value in supplied_ids]
    unknown = screen["unknown_identifiers"]
    other_flags = (not screen["surface_form"] or bool(screen["extra_numbers"])
                   or screen["explicit_approval_contradiction"] or screen["collapse"]
                   or not screen["nonempty_bounded"])
    return {
        "mechanical_pass": screen["mechanical_pass"], "missing_required": missing,
        "missing_supplied_identifiers": missing_ids,
        "missing_nonidentifier_literals": [value for value in missing if value not in supplied_ids],
        "unknown_identifiers": unknown,
        "known_training_identifier_intrusions": None if training_ids is None else sorted(set(unknown) & training_ids),
        "identifier_only_screen_failure": not screen["mechanical_pass"] and not other_flags
                                          and len(missing) == len(missing_ids) and bool(missing_ids or unknown),
        "surface_form_failure": not screen["surface_form"],
        "extra_numbers": screen["extra_numbers"], "approval_contradiction": screen["explicit_approval_contradiction"],
        "collapse": screen["collapse"], "length_or_empty_failure": not screen["nonempty_bounded"],
    }


def main():
    from transformers import AutoTokenizer

    c.binding()
    if c.sha(c.OUT / "decision.json") != DECISION_SHA:
        raise RuntimeError("R102 decision identity changed")
    if OUT.exists():
        raise RuntimeError("Preserve previous descriptive audit")
    tokenizer = AutoTokenizer.from_pretrained(c.PARENT, local_files_only=True)
    population = c.read(c.OUT / "development.json")
    training = c.read(c.OUT / "train.json")
    aliases = c.read(c.OUT / "aliases.json")
    ids_by_policy = {"repeat": {value for row in training for value in c.entities(row)},
                    "augment": {value for row in aliases for value in c.entities(row["row"])}}
    prompt_sets = {"repeat": {row["prompt"] for row in training},
                   "augment": {row["row"]["prompt"] for row in aliases}}
    original_fit = c.read(ROOT / "results/grounded_english_r99_fit_audit/training_selection.json")
    all_rows, summaries, inputs = [], {}, {}
    for name in (*c.endpoint_names(), "parent", "retained_full"):
        path = c.OUT / "evaluation" / name / "development.jsonl"
        receipt = c.read(path.parent / "receipt.json")
        if c.sha(path) != receipt["files"]["development.jsonl"]:
            raise RuntimeError("Raw generation evidence changed")
        inputs[path.relative_to(ROOT).as_posix()] = c.sha(path)
        observed = c.rows(path)
        c.validate_generation(observed, population, tokenizer)
        policy = "augment" if "_augment_" in name else "repeat"
        training_ids = None if name == "parent" else ids_by_policy[policy]
        rows = [{"endpoint": name, "id": ref["id"], "stratum": ref["stratum"],
                 **classify(ref, row["output"], training_ids)} for ref, row in zip(population, observed)]
        counts = Counter()
        for row in rows:
            for key in ("mechanical_pass", "missing_required", "missing_supplied_identifiers",
                        "missing_nonidentifier_literals", "unknown_identifiers", "identifier_only_screen_failure",
                        "surface_form_failure", "extra_numbers", "approval_contradiction", "collapse", "length_or_empty_failure"):
                counts[key] += bool(row[key])
        summaries[name] = {"rows": len(rows), "overlapping_reason_counts": dict(counts),
                           "rows_with_known_training_identifier_intrusion": None if training_ids is None else
                           sum(bool(row["known_training_identifier_intrusions"]) for row in rows),
                           "original_fit_rows_actually_seen_in_current_training": None if name == "parent" else
                           sum(ref["prompt"] in prompt_sets[policy] for ref in original_fit)}
        all_rows += rows
    OUT.mkdir()
    with (OUT / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in all_rows:
            c.append(stream, row)
    result = {"format": "r102-post-run-error-inventory/1", "scope": "Descriptive post-run analysis; not preregistered promotion evidence",
              "source_decision_sha256": DECISION_SHA, "input_rows": inputs, "summaries": summaries,
              "rows_sha256": c.sha(OUT / "rows.jsonl"), "new_model_calls": 0, "new_training_steps": 0,
              "limitations": ["Overlapping error counts, not an alternative quality score",
                              "Known-training-ID analysis uses the original locked identifier parser",
                              "Original-example fit is NOT in-sample for the augmented condition",
                              "Parent suitability and native host capacity are not resolved by this inventory"]}
    c.write_once(OUT / "summary.json", result)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
