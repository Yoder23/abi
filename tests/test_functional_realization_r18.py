from __future__ import annotations

from collections import Counter

from experiments.functional_realization_r18.isolated_worker import infer_templates
from experiments.functional_realization_r18.package import PACKAGE_FORMAT, realize
from experiments.functional_realization_r18.verify import _derive_package
from experiments.linguistic_realization_r17.frames import SIGNATURES
from experiments.linguistic_realization_r17.frames_v2 import public_rows_v2


def _record(row: dict[str, object]) -> dict[str, object]:
    return {
        "record_id": row["record_id"],
        "signature": row["signature"],
        "slots": row["slots"],
        "output": row["expected"],
    }


def test_factorized_support_repairs_number_invariant_minority() -> None:
    records = [_record(row) for row in public_rows_v2("extraction")]
    target = "question|past|positive|singular"
    target_rows = [row for row in records if row["signature"] == target]
    for row in target_rows[:2]:
        slots = row["slots"]
        row["output"] = f"{slots['subject']} {slots['verb_past']} {slots['object']}?"
    templates, diagnostics = infer_templates(records, list(SIGNATURES))
    independent = Counter(
        "{subject} {verb_past} {object}?" if index < 2 else "Did {subject} {verb_base} {object}?"
        for index in range(3)
    ).most_common(1)[0][0]
    assert independent != templates[target]
    package = {
        "format": PACKAGE_FORMAT,
        "namespace": "english/core/realization",
        "factorized_axis": "subject_number",
        "templates": templates,
        "support": diagnostics["support"],
    }
    assert all(
        realize(package, row["signature"], row["slots"]) == row["expected"]
        for row in public_rows_v2("evaluation")
    )


def test_r18_protocol_keeps_claim_bounded() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = " ".join(
        (root / "experiments/functional_realization_r18/PUBLIC_PROTOCOL.md")
        .read_text(encoding="utf-8")
        .split()
    )
    assert "would not prove unrestricted fluency" in text
    assert "superiority to LoRA or distillation" in text


def test_independent_verifier_derives_same_factorized_package() -> None:
    records = [_record(row) for row in public_rows_v2("extraction")]
    target = "question|past|positive|singular"
    for row in [item for item in records if item["signature"] == target][:2]:
        slots = row["slots"]
        row["output"] = f"{slots['subject']} {slots['verb_past']} {slots['object']}?"
    worker_templates, worker_diagnostics = infer_templates(records, list(SIGNATURES))
    independent, rejected = _derive_package(records)
    assert rejected == []
    assert independent["templates"] == worker_templates
    assert independent["support"] == worker_diagnostics["support"]
