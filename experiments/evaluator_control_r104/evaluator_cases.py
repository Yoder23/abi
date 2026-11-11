"""Deterministic author-specified unit/challenge cases; NOT human quality labels.

No case or model answer is fed to a source or recipient. Expectations encode
literal, numeric, polarity, length and formatting obligations of the task only.
"""

from __future__ import annotations

import copy

from . import core as c


def cases():
    result = []
    for index, task in enumerate(c.TASKS):
        code, owner = f"NOTEF104{index}", f"PersonF104{index}A"
        row = {"id": f"fixture-{task}", "task": task, "code": code,
               "prompt": f"{code}: {owner} coordinates the project for Monday. Approval is pending.",
               "required": [code] if task in ("clarification", "abstention") else [code, owner, "Monday"]}
        prose = f"{code} is coordinated by {owner} for Monday. Approval is pending."
        good = {
            "summarization": prose, "rewriting": prose, "conversation": prose,
            "tone_control": prose,
            "email_drafting": f"Dear colleague,\n{prose}\nBest regards,\nCoordinator",
            "format_control": f"- {code} is coordinated by {owner}.\n- Planned day: Monday.\n- Approval is pending.",
            "clarification": f"What is the start time for {code}?",
            "abstention": f"The approved budget for {code} cannot be determined from the supplied notes.",
        }[task]
        variants = {
            "valid": (good, True),
            "missing_identifier": (good.replace(code, "the project"), False),
            "foreign_identifier": (good + " PersonUNKNOWN is involved.", False),
            "invented_number": (good + " Cost: 937 dollars.", False),
            "invented_day": (good + " Tuesday is planned.", False),
            "contradictory_present": (good + " The arrangements are confirmed.", False),
            "guaranteed_future": (good + " It will be approved.", False),
            "collapse": (good + " The project is underway." * 5, False),
            "length": (good + " " + " ".join(f"word{v}" for v in range(121)), False),
            "empty": ("", False),
        }
        for label, (text, expected) in variants.items():
            result.append({"id": f"{task}-{label}", "row": copy.deepcopy(row),
                           "answer": text, "expected_pass": expected, "basis": label})
    # Minimal contrasts exercise clause-local polarity, not a semantic judge.
    row = next(r["row"] for r in result if r["id"] == "rewriting-valid")
    lead = next(r["answer"] for r in result if r["id"] == "rewriting-valid")
    for introducer in ("If", "When", "Once", "Until", "Before", "After", "As soon as"):
        for subject in ("the details", "the arrangements", "they"):
            conditional = f" {introducer} {subject} are confirmed, I can send an update."
            for label, suffix, expected in (
                ("conditional", conditional, True),
                ("comma_assertion", conditional + " The plans are approved.", False),
                ("conjunction_assertion", f" {introducer} {subject} are confirmed and the plans are approved, I can send an update.", False),
            ):
                result.append({"id": f"scope-{introducer}-{subject}-{label}", "row": row,
                               "answer": lead + suffix, "expected_pass": expected, "basis": label})
    for label, replacement, expected in (
        ("negated", "The arrangements are not confirmed.", True),
        ("not_yet", "The arrangements are not yet approved.", True),
        ("past_positive", "The arrangements were approved.", False),
        ("approval_granted", "Approval was granted.", False),
        ("approved_yes", "Approved: yes.", False),
    ):
        result.append({"id": label, "row": row, "answer": lead + " " + replacement,
                       "expected_pass": expected, "basis": label})
    for task, label, transform in (
        ("format_control", "one_line", lambda s: s.replace("\n", " ")),
        ("email_drafting", "no_greeting", lambda s: s.replace("Dear colleague,", "")),
        ("email_drafting", "no_signoff", lambda s: s.replace("Best regards,", "")),
        ("tone_control", "one_sentence", lambda s: s.replace(". Approval", "; approval")),
        ("summarization", "four_sentences", lambda s: s + " Please wait. Stay tuned."),
        ("clarification", "two_questions", lambda s: s + " Why?"),
        ("abstention", "no_abstention", lambda s: s.replace("cannot be determined from the supplied notes", "is available")),
    ):
        ref = next(r for r in result if r["id"] == f"{task}-valid")
        result.append({"id": f"{task}-{label}", "row": ref["row"], "answer": transform(ref["answer"]),
                       "expected_pass": False, "basis": label})
    return result


def audit():
    records = [{**case, "observed": c.screen(case["row"], case["answer"])} for case in cases()]
    mismatches = [r["id"] for r in records if r["observed"]["mechanical_pass"] != r["expected_pass"]]
    return {"format": "r104-evaluator-audit/1", "cases": records, "mismatches": mismatches,
            "label_origin": "author-specified formal obligations; no human fluency ratings",
            "claim": "Finite mechanical regression coverage, not general semantic validation"}


def require_audit():
    expected = audit()
    if expected["mismatches"]:
        raise RuntimeError(f"Evaluator fixture mismatch: {expected['mismatches']}")
    path = c.OUT / "evaluator_audit.json"
    if c.read(path) != expected:
        raise RuntimeError("Missing or stale evaluator audit")
    return expected
