from scripts.audit_r102_errors import classify


def reference():
    return {"id": "audit", "task": "rewriting", "code": "NOTENEW",
            "prompt": "NOTENEW is coordinated by PersonNew on Thursday; approval is pending.",
            "required": ["NOTENEW", "PersonNew", "Thursday"]}


def test_known_training_identifier_intrusion():
    result = classify(reference(), "NOTEOLD is coordinated by PersonNew on Thursday; approval is pending.", {"NOTEOLD"})
    assert result["known_training_identifier_intrusions"] == ["NOTEOLD"]
    assert result["missing_supplied_identifiers"] == ["NOTENEW"]
    assert result["identifier_only_screen_failure"]
    assert not result["mechanical_pass"]


def test_missing_day_is_not_an_identifier_only_failure():
    result = classify(reference(), "NOTENEW is coordinated by PersonNew on Friday; approval is pending.", set())
    assert result["missing_nonidentifier_literals"] == ["Thursday"]
    assert not result["identifier_only_screen_failure"]


def test_unknown_is_not_automatically_memorized():
    result = classify(reference(), "NOTEUNKNOWN is coordinated by PersonNew on Thursday; approval is pending.", {"NOTEOLD"})
    assert result["unknown_identifiers"] == ["NOTEUNKNOWN"]
    assert result["known_training_identifier_intrusions"] == []


def test_parent_history_is_not_assumed():
    result = classify(reference(), "NOTEOLD is coordinated by PersonNew on Thursday; approval is pending.", None)
    assert result["known_training_identifier_intrusions"] is None


def test_clean_response_has_no_failure_flags():
    result = classify(reference(), reference()["prompt"], set())
    assert result["mechanical_pass"]
    assert not result["identifier_only_screen_failure"]
    assert result["missing_required"] == []
