import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase2_common import Phase2Error, canonical_json_bytes, sha256_file
from abi.capability_compiler_phase2_rater_session import (
    export_completed_form,
    initialize_session,
    load_progress,
    rate_interactively,
    record_rating,
)


def test_rater_session_protocol_bindings_and_zero_rating_boundary():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "ABI_CAPABILITY_COMPILER_PHASE2_RATER_SESSION_PROTOCOL_V1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"] == "PREREGISTERED_BEFORE_ANY_HUMAN_RATING_WAS_COMPLETED"
    for relative, expected in protocol["implementation_bindings"].items():
        assert sha256_file(root / relative) == expected
    parent = root / protocol["parent_handoff"]["protocol"]
    assert sha256_file(parent) == protocol["parent_handoff"]["sha256"]
    packet = root / "results/abi_capability_compiler_phase2/human_rating_packet_v1"
    filled = 0
    for index in range(1, 4):
        for row in [
            json.loads(line)
            for line in (packet / f"rater_form_{index}.jsonl").read_bytes().splitlines()
            if line
        ]:
            filled += row["preference"] is not None
    assert filled == protocol["parent_handoff"]["completed_preferences_at_preregistration"] == 0


def test_production_forms_initialize_as_three_key_free_sessions(tmp_path):
    root = Path(__file__).resolve().parents[1]
    packet = root / "results/abi_capability_compiler_phase2/human_rating_packet_v1"
    hashes = []
    for index in range(1, 4):
        session_dir = tmp_path / f"session-{index}"
        session = initialize_session(
            packet_dir=packet,
            form_index=index,
            session_dir=session_dir,
            rater_id=f"production-dry-run-{index}",
        )
        assert session["ratings_required"] == 7000
        assert session["blinding_key_read_by_session_tool"] is False
        assert not (session_dir / "blinding_key.jsonl").exists()
        hashes.append(session["blinded_form"]["sha256"])
    assert len(set(hashes)) == 3


def _write_jsonl(path: Path, rows):
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _row(form_index: int, index: int):
    return {
        "rating_id": f"rater_form_{form_index}-pair-{index}",
        "pair_id": f"pair-{index}",
        "rater_form": f"rater_form_{form_index}",
        "capability": "grammar",
        "prompt": f"prompt {index}",
        "output_A": f"output A {index}",
        "output_B": f"output B {index}",
        "preference": None,
        "fluency_A_1_to_5": None,
        "fluency_B_1_to_5": None,
        "grounding_and_adherence_A_1_to_5": None,
        "grounding_and_adherence_B_1_to_5": None,
        "repetition_or_collapse_A": None,
        "repetition_or_collapse_B": None,
        "rater_comment": None,
    }


def _rating(preference="A"):
    return {
        "preference": preference,
        "fluency_A_1_to_5": 5,
        "fluency_B_1_to_5": 4,
        "grounding_and_adherence_A_1_to_5": 5,
        "grounding_and_adherence_B_1_to_5": 4,
        "repetition_or_collapse_A": False,
        "repetition_or_collapse_B": False,
        "rater_comment": None,
    }


def _packet(tmp_path: Path):
    packet = tmp_path / "packet"
    packet.mkdir()
    bindings = {}
    for form_index in range(1, 4):
        path = packet / f"rater_form_{form_index}.jsonl"
        _write_jsonl(path, [_row(form_index, 1), _row(form_index, 2)])
        bindings[path.name] = {"rows": 2, "sha256": sha256_file(path)}
    key = packet / "blinding_key.jsonl"
    key.write_text("DO NOT READ", encoding="utf-8")
    bindings[key.name] = {"rows": 6, "sha256": sha256_file(key)}
    manifest = {
        "format": "abi-capability-compiler-phase2-human-rating-packet/1",
        "status": "AWAITING_THREE_INDEPENDENT_HUMAN_RATERS",
        "rater_forms": 3,
        "file_bindings": bindings,
    }
    (packet / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return packet


def _session(tmp_path: Path):
    packet = _packet(tmp_path)
    session = tmp_path / "session"
    initialize_session(packet_dir=packet, form_index=1, session_dir=session, rater_id="person-1")
    return packet, session


def test_session_is_bound_to_one_blinded_form_and_never_copies_key(tmp_path):
    packet, session = _session(tmp_path)
    value = json.loads((session / "session.json").read_text(encoding="utf-8"))
    assert value["rater_form"] == "rater_form_1"
    assert value["blinding_key_read_by_session_tool"] is False
    assert (session / "blinded_form.jsonl").read_bytes() == (packet / "rater_form_1.jsonl").read_bytes()
    assert not (session / "blinding_key.jsonl").exists()


def test_resume_and_complete_export(tmp_path):
    _packet_dir, session = _session(tmp_path)
    rows = load_progress(session)["rows"]
    record_rating(session_dir=session, rating_id=rows[0]["rating_id"], rating=_rating())
    assert load_progress(session)["remaining"] == 1
    with pytest.raises(Phase2Error, match="incomplete"):
        export_completed_form(session_dir=session, output=tmp_path / "incomplete.jsonl")
    record_rating(session_dir=session, rating_id=rows[1]["rating_id"], rating=_rating("TIE"))
    output = tmp_path / "rater_form_1.completed.jsonl"
    receipt = export_completed_form(session_dir=session, output=output)
    completed = [json.loads(line) for line in output.read_bytes().splitlines() if line]
    assert [row["preference"] for row in completed] == ["A", "TIE"]
    assert receipt["ratings"] == 2
    assert receipt["blinding_key_read_by_session_tool"] is False


def test_invalid_rating_and_cross_form_identity_are_rejected(tmp_path):
    _packet_dir, session = _session(tmp_path)
    with pytest.raises(Phase2Error, match="not in this blinded form"):
        record_rating(session_dir=session, rating_id="rater_form_2-pair-1", rating=_rating())
    invalid = _rating()
    invalid["fluency_A_1_to_5"] = 6
    with pytest.raises(Phase2Error, match="1-to-5"):
        record_rating(session_dir=session, rating_id="rater_form_1-pair-1", rating=invalid)


def test_blinded_form_tampering_is_rejected_on_resume(tmp_path):
    _packet_dir, session = _session(tmp_path)
    path = session / "blinded_form.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    rows[0]["output_A"] = "tampered"
    _write_jsonl(path, rows)
    with pytest.raises(Phase2Error, match="form changed"):
        load_progress(session)


def test_event_chain_tampering_is_rejected(tmp_path):
    _packet_dir, session = _session(tmp_path)
    record_rating(session_dir=session, rating_id="rater_form_1-pair-1", rating=_rating())
    path = session / "rating_events.jsonl"
    event = json.loads(path.read_text(encoding="utf-8"))
    event["rating"]["preference"] = "B"
    path.write_bytes(canonical_json_bytes(event))
    with pytest.raises(Phase2Error, match="event chain"):
        load_progress(session)


def test_duplicate_requires_explicit_hash_chained_revision(tmp_path):
    _packet_dir, session = _session(tmp_path)
    rating_id = "rater_form_1-pair-1"
    first = record_rating(session_dir=session, rating_id=rating_id, rating=_rating())
    with pytest.raises(Phase2Error, match="explicit revision"):
        record_rating(session_dir=session, rating_id=rating_id, rating=_rating("B"))
    second = record_rating(
        session_dir=session, rating_id=rating_id, rating=_rating("B"), allow_revision=True
    )
    assert second["revision_of_sequence"] == first["sequence"]
    assert load_progress(session)["latest"][rating_id]["rating"]["preference"] == "B"


def test_interactive_session_can_stop_and_resume(tmp_path):
    _packet_dir, session = _session(tmp_path)
    answers = iter(["A", "5", "4", "5", "4", "n", "n", "", "Q"])
    result = rate_interactively(
        session_dir=session,
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _text: None,
    )
    assert result == {"recorded_this_run": 1, "completed": 1, "remaining": 1}
    assert load_progress(session)["remaining"] == 1


def test_initializer_rejects_already_rated_or_identity_exposing_form(tmp_path):
    packet = _packet(tmp_path)
    path = packet / "rater_form_1.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    rows[0]["system_A"] = "T0"
    _write_jsonl(path, rows)
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_bindings"][path.name]["sha256"] = sha256_file(path)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(Phase2Error, match="not blind"):
        initialize_session(
            packet_dir=packet,
            form_index=1,
            session_dir=tmp_path / "session",
            rater_id="person-1",
        )
