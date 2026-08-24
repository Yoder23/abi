import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase2_common import Phase2Error, canonical_json_bytes, sha256_file
from abi.human_rate import human_rate


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


def _packet(tmp_path: Path) -> Path:
    packet = tmp_path / "packet"
    packet.mkdir()
    bindings = {}
    for index in range(1, 4):
        path = packet / f"rater_form_{index}.jsonl"
        path.write_bytes(canonical_json_bytes(_row(index, 1)))
        bindings[path.name] = {"rows": 1, "sha256": sha256_file(path)}
    key = packet / "blinding_key.jsonl"
    key.write_text("never read", encoding="utf-8")
    bindings[key.name] = {"rows": 3, "sha256": sha256_file(key)}
    manifest = {
        "format": "abi-capability-compiler-phase2-human-rating-packet/1",
        "status": "AWAITING_THREE_INDEPENDENT_HUMAN_RATERS",
        "rater_forms": 3,
        "file_bindings": bindings,
    }
    (packet / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return packet


def _answers():
    return iter(["A", "5", "4", "5", "4", "n", "n", ""])


def test_one_command_completes_locks_and_signs_without_reading_key(tmp_path):
    packet = _packet(tmp_path)
    answers = _answers()
    result = human_rate(
        rater="R1",
        packet_dir=packet,
        work_dir=tmp_path / "work",
        rater_identity="Person One",
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _text: None,
    )
    assert result["status"] == "COMPLETE_LOCKED_AND_SIGNED"
    assert result["ratings"] == 1
    assert "PRIVATE KEY" not in json.dumps(result)
    assert not (tmp_path / "work/sessions/R1/blinding_key.jsonl").exists()


def test_duplicate_identity_cannot_take_two_forms(tmp_path):
    packet = _packet(tmp_path)
    answers = _answers()
    human_rate(
        rater="R1",
        packet_dir=packet,
        work_dir=tmp_path / "work",
        rater_identity="Same Person",
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _text: None,
    )
    with pytest.raises(Phase2Error, match="duplicate rater identity"):
        human_rate(
            rater="R2",
            packet_dir=packet,
            work_dir=tmp_path / "work",
            rater_identity="Same Person",
            input_fn=lambda _prompt: "Q",
            output_fn=lambda _text: None,
        )


def test_command_resumes_existing_session(tmp_path):
    packet = _packet(tmp_path)
    result = human_rate(
        rater="R3",
        packet_dir=packet,
        work_dir=tmp_path / "work",
        rater_identity="Person Three",
        input_fn=lambda _prompt: "Q",
        output_fn=lambda _text: None,
    )
    assert result["status"] == "IN_PROGRESS"
    assert result["remaining"] == 1
