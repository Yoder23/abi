import copy
import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase2_common import (
    CAPABILITIES,
    Phase2Error,
    canonical_json_bytes,
    sha256_file,
)
from abi.capability_compiler_phase2_human_ratings import (
    lock_completed_forms,
    score_locked_forms,
    verify_scored_manifest,
)


CANDIDATES = ("L0", "L1", "D0", "D1", "D2")


def test_human_scoring_preregistration_bindings_and_zero_ratings():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_SCORING_PROTOCOL_V1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"] == "PREREGISTERED_BEFORE_ANY_HUMAN_RATING_WAS_COMPLETED"
    for relative, expected in protocol["implementation_bindings"].items():
        assert sha256_file(root / relative) == expected
    packet = root / protocol["sealed_packet"]["path"]
    assert sha256_file(packet) == protocol["sealed_packet"]["sha256"]
    packet_dir = packet.parent
    filled = 0
    for index in range(1, 4):
        for row in [
            json.loads(line)
            for line in (packet_dir / f"rater_form_{index}.jsonl").read_bytes().splitlines()
            if line
        ]:
            filled += row["preference"] is not None
    assert filled == protocol["sealed_packet"]["completed_preferences_at_preregistration"] == 0


def test_production_blank_packet_cannot_be_locked_as_completed(tmp_path):
    root = Path(__file__).resolve().parents[1]
    packet = root / "results/abi_capability_compiler_phase2/human_rating_packet_v1"
    completed = tmp_path / "completed"
    completed.mkdir()
    for index in range(1, 4):
        source = packet / f"rater_form_{index}.jsonl"
        target = completed / f"rater_form_{index}.completed.jsonl"
        target.write_bytes(source.read_bytes())
    with pytest.raises(Phase2Error, match="preference"):
        lock_completed_forms(
            packet_dir=packet,
            completed_dir=completed,
            output=completed / "blind_lock.json",
        )


def _write_jsonl(path: Path, rows):
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _fixture(tmp_path: Path):
    packet = tmp_path / "results/abi_capability_compiler_phase2/human_rating_packet_v1"
    completed = tmp_path / "results/abi_capability_compiler_phase2/human_ratings_v1"
    packet.mkdir(parents=True)
    completed.mkdir(parents=True)
    keys = []
    bindings = {}
    total = len(CAPABILITIES) * len(CANDIDATES) * 3
    for form_index in range(1, 4):
        templates = []
        ratings = []
        for candidate in CANDIDATES:
            for capability in CAPABILITIES:
                pair_id = f"{candidate}-vs-T0-{capability}"
                rating_id = f"rater_form_{form_index}-{pair_id}"
                candidate_is_a = (form_index + len(capability)) % 2 == 0
                row = {
                    "rating_id": rating_id,
                    "pair_id": pair_id,
                    "rater_form": f"rater_form_{form_index}",
                    "capability": capability,
                    "prompt": f"prompt {capability}",
                    "output_A": "candidate" if candidate_is_a else "reference",
                    "output_B": "reference" if candidate_is_a else "candidate",
                    "preference": None,
                    "fluency_A_1_to_5": None,
                    "fluency_B_1_to_5": None,
                    "grounding_and_adherence_A_1_to_5": None,
                    "grounding_and_adherence_B_1_to_5": None,
                    "repetition_or_collapse_A": None,
                    "repetition_or_collapse_B": None,
                    "rater_comment": None,
                }
                templates.append(row)
                rated = copy.deepcopy(row)
                rated.update(
                    {
                        "preference": "A" if candidate_is_a else "B",
                        "fluency_A_1_to_5": 5,
                        "fluency_B_1_to_5": 4,
                        "grounding_and_adherence_A_1_to_5": 5,
                        "grounding_and_adherence_B_1_to_5": 4,
                        "repetition_or_collapse_A": False,
                        "repetition_or_collapse_B": False,
                    }
                )
                ratings.append(rated)
                keys.append(
                    {
                        "rating_id": rating_id,
                        "pair_id": pair_id,
                        "rater_form": f"rater_form_{form_index}",
                        "probe_id": capability,
                        "candidate_system": candidate,
                        "reference_system": "T0",
                        "system_A": candidate if candidate_is_a else "T0",
                        "system_B": "T0" if candidate_is_a else candidate,
                    }
                )
        template_path = packet / f"rater_form_{form_index}.jsonl"
        _write_jsonl(template_path, templates)
        _write_jsonl(completed / f"rater_form_{form_index}.completed.jsonl", ratings)
        bindings[template_path.name] = {
            "rows": len(templates),
            "sha256": sha256_file(template_path),
        }
    key_path = packet / "blinding_key.jsonl"
    _write_jsonl(key_path, keys)
    bindings[key_path.name] = {"rows": total, "sha256": sha256_file(key_path)}
    manifest = {
        "format": "abi-capability-compiler-phase2-human-rating-packet/1",
        "status": "AWAITING_THREE_INDEPENDENT_HUMAN_RATERS",
        "candidate_systems": list(CANDIDATES),
        "rater_forms": 3,
        "pairs_per_form": len(CAPABILITIES) * len(CANDIDATES),
        "ratings_required": total,
        "file_bindings": bindings,
    }
    (packet / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    attestations = {
        "format": "abi-capability-compiler-phase2-rater-attestations/1",
        "raters": [
            {
                "rater_id": f"person-{index}",
                "rater_form": f"rater_form_{index}",
                "human_rater": True,
                "completed_independently": True,
                "answer_key_not_accessed_before_lock": True,
                "other_ratings_not_accessed_before_lock": True,
                "completed_utc": f"2026-08-1{index}T00:00:00Z",
                "conflict_disclosure": "none",
            }
            for index in range(1, 4)
        ],
        "custodian": {
            "custodian_id": "custodian-1",
            "verified_three_distinct_people": True,
            "withheld_answer_key_and_cross_rater_access_until_lock": True,
        },
    }
    (completed / "attestations.json").write_bytes(canonical_json_bytes(attestations))
    return packet, completed


def test_lock_then_unblind_and_score(tmp_path):
    packet, completed = _fixture(tmp_path)
    lock_path = completed / "blind_lock.json"
    lock = lock_completed_forms(packet_dir=packet, completed_dir=completed, output=lock_path)
    assert lock["blinding_key_read_by_lock_tool"] is False
    result = score_locked_forms(
        packet_dir=packet,
        completed_dir=completed,
        lock_path=lock_path,
        output=completed / "manifest.json",
    )
    assert result["status"] == "PASS"
    assert result["ratings"] == len(CAPABILITIES) * len(CANDIDATES) * 3
    assert all(value["candidate_preference_credit"]["lower_95"] == 1.0 for value in result["systems"].values())


def test_incomplete_rating_cannot_lock(tmp_path):
    packet, completed = _fixture(tmp_path)
    path = completed / "rater_form_1.completed.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    rows[0]["preference"] = None
    _write_jsonl(path, rows)
    with pytest.raises(Phase2Error, match="preference"):
        lock_completed_forms(packet_dir=packet, completed_dir=completed, output=completed / "lock.json")


def test_modified_blinded_content_cannot_lock(tmp_path):
    packet, completed = _fixture(tmp_path)
    path = completed / "rater_form_1.completed.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    rows[0]["output_A"] += " tampered"
    _write_jsonl(path, rows)
    with pytest.raises(Phase2Error, match="blinded prompt/output"):
        lock_completed_forms(packet_dir=packet, completed_dir=completed, output=completed / "lock.json")


def test_duplicate_rater_identity_cannot_lock(tmp_path):
    packet, completed = _fixture(tmp_path)
    path = completed / "attestations.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["raters"][1]["rater_id"] = value["raters"][0]["rater_id"]
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(Phase2Error, match="distinct"):
        lock_completed_forms(packet_dir=packet, completed_dir=completed, output=completed / "lock.json")


def test_missing_custodian_attestation_cannot_lock(tmp_path):
    packet, completed = _fixture(tmp_path)
    path = completed / "attestations.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["custodian"]["verified_three_distinct_people"] = False
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(Phase2Error, match="three distinct people"):
        lock_completed_forms(packet_dir=packet, completed_dir=completed, output=completed / "lock.json")


def test_form_change_after_lock_prevents_unblinding(tmp_path):
    packet, completed = _fixture(tmp_path)
    lock_path = completed / "blind_lock.json"
    lock_completed_forms(packet_dir=packet, completed_dir=completed, output=lock_path)
    path = completed / "rater_form_2.completed.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    rows[0]["preference"] = "TIE"
    _write_jsonl(path, rows)
    with pytest.raises(Phase2Error, match="changed after blinded lock"):
        score_locked_forms(
            packet_dir=packet,
            completed_dir=completed,
            lock_path=lock_path,
            output=completed / "manifest.json",
        )


def test_scoring_cannot_run_without_lock(tmp_path):
    packet, completed = _fixture(tmp_path)
    with pytest.raises((Phase2Error, FileNotFoundError)):
        score_locked_forms(
            packet_dir=packet,
            completed_dir=completed,
            lock_path=completed / "missing-lock.json",
            output=completed / "manifest.json",
        )


def test_nonreproducible_score_manifest_is_rejected(tmp_path):
    packet, completed = _fixture(tmp_path)
    lock_path = completed / "blind_lock.json"
    manifest_path = completed / "manifest.json"
    lock_completed_forms(packet_dir=packet, completed_dir=completed, output=lock_path)
    score_locked_forms(
        packet_dir=packet,
        completed_dir=completed,
        lock_path=lock_path,
        output=manifest_path,
    )
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["systems"]["L0"]["candidate_preference_credit"]["point"] = 0.123
    manifest_path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(Phase2Error, match="not reproducible"):
        verify_scored_manifest(root=tmp_path, path=manifest_path)
