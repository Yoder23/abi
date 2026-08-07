import hashlib
import json
from pathlib import Path

from abi.capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from abi.capability_compiler_phase2_teacher import development_probes
from abi.capability_compiler_phase3_bpe_core import _layercake_api
from abi.capability_compiler_phase3_segment_router import _semantic_segments


ROOT = Path(__file__).resolve().parents[1]
V50 = ROOT / "results/abi_capability_compiler_phase3_route_capacity"
V51 = ROOT / "results/abi_capability_compiler_phase3_route_capacity_fit/attribution_v51"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_embedded_evidence_hash(document):
    expected = document.pop("evidence_sha256")
    assert hashlib.sha256(canonical_json_bytes(document)).hexdigest() == expected


def test_v50_failed_closed_and_raw_evidence_is_bound() -> None:
    result = _json(ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_CAPACITY_RESULT_V50.json")
    candidate = V50 / "development_v50/U0-seed240050"
    evaluation = V50 / "evaluation_v50/U0-seed240050"
    decision = _json(evaluation / "decision.json")

    assert result["status"] == "COMPLETE_FAILED_NONPROMOTIONAL_CAPACITY_UPPER_BOUND"
    assert result["promotion_eligible"] is False
    assert result["phase3_certified"] is False
    assert result["phase4_open"] is False
    assert sha256_file(candidate / "model.safetensors") == result["candidate"]["checkpoint_sha256"]
    assert sha256_file(evaluation / "development_outputs.jsonl") == result["autonomous_evaluation"]["outputs_sha256"]
    assert sha256_file(evaluation / "decision.json") == result["autonomous_evaluation"]["decision_file_sha256"]
    assert decision["functional_passes"] == 785
    assert decision["repetition_collapses"] == 36
    assert decision["route_correct"] == 1400
    assert decision["initial_screen_pass"] is False
    _verify_embedded_evidence_hash(decision)


def test_v50_control_text_reproduces_training_token_path() -> None:
    protocol = _json(ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_CAPACITY_PROTOCOL_V50.json")
    candidate = V50 / "development_v50/U0-seed240050"
    controls = {row["capability"]: row for row in _json(candidate / "route_controls.json")}
    _, _, tokenizer_type, _, _ = _layercake_api(ROOT, protocol)
    tokenizer = tokenizer_type.from_document(_json(candidate / "tokenizer.json"))
    probes = development_probes(ROOT / protocol["development_catalog"])
    outputs = [json.loads(line) for line in (V50 / "evaluation_v50/U0-seed240050/development_outputs.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(probes) == len(outputs) == 1400
    for probe, output in zip(probes, outputs):
        route = output["predicted_route"]
        control = controls[route]
        piece = bytes.fromhex(control["piece_hex"])
        body = _semantic_segments(str(probe["prompt"]))[-1]
        deployed = [tokenizer.lexeme_to_id[item] for item in tokenizer.split(piece.decode("utf-8") + "\n" + body)]
        trained = [control["token_id"]] + [tokenizer.lexeme_to_id[item] for item in tokenizer.split("\n" + body)]
        assert deployed == trained


def test_v51_fit_attribution_is_complete_and_failed_closed() -> None:
    result = _json(ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_ROUTE_CAPACITY_FIT_RESULT_V51.json")
    decision = _json(V51 / "decision.json")
    rows = [json.loads(line) for line in (V51 / "training_fit_rows.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == result["records"] == 7000
    assert sum(row["actions"] for row in rows) == result["actions"] == 56029
    assert sum(row["correct_actions"] for row in rows) == result["correct_actions"] == 54084
    assert sum(row["exact_sequence"] for row in rows) == result["exact_sequences"] == 5540
    assert decision["classification"] == "BRIDGE_OR_BACKBONE_FIT_LIMITED"
    assert result["fit_gates"]["action_accuracy_pass"] is False
    assert result["fit_gates"]["exact_sequence_rate_pass"] is False
    assert result["training_authorized"] is False
    assert result["phase3_certified"] is False
    assert result["phase4_open"] is False
    assert sha256_file(V51 / "training_fit_rows.jsonl") == result["rows_sha256"]
    assert sha256_file(V51 / "decision.json") == result["decision_file_sha256"]
    _verify_embedded_evidence_hash(decision)
