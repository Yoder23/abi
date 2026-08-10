import json

from abi.capability_compiler_phase3_repetition_metric_audit import run


def test_audit_reads_rows_and_preserves_v1(monkeypatch, tmp_path):
    import abi.capability_compiler_phase3_repetition_metric_audit as module

    rows = [
        {"probe_id": "negative", "output": "A clear and ordinary sentence with enough useful words.", "repetition_collapse": False},
        {"probe_id": "positive", "output": "echo " * 10, "repetition_collapse": True},
    ]
    relative = "rows.jsonl"
    (tmp_path / relative).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "SYSTEMS", {"candidate_V443": relative, "teacher_T0": relative, "failed_A3": relative, "failed_C3": relative, "failed_D2": relative})
    monkeypatch.setattr(module, "NEGATIVE_REFERENCES", {"candidate_V443": ["negative"]})
    monkeypatch.setattr(module, "ACTUAL_POSITIVE_REFERENCE", ("candidate_V443", "positive"))
    monkeypatch.setattr(module, "SYNTHETIC_POSITIVES", ["echo " * 10])
    monkeypatch.setattr(module, "SYNTHETIC_NEGATIVES", ["ordinary unique prose remains healthy here today"])

    result = run(tmp_path, protocol, tmp_path / "result.json")

    assert result["systems"]["candidate_V443"]["historical_v1_collapses"] == 1
    assert result["systems"]["candidate_V443"]["recomputed_v1_collapses"] == 1
    assert result["controls"]["actual_v443_loop_detected"] is True
