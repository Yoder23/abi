from abi.capability_compiler_phase4_b40_baseline_pack import normalize_memberships


def test_b40_normalization_has_budget_specific_identity(monkeypatch):
    monkeypatch.setattr(
        "abi.capability_compiler_phase4_b40_baseline_pack.b50.normalize_memberships",
        lambda selected, tokenizer: [
            {"source_artifact": "phase1_ir", "native_record_id": "r", "format": "old", "ir_record_id": "old"}
        ],
    )
    rows = normalize_memberships({}, object())
    assert rows[0]["format"] == "abi-phase4-exact-b40-baseline-membership/1"
    assert len(rows[0]["ir_record_id"]) == 64
    assert rows[0]["ir_record_id"] != "old"
