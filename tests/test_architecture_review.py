from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_architecture_review_preserves_claim_boundary():
    doc = (ROOT / "ABI_ARCHITECTURE_REVIEW.md").read_text(encoding="utf-8")

    assert "not yet architecturally sufficient" in doc
    assert "lossless universal domain migration" in doc
    assert "scoped frozen-core ABI transfer" in doc


def test_architecture_review_names_current_structural_blockers():
    doc = (ROOT / "ABI_ARCHITECTURE_REVIEW.md").read_text(encoding="utf-8")

    for phrase in [
        "Phase C native target ABI oracle",
        "target native oracle",
        "source-preservation evaluation",
        "selective-transfer evaluation",
        "off-domain noninterference",
        "oracle-light calibration",
        "ABIArtifact",
        "compatibility certificate",
        "cost ledger",
    ]:
        assert phrase in doc


def test_architecture_review_blocks_lateral_experiments():
    doc = (ROOT / "ABI_ARCHITECTURE_REVIEW.md").read_text(encoding="utf-8")

    assert "Do not start more frontier runs" in doc
    assert "If an experiment does not answer one of those questions" in doc
