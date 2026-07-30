from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_doc(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_formal_transfer_theorem_states_impossibility_and_conditional_result():
    doc = read_doc("FORMAL_UNIVERSAL_TRANSFER.md")
    assert "Theorem 1 (No assumption-free universal transfer)" in doc
    assert "Theorem 2 (Universal ABI transfer over the compatible class)" in doc
    assert "No algorithm can guarantee non-inferior domain transfer" in doc
    assert "all models satisfying an ABI compatibility certificate" in doc


def test_formal_transfer_theorem_names_all_certificate_assumptions():
    doc = read_doc("FORMAL_UNIVERSAL_TRANSFER.md")
    for heading in [
        "A1. Coordinate alignment",
        "A2. Domain equivariance",
        "A3. Interface calibration",
        "A4. Output-head Lipschitzness",
        "A5. Top-k margin",
    ]:
        assert heading in doc


def test_formal_transfer_theorem_defines_next_generation_certificate():
    doc = read_doc("FORMAL_UNIVERSAL_TRANSFER.md")
    assert "What a GPT-5 -> GPT-6 Claim Would Need" in doc
    assert '"domain_core_frozen": true' in doc
    assert '"nib_certificate"' in doc
    assert "without full target retraining" in doc


def test_release_docs_link_formal_theorem_without_overclaiming():
    readme = read_doc("README.md")
    claims = read_doc("CLAIMS.md")
    assert "FORMAL_UNIVERSAL_TRANSFER.md" in readme
    assert "FORMAL_UNIVERSAL_TRANSFER.md" in claims
    assert "Assumption-free theorem-level mathematical proof over all possible models" in claims
    assert "Automatic transfer to a new target model without an ABI compatibility certificate" in claims
