import copy
import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_fit_diagnostic_verifier import verify_document


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/abi_capability_compiler_phase3_fit_diagnostic/fit_generalization_v26.json"


def stored_result():
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_exact_v26_document_passes_pure_integrity_verification():
    result = stored_result()
    assert verify_document(result, result)["status"] == "PASS"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("phase3_certified",), True),
        (("systems", "V23", "training_teacher_forced", "action_accuracy"), 1.0),
        (("systems", "V24", "checkpoint_sha256"), "0" * 64),
        (("ownership", "layercake_host_regression"), True),
    ],
)
def test_v26_mutations_are_rejected(path, replacement):
    expected = stored_result()
    mutated = copy.deepcopy(expected)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(Phase3Error):
        verify_document(mutated, expected)
