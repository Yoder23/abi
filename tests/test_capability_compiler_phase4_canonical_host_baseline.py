import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_canonical_host_baseline import load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_CANONICAL_HOST_BASELINE_PROTOCOL_V628.json"


def test_protocol_loads_immutable_host() -> None:
    protocol, _, base = load_protocol(ROOT, PROTOCOL)
    assert protocol["host_checkpoint_sha256"] == base["host"]["parent_checkpoint_sha256"]


def test_final_access_mutation_fails_closed(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["final_test_access"] = "ALLOWED"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance changed"):
        load_protocol(ROOT, path)
