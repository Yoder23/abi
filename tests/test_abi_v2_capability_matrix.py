from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from abi_v2.capability_matrix import (
    FrozenHostAdapter,
    MatrixError,
    _domain_generate,
    _mutate_random_equal_size,
    _mutate_shuffle_equal_size,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistered_matrix_amendment_binds_implementation_and_frozen_inputs() -> None:
    base_path = ROOT / "abi_v2/matrix_protocol.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    amendment = json.loads(
        (ROOT / "abi_v2/matrix_protocol_amendment3.json").read_text(encoding="utf-8")
    )
    assert amendment["status"] == "PREREGISTERED_PRE_OBSERVATION_UTF8_TYPE_AMENDMENT"
    assert _sha256(base_path) == amendment["base_protocol_sha256"]
    assert amendment["receiver_observations_before_amendment"] == 0
    assert amendment["bounded_architecture_repair_consumed"] is False
    assert amendment["receiver_quality_aggregate_seen_before_amendment"] is True
    assert amendment["quality_or_isolation_change_authorized"] is False
    assert amendment["bounded_instrumentation_repair_consumed"] is True
    assert base["hosts"] == ["layercake", "qwen2", "pythia"]
    assert base["capabilities"] == ["english", "python", "chemistry", "civics"]
    assert (
        _sha256(ROOT / amendment["implementation"]["path"])
        == amendment["implementation"]["sha256"]
    )
    for relative, expected in {**base["bindings"], **amendment["bindings"]}.items():
        assert _sha256((ROOT / relative).resolve()) == expected


def test_source_success_locks_have_preregistered_depth() -> None:
    locks = json.loads(
        (ROOT / "results/abi_v2/semantic_retention/source_success_locks.json").read_text(
            encoding="utf-8"
        )
    )
    assert locks["english"]["successful_task_count"] == 1381
    assert {
        domain: value["successful_task_count"] for domain, value in locks["domains"].items()
    } == {"python": 100, "chemistry": 100, "civics": 100}
    assert locks["receiver_outputs_seen_before_lock"] is False


def test_frozen_identity_adapter_realizes_exact_utf8_and_fails_when_removed() -> None:
    manifest = json.loads(
        (ROOT / "results/abi_v2/adapters/manifest.json").read_text(encoding="utf-8")
    )
    binding = manifest["adapters"]["layercake"]
    adapter = FrozenHostAdapter(
        path=ROOT / binding["path"], expected_sha256=binding["sha256"]
    )
    realized = adapter.realize(
        prompt="Preserve café and 東.",
        output="café and 東",
        capability_id="test-only",
        position=0,
    )
    assert realized["output"] == "café and 東"
    adapter.enabled = False
    with pytest.raises(MatrixError, match="absent"):
        adapter.realize(
            prompt="neutral",
            output="neutral",
            capability_id="test-only",
            position=1,
        )


def test_random_and_shuffled_hostile_packages_preserve_size_not_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.cake"
    source.write_bytes(bytes(range(256)) * 10000 + b"tail")
    random_path = tmp_path / "random.cake"
    shuffled_path = tmp_path / "shuffled.cake"
    _mutate_random_equal_size(source, random_path, seed=19)
    _mutate_shuffle_equal_size(source, shuffled_path)
    assert source.stat().st_size == random_path.stat().st_size == shuffled_path.stat().st_size
    assert _sha256(source) != _sha256(random_path)
    assert _sha256(source) != _sha256(shuffled_path)


def test_wrong_english_control_uses_raw_prompt_without_domain_wrapper() -> None:
    class Generated:
        output = b"wrong-package-output"
        actions = (3, 5)

    class Host:
        def __init__(self) -> None:
            self.prompt = ""

        def generate(self, _cake_id: str, prompt: str) -> Generated:
            self.prompt = prompt
            return Generated()

    class Runtime:
        host = Host()

    runtime = Runtime()
    output, actions = _domain_generate(
        runtime,
        {"python": {"cake_id": "python-test"}},
        "python",
        "ordinary English prompt",
        catalog_wrapped=False,
    )
    assert output == "wrong-package-output"
    assert actions == [3, 5]
    assert runtime.host.prompt == "ordinary English prompt\n"
