import copy
import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase6_composition import load_protocol
from abi.capability_compiler_phase6_verify import (
    _read_jsonl,
    verify_seed_document,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "ABI_CAPABILITY_COMPILER_PHASE6_COMPOSITION_REPAIR_PROTOCOL_V1034.json"


@pytest.fixture(scope="module")
def exact_fixture():
    protocol, protocol_sha = load_protocol(ROOT, PROTOCOL_PATH)
    base = ROOT / "results/abi_capability_compiler_phase6_composition/run_v1032/seed104729"
    return (
        protocol,
        protocol_sha,
        json.loads((base / "result.json").read_text(encoding="utf-8")),
        _read_jsonl(base / "observations.jsonl"),
        json.loads((base / "provenance.json").read_text(encoding="utf-8")),
    )


def _verify(fixture, mutation):
    protocol, protocol_sha, raw_result, raw_observations, raw_provenance = fixture
    result = copy.deepcopy(raw_result)
    observations = copy.deepcopy(raw_observations)
    provenance = copy.deepcopy(raw_provenance)
    if mutation is not None:
        mutation(result, observations, provenance)
    return verify_seed_document(
        root=ROOT,
        protocol=protocol,
        protocol_sha=protocol_sha,
        seed=104729,
        result=result,
        observations=observations,
        provenance=provenance,
    )["gates"]


def test_exact_phase6_seed_recomputes_cleanly(exact_fixture):
    assert all(_verify(exact_fixture, None).values())


@pytest.mark.parametrize(
    "mutation,failed_gate",
    [
        (lambda r, o, p: o.pop(), "raw_row_depth"),
        (
            lambda r, o, p: next(
                row for row in o if row["mode"] == "composed_host_selected_domain"
            ).update(output="not a valid answer"),
            "selected_function_recomputed",
        ),
        (
            lambda r, o, p: next(
                row for row in o if row["mode"] == "composed_host_selected_domain"
            ).update(output="loop " * 100),
            "selected_zero_repetition_collapse_v2",
        ),
        (
            lambda r, o, p: next(
                row for row in o if row["mode"] == "composed_host_selected_domain"
            )["telemetry_delta"].update(evil={"prefill_calls": 1}),
            "selected_execution_recomputed",
        ),
        (
            lambda r, o, p: next(
                row for row in o if row["mode"] == "composed_host_selected_domain"
            ).update(selected=["wrong-cake"]),
            "selected_execution_recomputed",
        ),
        (
            lambda r, o, p: next(
                row for row in o if row["mode"] == "structured_three_domain_composition"
            )["components"].pop(),
            "composition_index_and_component_identity",
        ),
        (
            lambda r, o, p: next(
                row for row in o if row["mode"] == "structured_three_domain_composition"
            )["components"][0].update(output="invalid"),
            "composition_function_recomputed",
        ),
        (
            lambda r, o, p: next(
                row for row in o if row["mode"] == "structured_three_domain_composition"
            )["components"][0]["telemetry_delta"].update(evil={"decode_step_calls": 1}),
            "composition_execution_recomputed",
        ),
        (
            lambda r, o, p: next(
                row for row in o if row["mode"] == "structured_three_domain_composition"
            )["components"][0].update(output="repeat " * 100),
            "composition_zero_repetition_collapse_v2",
        ),
        (
            lambda r, o, p: next(
                row for row in o if row["mode"] == "conflict_quarantine"
            ).update(core_fallback=True),
            "conflict_quarantine_recomputed",
        ),
        (
            lambda r, o, p: r.update(protocol_sha256="0" * 64),
            "seed_and_protocol_identity",
        ),
        (
            lambda r, o, p: r["core_after"].update(payload_hash="0" * 64),
            "core_identity_recomputed",
        ),
        (
            lambda r, o, p: r["registry_archive_hashes"].update(
                {"abi-python-token-plan": "0" * 64}
            ),
            "package_registry_identity_recomputed",
        ),
        (
            lambda r, o, p: p["domains"]["python"]["selected_record_ids"].pop(),
            "provenance_recomputed",
        ),
        (
            lambda r, o, p: p["deletion_lineage"]["record_id_to_artifact"].popitem(),
            "provenance_recomputed",
        ),
        (
            lambda r, o, p: p.update(deployed_multiple_source_models=True),
            "provenance_recomputed",
        ),
    ],
)
def test_adversarial_phase6_mutations_fail_closed(
    exact_fixture, mutation, failed_gate
):
    gates = _verify(exact_fixture, mutation)
    assert gates[failed_gate] is False
