import copy
import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase5_selective_product import load_protocol
from abi.capability_compiler_phase5_verify import (
    _read_jsonl,
    verify_result_document,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "ABI_CAPABILITY_COMPILER_PHASE5_SELECTIVE_PRODUCT_REPAIR_PROTOCOL_V1026.json"


def _fixture():
    protocol, protocol_sha = load_protocol(ROOT, PROTOCOL_PATH)
    result = json.loads(
        (
            ROOT
            / "results/abi_capability_compiler_phase5_selective_product/run_v1024/ABI-seed104729/result.json"
        ).read_text(encoding="utf-8")
    )
    observations = _read_jsonl(
        ROOT
        / "results/abi_capability_compiler_phase5_selective_product/run_v1024/ABI-seed104729/observations.jsonl"
    )
    return protocol, protocol_sha, result, observations


def _verify(result, observations):
    protocol, protocol_sha, _, _ = _fixture()
    return verify_result_document(
        root=ROOT,
        protocol=protocol,
        product_protocol_sha=protocol_sha,
        system="ABI",
        seed=104729,
        result=result,
        observations=observations,
    )["gates"]


def test_exact_abi_evidence_recomputes_cleanly():
    _, _, result, observations = _fixture()
    assert all(_verify(result, observations).values())


@pytest.mark.parametrize(
    "mutation,failed_gate",
    [
        (lambda r, o: o.pop(), "row_mode_depth"),
        (lambda r, o: o.append(copy.deepcopy(o[0])), "row_mode_depth"),
        (
            lambda r, o: next(x for x in o if x["mode"] == "selected_domain_installed").update(output="invalid"),
            "selected_function_recomputed",
        ),
        (
            lambda r, o: next(x for x in o if x["mode"] == "domain_missing_before_install").update(output="Hydrogen is element one."),
            "missing_abstention_recomputed",
        ),
        (
            lambda r, o: next(x for x in o if x["mode"] == "domain_missing_before_install").update(core_fallback=True),
            "missing_zero_execution_recomputed",
        ),
        (
            lambda r, o: next(x for x in o if x["mode"] == "domain_missing_before_install")["telemetry_delta"].update(evil={"prefill_calls": 1}),
            "missing_zero_execution_recomputed",
        ),
        (
            lambda r, o: next(x for x in o if x["mode"] == "label_spoof").update(selected=["wrong-cake"]),
            "spoof_outer_selection_recomputed",
        ),
        (
            lambda r, o: next(x for x in o if x["mode"] == "cross_domain_uninstalled").update(selected=["wrong-cake"]),
            "cross_domain_zero_execution_recomputed",
        ),
        (
            lambda r, o: next(x for x in o if x["mode"] == "quarantine").update(core_fallback=True),
            "quarantine_zero_execution_recomputed",
        ),
        (
            lambda r, o: r["core_after"].update(payload_hash="0" * 64),
            "core_identity_recomputed",
        ),
        (
            lambda r, o: r["lifecycle"][0].update(archive_unchanged=False),
            "lifecycle_recomputed",
        ),
        (
            lambda r, o: r.update(protocol_sha256="0" * 64),
            "protocol_identity",
        ),
    ],
)
def test_adversarial_evidence_mutations_fail_closed(mutation, failed_gate):
    _, _, result, observations = _fixture()
    mutation(result, observations)
    gates = _verify(result, observations)
    assert gates[failed_gate] is False
    assert gates["result_evidence_hash"] is False or failed_gate in {
        "row_mode_depth",
        "selected_function_recomputed",
        "missing_abstention_recomputed",
        "missing_zero_execution_recomputed",
        "spoof_outer_selection_recomputed",
        "cross_domain_zero_execution_recomputed",
        "quarantine_zero_execution_recomputed",
    }
