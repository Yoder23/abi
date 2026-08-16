import copy

import pytest


def _valid():
    return {
        "format": "abi-capability-compiler-phase8-release-readiness-result/1",
        "status": "PASS_PHASE8_LOCAL_RELEASE_READINESS",
        "source": {
            "abi_phase7_seal_commit": "a" * 40,
            "layercake_commit": "b" * 40,
        },
        "development_hardware": {"fingerprint_sha256": "c" * 64},
        "files": {"artifact": {"sha256": "d" * 64, "bytes": 4}},
        "file_count": 1,
        "total_bytes": 4,
        "external_gates": {
            "independent_operator_complete": False,
            "independent_hardware_complete": False,
        },
        "phase8_certified": False,
    }


def _gates(document):
    expected_files = {"artifact": {"sha256": "d" * 64, "bytes": 4}}
    return {
        "format_exact": document["format"]
        == "abi-capability-compiler-phase8-release-readiness-result/1",
        "local_status_exact": document["status"]
        == "PASS_PHASE8_LOCAL_RELEASE_READINESS",
        "file_inventory_exact": document["files"] == expected_files,
        "file_count_exact": document["file_count"] == 1,
        "byte_count_exact": document["total_bytes"] == 4,
        "abi_phase7_seal_ancestor": document["source"]["abi_phase7_seal_commit"]
        == "a" * 40,
        "layercake_commit_exact": document["source"]["layercake_commit"]
        == "b" * 40,
        "development_hardware_bound": document["development_hardware"][
            "fingerprint_sha256"
        ]
        == "c" * 64,
        "external_operator_not_self_attested": document["external_gates"][
            "independent_operator_complete"
        ]
        is False,
        "external_hardware_not_self_attested": document["external_gates"][
            "independent_hardware_complete"
        ]
        is False,
        "phase8_not_certified_locally": document["phase8_certified"] is False,
    }


def test_exact_local_readiness_document_passes():
    assert all(_gates(_valid()).values())


@pytest.mark.parametrize(
    "mutation,failed_gate",
    [
        (lambda d: d.update(format="wrong"), "format_exact"),
        (lambda d: d.update(status="CERTIFIED"), "local_status_exact"),
        (lambda d: d["files"].clear(), "file_inventory_exact"),
        (lambda d: d.update(file_count=2), "file_count_exact"),
        (lambda d: d.update(total_bytes=5), "byte_count_exact"),
        (
            lambda d: d["source"].update(abi_phase7_seal_commit="0" * 40),
            "abi_phase7_seal_ancestor",
        ),
        (
            lambda d: d["source"].update(layercake_commit="0" * 40),
            "layercake_commit_exact",
        ),
        (
            lambda d: d["development_hardware"].update(fingerprint_sha256="0" * 64),
            "development_hardware_bound",
        ),
        (
            lambda d: d["external_gates"].update(independent_operator_complete=True),
            "external_operator_not_self_attested",
        ),
        (
            lambda d: d["external_gates"].update(independent_hardware_complete=True),
            "external_hardware_not_self_attested",
        ),
        (lambda d: d.update(phase8_certified=True), "phase8_not_certified_locally"),
        (
            lambda d: d["files"]["artifact"].update(sha256="0" * 64),
            "file_inventory_exact",
        ),
        (
            lambda d: d["files"]["artifact"].update(bytes=5),
            "file_inventory_exact",
        ),
        (
            lambda d: d["files"].update(extra={"sha256": "e" * 64, "bytes": 1}),
            "file_inventory_exact",
        ),
        (
            lambda d: d["development_hardware"].update(fingerprint_sha256="C" * 64),
            "development_hardware_bound",
        ),
    ],
)
def test_local_readiness_mutations_fail_closed(mutation, failed_gate):
    document = copy.deepcopy(_valid())
    mutation(document)
    assert _gates(document)[failed_gate] is False
