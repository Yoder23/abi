from abi.capability_compiler_phase4_b50_v23_conformance import (
    FORMAT,
    RESULT_FORMAT,
    _v23_manifest_document,
)


class _Manifest:
    tensor_payload_hash = "a" * 64

    def canonical_dict(self):
        return {
            "minimum_host_capabilities": {"features": ["safe_tensors"]},
            "tensor_payload_hash": self.tensor_payload_hash,
            "package_hash": "b" * 64,
            "evaluation_evidence": {},
        }


class _Package:
    manifest = _Manifest()


def test_v23_conformance_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-v23-conformance/1"
    assert RESULT_FORMAT == (
        "abi-capability-compiler-phase4-b50-v23-conformance-result/1"
    )


def test_v23_manifest_changes_interface_but_preserves_parent_payload_provenance():
    api = {
        "single_parse_feature": "single_authenticated_package_activation",
        "abi_version": "lc-direct-neural-core/23",
        "abi_sha256": "c" * 64,
    }
    document = _v23_manifest_document(_Package(), api, seed=104729)
    assert document["abi_version"] == "lc-direct-neural-core/23"
    assert document["tensor_payload_hash"] == ""
    assert document["package_hash"] == ""
    assert document["evaluation_evidence"]["parent_tensor_payload_hash"] == "a" * 64
    assert document["minimum_host_capabilities"]["features"] == [
        "safe_tensors",
        "single_authenticated_package_activation",
    ]
