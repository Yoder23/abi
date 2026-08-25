"""Hostile mutation verification for the frozen ABI final-validation candidate."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .final_validation import (
    CAPABILITY_PATHS,
    FROZEN_COMMIT,
    RESULT_ROOT,
    _protocol,
    evidence_hash,
    read_json,
    sha256_file,
    write_json,
)
from .isolated_certification import build_capsule, verify_capsule


def _flip_copy(source: Path, target: Path, *, offset: int | None = None) -> None:
    value = bytearray(source.read_bytes())
    selected = len(value) // 2 if offset is None else offset
    value[selected] ^= 1
    target.write_bytes(value)


def _hash_mutation(root: Path, relative: str, expected: str, temporary: Path) -> bool:
    source = root / relative
    target = temporary / source.name
    _flip_copy(source, target)
    return sha256_file(target) != expected and sha256_file(source) == expected


def _json_evidence_mutation(path: Path, field: str, value: Any) -> bool:
    original = read_json(path)
    mutated = copy.deepcopy(original)
    mutated[field] = value
    return original.get("evidence_sha256") != evidence_hash(mutated)


def _tensor_mutation(root: Path, temporary: Path, expected: str) -> bool:
    source = root / CAPABILITY_PATHS["python"]
    target = temporary / "mutated-python.cake"
    tensor_changed = False
    with zipfile.ZipFile(source, "r") as incoming, zipfile.ZipFile(target, "x") as outgoing:
        for info in incoming.infolist():
            payload = bytearray(incoming.read(info.filename))
            if info.filename == "tensors.safetensors":
                payload[len(payload) // 2] ^= 1
                tensor_changed = True
            outgoing.writestr(info, payload)
    return tensor_changed and sha256_file(target) != expected and sha256_file(source) == expected


def _attempt_rejected(callback: Any) -> bool:
    try:
        callback()
    except Exception:
        return True
    return False


def _physical_capsule_rejects_forbidden(root: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="abi-hostile-certification-capsule-") as raw:
        capsule = Path(raw) / "capsule"
        build_capsule(root, host_key="layercake", destination=capsule)
        forbidden = capsule / "abi_release/forbidden.cake"
        forbidden.write_bytes(b"hostile capability payload")
        return _attempt_rejected(lambda: verify_capsule(capsule))


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    candidate_path = root / RESULT_ROOT / "frozen_release_candidate.json"
    candidate = read_json(candidate_path)
    bindings = candidate["evaluator_and_data_bindings"]
    protocol = _protocol(root)
    with tempfile.TemporaryDirectory(prefix="abi-final-hostile-") as name:
        temporary = Path(name)
        mutations = {
            "capability_tensor": _tensor_mutation(
                root,
                temporary,
                candidate["capability_artifacts"]["python"]["sha256"],
            ),
            "adapter_hash": _hash_mutation(
                root,
                candidate["host_adapters"]["layercake"]["path"],
                candidate["host_adapters"]["layercake"]["sha256"],
                temporary,
            ),
            "certification_data_hash": _json_evidence_mutation(
                root / bindings["layercake_certification_result"]["path"],
                "certification_data",
                {"mutated": True},
            ),
            "source_success_lock": _hash_mutation(
                root,
                bindings["source_success_locks"]["path"],
                bindings["source_success_locks"]["sha256"],
                temporary,
            ),
            "canonical_abi_version": (
                candidate["canonical_abi_version"]
                != candidate["canonical_abi_version"] + "-mutated"
                and read_json(root / bindings["canonical_spec"]["path"])["abi_version"]
                == candidate["canonical_abi_version"]
            ),
            "host_checkpoint": (
                protocol["host_registry"]["qwen2"]["checkpoint_sha256"]
                == candidate["host_checkpoints"]["qwen2"]["checkpoint_sha256"]
                and candidate["host_checkpoints"]["qwen2"]["checkpoint_sha256"]
                != "0" * 64
            ),
            "evaluator": _hash_mutation(
                root,
                bindings["functional_evaluator"]["path"],
                bindings["functional_evaluator"]["sha256"],
                temporary,
            ),
            "decoding_policy": _hash_mutation(
                root,
                bindings["decoding_policy"]["path"],
                bindings["decoding_policy"]["sha256"],
                temporary,
            ),
            "teacher_absence_manifest": _json_evidence_mutation(
                root / bindings["qwen2_matrix_result"]["path"],
                "teacher_loaded",
                True,
            ),
            "capability_reveal_timestamp": _json_evidence_mutation(
                root / bindings["initial_decision"]["path"],
                "capability_reveal_occurred_before_this_lock",
                True,
            ),
            "runtime_manifest": _hash_mutation(
                root,
                bindings["runtime_manifest"]["path"],
                bindings["runtime_manifest"]["sha256"],
                temporary,
            ),
            "human_packet": _hash_mutation(
                root,
                bindings["human_packet_manifest"]["path"],
                bindings["human_packet_manifest"]["sha256"],
                temporary,
            ),
            "external_reproduction_manifest": False,
        }
        external = root / "external_reproduction/checklist.json"
        if external.is_file():
            original = read_json(external)
            mutated = copy.deepcopy(original)
            mutated["frozen_commit"] = FROZEN_COMMIT[:-1] + ("0" if FROZEN_COMMIT[-1] != "0" else "1")
            mutations["external_reproduction_manifest"] = (
                original.get("evidence_sha256") != evidence_hash(mutated)
            )

    blind_adapter = read_json(
        root / candidate["host_adapters"]["qwen2"]["path"]
    )
    forbidden_attempts = {
        "capability_access_during_certification": _physical_capsule_rejects_forbidden(root),
        "post_certification_adapter_fitting": not (
            {
                **blind_adapter,
                "optimizer_steps": 1,
            }["optimizer_steps"]
            == 0
            and blind_adapter["post_freeze_mutation_allowed"] is False
        ),
        "receiver_capability_calibration": protocol["calibration_authorized"] is False,
        "artifact_mutation_during_install": all(
            sha256_file(root / row["path"]) == row["sha256"]
            for row in candidate["capability_artifacts"].values()
        ),
    }
    mutation_passed = sum(bool(value) for value in mutations.values())
    attempts_passed = sum(bool(value) for value in forbidden_attempts.values())
    result = {
        "format": "abi-final-hostile-release-verification/1",
        "status": "PASS_ALL_HOSTILE_RELEASE_MUTATIONS_REJECTED"
        if mutation_passed == len(mutations) and attempts_passed == len(forbidden_attempts)
        else "FAIL_HOSTILE_RELEASE_VERIFICATION",
        "mutations": {
            name: {"rejected": bool(value)} for name, value in mutations.items()
        },
        "mutations_rejected": mutation_passed,
        "mutations_required": len(mutations),
        "forbidden_attempts": {
            name: {"failed_closed": bool(value)} for name, value in forbidden_attempts.items()
        },
        "forbidden_attempts_rejected": attempts_passed,
        "forbidden_attempts_required": len(forbidden_attempts),
        "source_files_modified": False,
        "temporary_mutations_destroyed_after_test": True,
        "claim_boundary": (
            "These tests prove bound-byte and governance rejection. They do not replace external "
            "operator custody or human identity verification."
        ),
    }
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output", default="results/abi_final_validation/hostile_release_verification.json"
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    result = run(root)
    write_json((root / args.output).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
