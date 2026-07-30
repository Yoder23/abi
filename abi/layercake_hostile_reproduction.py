"""Run hostile fresh-root reproduction before final-test opening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence
import warnings
import zipfile

from .layercake_host_runtime import NativeHostRuntime
from .layercake_product_host import (
    LayerCakeProductHost,
    ProductHostError,
    verify_domain_package,
)


PROTOCOL_FORMAT = "abi-layercake-hostile-reproduction-protocol/1"
EVIDENCE_FORMAT = "abi-layercake-hostile-reproduction-evidence/1"
EXPECTED_LAYERCAKE_COMMIT = "04cf2927a16fba686cd640e18a78708e5658bbda"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProductHostError(f"JSON must be an object: {path}")
    return value


def _claim_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    claimed = payload.pop("evidence_sha256", None)
    if claimed != _canonical_sha(payload):
        raise ProductHostError("evidence claim hash mismatch")
    return str(claimed)


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "clean": not bool(status.strip()),
        "porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _rejection(name: str, action: Callable[[], Any]) -> dict[str, Any]:
    try:
        action()
    except Exception as exc:
        return {
            "name": name,
            "status": "PASS_REJECTED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    raise ProductHostError(f"hostile attack was accepted: {name}")


def _host_inputs(
    root: Path,
    combined: Mapping[str, Any],
    artifact_override: Path | None = None,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    artifact = (
        artifact_override
        if artifact_override is not None
        else (root / combined["english_host"]["artifact"]).resolve()
    )
    layercake_root = Path(
        combined["sealed_layercake"]["root"]
    ).resolve()
    protocol = _read(root / "COMBINED_LAYERCAKE_HOST_CERTIFICATION_PROTOCOL.json")
    packages = []
    for specification in protocol["domains"]:
        validation = _read(root / specification["validation"])
        first = validation["observations"][0]
        packages.append(
            {
                "domain": specification["domain"],
                "cake_id": specification["cake_id"],
                "package": (root / specification["package"]).resolve(),
                "public_key": (root / specification["public_key"]).resolve(),
                "prompt": first["prompt"],
                "expected_output_sha256": first["layercake_output_sha256"],
            }
        )
    return artifact, layercake_root, packages


def reproduce_hostile(
    *,
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    root = protocol_path.parent
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ProductHostError(f"hostile evidence is immutable: {output_path}")
    protocol = _read(protocol_path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status")
        != (
            "PREREGISTERED_AFTER_COMBINED_VALIDATION_PASS_"
            "BEFORE_HOSTILE_EXECUTION_AND_FINAL_TEST_OPENING"
        )
        or protocol.get("fresh_receiver_roots") != 3
        or protocol.get("final_test_accessed") is not False
    ):
        raise ProductHostError("hostile reproduction protocol is invalid")
    combined_spec = protocol["combined_host_certificate"]
    combined_path = (root / combined_spec["path"]).resolve()
    combined = _read(combined_path)
    if (
        _sha256_file(combined_path) != combined_spec["file_sha256"]
        or _claim_hash(combined) != combined_spec["evidence_sha256"]
        or combined.get("status") != "PASS"
        or combined.get("final_test_accessed") is not False
    ):
        raise ProductHostError("combined certificate prerequisite failed")
    artifact_override = None
    repair_spec = protocol.get("repair_certificate")
    override_spec = protocol.get("english_artifact_override")
    if repair_spec is not None or override_spec is not None:
        if not isinstance(repair_spec, dict) or not isinstance(
            override_spec, dict
        ):
            raise ProductHostError(
                "hostile English override contract is incomplete"
            )
        repair_path = (root / repair_spec["path"]).resolve()
        repair = _read(repair_path)
        artifact_override = (root / override_spec["path"]).resolve()
        metadata_path = artifact_override / "metadata.json"
        metadata = _read(metadata_path)
        if (
            _sha256_file(repair_path) != repair_spec["file_sha256"]
            or _claim_hash(repair) != repair_spec["evidence_sha256"]
            or repair.get("status") != "PASS"
            or repair.get("final_test_accessed") is not False
            or repair.get("candidate", {}).get("artifact")
            != str(artifact_override)
            or _sha256_file(metadata_path)
            != override_spec["metadata_file_sha256"]
            or metadata.get("evidence_sha256")
            != override_spec["metadata_evidence_sha256"]
            or metadata.get("runtime", {}).get("graph_sha256")
            != override_spec["runtime_graph_sha256"]
            or metadata.get("symbolic_surface", {}).get("sha256")
            != override_spec["symbolic_surface_sha256"]
            or metadata.get("host", {}).get(
                "deployment_manifest_sha256"
            )
            != override_spec["host_manifest_sha256"]
            or metadata.get("host", {}).get(
                "teacher_present_at_inference"
            )
            is not False
            or metadata.get("host", {}).get(
                "source_transformer_blocks_retained"
            )
            != 0
        ):
            raise ProductHostError(
                "hostile English override prerequisite failed"
            )
    artifact, layercake_root, packages = _host_inputs(
        root,
        combined,
        artifact_override,
    )
    before = _git_state(layercake_root)
    if (
        before["commit"] != EXPECTED_LAYERCAKE_COMMIT
        or before["clean"] is not True
    ):
        raise ProductHostError("sealed LayerCake is not pristine")

    receivers: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="abi-hostile-") as temporary:
        temporary_root = Path(temporary)
        for receiver_index in range(3):
            registry = temporary_root / f"receiver-{receiver_index}"
            with LayerCakeProductHost(
                english_artifact=artifact,
                layercake_root=layercake_root,
                registry_root=registry,
                threads=16,
            ) as host:
                ordered = (
                    packages
                    if receiver_index % 2 == 0
                    else list(reversed(packages))
                )
                installs = [
                    host.install(row["package"], row["public_key"])
                    for row in ordered
                ]
                inactive = host.telemetry()
                exact_value = f"HOSTILE-RECEIVER-{receiver_index}"
                english = host.generate(
                    f"Reply with exactly {exact_value} and nothing else.",
                    max_new_tokens=64,
                )
                domain_rows = []
                for row in ordered:
                    result = host.generate(
                        row["prompt"],
                        cake_id=row["cake_id"],
                        max_new_tokens=96,
                        domain_device="cpu",
                    )
                    delta = result.evidence["telemetry_delta"]
                    inactive_calls = sum(
                        sum(values.values())
                        for cake_id, values in delta.items()
                        if cake_id != row["cake_id"]
                    )
                    if (
                        result.output_sha256
                        != row["expected_output_sha256"]
                        or inactive_calls
                    ):
                        raise ProductHostError(
                            "hostile receiver output or isolation changed"
                        )
                    domain_rows.append(
                        {
                            "domain": row["domain"],
                            "cake_id": row["cake_id"],
                            "output_sha256": result.output_sha256,
                            "inactive_execution_calls": inactive_calls,
                        }
                    )
                receivers.append(
                    {
                        "receiver_index": receiver_index,
                        "fresh_registry": True,
                        "install_order": [
                            row["cake_id"] for row in ordered
                        ],
                        "install_statuses": [
                            row["status"] for row in installs
                        ],
                        "worker_lazy_before_selection": (
                            inactive["active_domain_worker_devices"] == []
                        ),
                        "english_output": english.output,
                        "english_exact": english.output == exact_value,
                        "domain_results": domain_rows,
                    }
                )
            if host.telemetry()["active_domain_worker_devices"]:
                raise ProductHostError("domain worker survived receiver close")

        primary = packages[0]
        tampered = temporary_root / "tampered.cake"
        raw = bytearray(primary["package"].read_bytes())
        raw[len(raw) // 2] ^= 1
        tampered.write_bytes(raw)

        duplicate = temporary_root / "duplicate.cake"
        with zipfile.ZipFile(primary["package"], "r") as source:
            members = [(name, source.read(name)) for name in source.namelist()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(
                duplicate, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                for name, data in members:
                    archive.writestr(name, data)
                archive.writestr("manifest.json", members[0][1])

        traversal = temporary_root / "traversal.cake"
        with zipfile.ZipFile(
            traversal, "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for name, data in members:
                archive.writestr(name, data)
            archive.writestr("../escape", b"forbidden")

        wrong_key = packages[1]["public_key"]
        attacks = [
            _rejection(
                "single_byte_package_corruption",
                lambda: verify_domain_package(tampered, primary["public_key"]),
            ),
            _rejection(
                "wrong_ed25519_publisher_key",
                lambda: verify_domain_package(primary["package"], wrong_key),
            ),
            _rejection(
                "duplicate_archive_member",
                lambda: verify_domain_package(duplicate, primary["public_key"]),
            ),
            _rejection(
                "path_traversal_archive_member",
                lambda: verify_domain_package(traversal, primary["public_key"]),
            ),
        ]
        with LayerCakeProductHost(
            english_artifact=artifact,
            layercake_root=layercake_root,
            registry_root=temporary_root / "uninstalled",
            threads=16,
        ) as uninstalled_host:
            attacks.append(
                _rejection(
                    "uninstalled_cake_selection",
                    lambda: uninstalled_host.generate(
                        "hostile",
                        cake_id=primary["cake_id"],
                        max_new_tokens=8,
                    ),
                )
            )

        corrupted_artifact = temporary_root / "corrupted-native"
        shutil.copytree(artifact, corrupted_artifact)
        vocabulary = corrupted_artifact / "output-vocabulary.json"
        vocabulary_raw = bytearray(vocabulary.read_bytes())
        vocabulary_raw[len(vocabulary_raw) // 2] ^= 1
        vocabulary.write_bytes(vocabulary_raw)
        attacks.append(
            _rejection(
                "stale_native_runtime_component",
                lambda: NativeHostRuntime(corrupted_artifact, threads=1),
            )
        )

    after = _git_state(layercake_root)
    gates = {
        "three_fresh_receivers": len(receivers) == 3,
        "all_receiver_english_exact": all(
            row["english_exact"] for row in receivers
        ),
        "all_workers_lazy_before_selection": all(
            row["worker_lazy_before_selection"] for row in receivers
        ),
        "all_domain_outputs_exact": all(
            all(
                domain["output_sha256"]
                == next(
                    package["expected_output_sha256"]
                    for package in packages
                    if package["cake_id"] == domain["cake_id"]
                )
                for domain in row["domain_results"]
            )
            for row in receivers
        ),
        "all_inactive_execution_zero": all(
            all(
                domain["inactive_execution_calls"] == 0
                for domain in row["domain_results"]
            )
            for row in receivers
        ),
        "all_attacks_rejected": all(
            row["status"] == "PASS_REJECTED" for row in attacks
        ),
        "sealed_layercake_unchanged": after == before,
        "final_test_unopened": True,
    }
    evidence: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "combined_host_certificate": combined_spec,
        "receivers": receivers,
        "attacks": attacks,
        "gates": gates,
        "sealed_layercake": {
            "root": str(layercake_root),
            "before": before,
            "after": after,
            "unchanged": after == before,
        },
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "final_test_accessed": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = reproduce_hostile(
        protocol_path=args.protocol,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_sha256": result["evidence_sha256"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
