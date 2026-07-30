"""Certify ABI-produced domain cakes against LayerCake's sealed direct ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import torch

from .layercake_domains import (
    DIRECT_ABI_SHA256,
    DIRECT_ABI_VERSION,
    LAYERCAKE_COMMIT,
    DomainConformanceError,
    _canonical_sha,
    _import_layercake,
    _sha256_file,
    _write_json,
)


PROTOCOL_FORMAT = "abi-layercake-domain-package-certification-protocol/1"
EVIDENCE_FORMAT = "abi-layercake-domain-package-certification-evidence/1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DomainConformanceError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise DomainConformanceError(f"JSON evidence must be an object: {path}")
    return value


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
        "porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _relative(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DomainConformanceError("protocol path escapes ABI repository") from exc
    if not path.is_file():
        raise DomainConformanceError(f"protocol evidence is missing: {value}")
    return path


def _load_protocol(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if (
        value.get("format") != PROTOCOL_FORMAT
        or value.get("status")
        != "PREREGISTERED_BEFORE_PACKAGE_RUNTIME_CERTIFICATION"
    ):
        raise DomainConformanceError("package certification protocol is invalid")
    target = value.get("immutable_layercake_target", {})
    if (
        target.get("repository_commit") != LAYERCAKE_COMMIT
        or target.get("direct_decoder_abi_version") != DIRECT_ABI_VERSION
        or target.get("direct_decoder_abi_sha256") != DIRECT_ABI_SHA256
        or target.get("sealed_repository_may_be_modified") is not False
    ):
        raise DomainConformanceError("package protocol does not bind the sealed ABI")
    if value.get("required_seeds") != [9824, 9825, 9826]:
        raise DomainConformanceError("package protocol requires exact paired seeds")
    if value.get("receiver_count") != 3:
        raise DomainConformanceError("package protocol requires three receivers")
    domains = value.get("domains")
    if (
        not isinstance(domains, list)
        or len(domains) != 3
        or len({row.get("domain") for row in domains}) != len(domains)
    ):
        raise DomainConformanceError("package protocol domains are invalid")
    return value


def _validate_domain_lineage(
    repository_root: Path,
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    domain = str(specification["domain"])
    runs = specification.get("runs")
    if not isinstance(runs, list) or len(runs) != 3:
        raise DomainConformanceError(f"{domain} lacks three paired runs")
    run_evidence: list[dict[str, Any]] = []
    for expected_seed, run in zip((9824, 9825, 9826), runs):
        if run.get("seed") != expected_seed:
            raise DomainConformanceError(f"{domain} seed order is invalid")
        training_path = _relative(repository_root, str(run["training"]))
        validation_path = _relative(repository_root, str(run["validation"]))
        training = _read_json(training_path)
        validation = _read_json(validation_path)
        if (
            training.get("format")
            != "abi-layercake-domain-training-evidence/1"
            or training.get("domain") != domain
            or training.get("seed") != expected_seed
            or training.get("training_rows") != specification["training_rows"]
            or training.get("accounting", {}).get("teacher_tokens")
            != specification["teacher_tokens"]
            or training.get("parameter_count") != specification["parameter_count"]
            or training.get("final_test_accessed") is not False
            or training.get("validation_accessed") is not False
        ):
            raise DomainConformanceError(f"{domain} training lineage mismatch")
        if (
            validation.get("format")
            != "abi-layercake-domain-validation-evidence/1"
            or validation.get("domain") != domain
            or validation.get("seed") != expected_seed
            or validation.get("status") != "PASS"
            or validation.get("observation_count") != 100
            or validation.get("source_passing_regressions") != 0
            or validation.get("layercake_passes", 0)
            < validation.get("source_passes", 0)
            or validation.get("invalid_utf8_outputs") != 0
            or validation.get("teacher_present_at_inference") is not False
            or validation.get("source_transformer_blocks_retained") != 0
            or validation.get("validation_used_for_training") is not False
            or validation.get("final_test_accessed") is not False
        ):
            raise DomainConformanceError(f"{domain} validation run failed gates")
        run_evidence.append(
            {
                "seed": expected_seed,
                "training_evidence_sha256": _sha256_file(training_path),
                "training_evidence_claim_sha256": training["evidence_sha256"],
                "validation_evidence_sha256": _sha256_file(validation_path),
                "validation_evidence_claim_sha256": validation["evidence_sha256"],
                "layercake_passes": validation["layercake_passes"],
                "source_passes": validation["source_passes"],
                "source_passing_regressions": 0,
            }
        )
    lower_path = _relative(
        repository_root, str(specification["adjacent_lower_failure"])
    )
    lower = _read_json(lower_path)
    if (
        lower.get("domain") != domain
        or lower.get("status") != "FAIL"
        or lower.get("source_passing_regressions", 0) <= 0
        or lower.get("final_test_accessed") is not False
    ):
        raise DomainConformanceError(f"{domain} adjacent lower budget did not fail")
    return {
        "domain": domain,
        "minimum_tested_passing_budget_index": specification[
            "minimum_tested_passing_budget_index"
        ],
        "training_rows": specification["training_rows"],
        "teacher_tokens": specification["teacher_tokens"],
        "parameter_count": specification["parameter_count"],
        "paired_runs": run_evidence,
        "adjacent_lower_failure": {
            "path": str(specification["adjacent_lower_failure"]),
            "evidence_sha256": _sha256_file(lower_path),
            "layercake_passes": lower["layercake_passes"],
            "source_passes": lower["source_passes"],
            "source_passing_regressions": lower[
                "source_passing_regressions"
            ],
        },
    }


def certify_domain_packages(
    *,
    protocol_path: str | Path,
    layercake_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    repository_root = protocol_path.parent
    layercake_root = Path(layercake_root).resolve()
    output_path = Path(output_path).resolve()
    protocol = _load_protocol(protocol_path)
    lc = _import_layercake(layercake_root)
    before = _git_state(layercake_root)
    if before != {
        "commit": LAYERCAKE_COMMIT,
        "clean": True,
        "porcelain_sha256": hashlib.sha256(b"").hexdigest(),
    }:
        raise DomainConformanceError("sealed LayerCake checkout is not pristine")

    lineage = [
        _validate_domain_lineage(repository_root, specification)
        for specification in protocol["domains"]
    ]
    packages: list[dict[str, Any]] = []
    trust_store: dict[str, bytes] = {}
    prompts: dict[str, str] = {}
    expected_outputs: dict[str, bytes] = {}
    for specification in protocol["domains"]:
        domain = str(specification["domain"])
        package_path = _relative(repository_root, str(specification["package"]))
        public_path = _relative(repository_root, str(specification["public_key"]))
        if (
            _sha256_file(package_path) != specification["archive_sha256"]
            or _sha256_file(public_path) != specification["public_key_sha256"]
        ):
            raise DomainConformanceError(f"{domain} package identity mismatch")
        public = public_path.read_bytes()
        # Read the key identifier from the signed manifest using its exact key.
        from zipfile import ZipFile

        with ZipFile(package_path, "r") as archive:
            manifest = json.loads(archive.read("manifest.json"))
            member_names = sorted(archive.namelist())
        key_id = str(manifest["signature"]["key_id"])
        trust_store[key_id] = public
        loaded = lc["load_package"](
            package_path, trust_store={key_id: public}
        )
        if (
            not loaded.signed
            or loaded.manifest.abi_version != DIRECT_ABI_VERSION
            or loaded.manifest.abi_hash != DIRECT_ABI_SHA256
            or loaded.manifest.cake_type != "portable_decoder"
            or loaded.manifest.domains != (domain,)
            or member_names
            != ["manifest.json", "signature.json", "tensors.safetensors"]
        ):
            raise DomainConformanceError(f"{domain} package contract mismatch")
        validation_path = _relative(
            repository_root, str(specification["runs"][0]["validation"])
        )
        first = _read_json(validation_path)["observations"][0]
        prompts[domain] = str(first["prompt"])
        expected_outputs[domain] = str(first["layercake_output"]).encode("utf-8")
        packages.append(
            {
                "domain": domain,
                "cake_id": loaded.manifest.cake_id,
                "path": package_path,
                "archive_sha256": specification["archive_sha256"],
                "archive_bytes": package_path.stat().st_size,
                "package_hash": loaded.manifest.package_hash,
                "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
                "key_id": key_id,
                "member_names": member_names,
            }
        )

    receiver_evidence: list[dict[str, Any]] = []
    cpu_reference_outputs: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="abi-domain-receivers-") as temp:
        temporary_root = Path(temp)
        for receiver_index in range(3):
            host = lc["DirectCakeHost"](
                temporary_root / f"receiver-{receiver_index}",
                abi_version=DIRECT_ABI_VERSION,
                abi_hash=DIRECT_ABI_SHA256,
                trust_store=trust_store,
                device="cpu",
            )
            installs = [host.install(row["path"]) for row in packages]
            verifies = [
                host.installer.verify(str(row["cake_id"])) for row in packages
            ]
            outputs: dict[str, str] = {}
            for row in packages:
                domain = str(row["domain"])
                result = host.generate(row["cake_id"], prompts[domain])
                if result.output != expected_outputs[domain]:
                    raise DomainConformanceError(
                        f"{domain} package differs from validation output"
                    )
                outputs[domain] = result.output.decode("utf-8")
                cpu_reference_outputs.setdefault(domain, result.output)
                if cpu_reference_outputs[domain] != result.output:
                    raise DomainConformanceError(
                        f"{domain} receiver output is not byte-identical"
                    )
            lifecycle: list[dict[str, Any]] = []
            if receiver_index == 0:
                for row in packages:
                    removed = host.remove(row["cake_id"])
                    reinstalled = host.install(row["path"])
                    verified = host.installer.verify(row["cake_id"])
                    if (
                        verified["payload_hash"] != row["tensor_payload_hash"]
                        or verified["package_hash"] != row["package_hash"]
                    ):
                        raise DomainConformanceError(
                            "lifecycle changed package identity"
                        )
                    lifecycle.append(
                        {
                            "cake_id": row["cake_id"],
                            "remove_status": removed["status"],
                            "reinstall_status": reinstalled["status"],
                            "verified_payload_hash": verified["payload_hash"],
                        }
                    )
            receiver_evidence.append(
                {
                    "receiver_index": receiver_index,
                    "fresh_registry": True,
                    "installed_ids": list(host.installed_ids()),
                    "install_statuses": [row["status"] for row in installs],
                    "verify_statuses": [row["status"] for row in verifies],
                    "output_sha256": {
                        domain: hashlib.sha256(value.encode("utf-8")).hexdigest()
                        for domain, value in outputs.items()
                    },
                    "lifecycle": lifecycle,
                }
            )

        sparse_host = lc["DirectCakeHost"](
            temporary_root / "sparse-receiver",
            abi_version=DIRECT_ABI_VERSION,
            abi_hash=DIRECT_ABI_SHA256,
            trust_store=trust_store,
            device="cpu",
        )
        for row in packages:
            sparse_host.install(row["path"])
        sparse_host.reset_telemetry()
        selected = packages[0]
        sparse_host.generate(
            selected["cake_id"], prompts[str(selected["domain"])]
        )
        telemetry = sparse_host.telemetry()
        inactive = {
            row["cake_id"]: telemetry[row["cake_id"]]
            for row in packages[1:]
        }
        if any(
            value["module_load_calls"]
            or value["prefill_calls"]
            or value["decode_step_calls"]
            for value in inactive.values()
        ):
            raise DomainConformanceError("inactive cake executed")

        tampered_path = temporary_root / "tampered.cake"
        shutil.copyfile(packages[0]["path"], tampered_path)
        raw = bytearray(tampered_path.read_bytes())
        raw[len(raw) // 2] ^= 1
        tampered_path.write_bytes(raw)
        tamper_error = ""
        try:
            tamper_host = lc["DirectCakeHost"](
                temporary_root / "tamper-receiver",
                abi_version=DIRECT_ABI_VERSION,
                abi_hash=DIRECT_ABI_SHA256,
                trust_store=trust_store,
                device="cpu",
            )
            tamper_host.install(tampered_path)
        except Exception as exc:  # The security property is rejection itself.
            tamper_error = f"{type(exc).__name__}: {exc}"
        if not tamper_error:
            raise DomainConformanceError("tampered archive was accepted")

        if protocol["require_cuda"] and not torch.cuda.is_available():
            raise DomainConformanceError("protocol requires CUDA execution")
        cuda_identity: dict[str, dict[str, Any]] = {}
        if torch.cuda.is_available():
            cuda_host = lc["DirectCakeHost"](
                temporary_root / "cuda-receiver",
                abi_version=DIRECT_ABI_VERSION,
                abi_hash=DIRECT_ABI_SHA256,
                trust_store=trust_store,
                device="cuda",
            )
            for row in packages:
                cuda_host.install(row["path"])
                domain = str(row["domain"])
                output = cuda_host.generate(row["cake_id"], prompts[domain]).output
                if output != cpu_reference_outputs[domain]:
                    raise DomainConformanceError(
                        f"{domain} CPU/CUDA output differs"
                    )
                cuda_identity[domain] = {
                    "byte_identical": True,
                    "output_sha256": hashlib.sha256(output).hexdigest(),
                }

    after = _git_state(layercake_root)
    if after != before:
        raise DomainConformanceError("sealed LayerCake checkout changed")
    evidence: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "status": "PASS_VALIDATION_PACKAGE_GATES_FINAL_TEST_UNOPENED",
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "layercake_target": {
            "root": str(layercake_root),
            "before": before,
            "after": after,
            "unchanged": True,
        },
        "lineage": lineage,
        "packages": [
            {key: value for key, value in row.items() if key != "path"}
            for row in packages
        ],
        "receivers": receiver_evidence,
        "receiver_count": len(receiver_evidence),
        "receiver_output_byte_identity": True,
        "lifecycle_identity": True,
        "sparse_execution": {
            "selected_cake_id": selected["cake_id"],
            "telemetry": telemetry,
            "inactive_telemetry": inactive,
            "inactive_cakes_executed": False,
        },
        "tamper_rejection": {
            "status": "PASS",
            "error": tamper_error,
        },
        "cpu_cuda_identity": cuda_identity,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "non_executable_safetensors_only": True,
        "final_test_accessed": False,
        "claim_boundary": protocol["claim_boundary"],
        "remaining_blockers": [
            "unopened_final_test",
            "combined_english_plus_domain_phase2_performance_recertification",
            "mathematics_typed_runtime_abi_not_available_in_sealed_host",
            "independent_hostile_reproduction",
        ],
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    _write_json(output_path, evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = certify_domain_packages(
        protocol_path=args.protocol,
        layercake_root=args.layercake_root,
        output_path=args.output,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
