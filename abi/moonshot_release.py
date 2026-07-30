"""Verify and run the bounded ABI-to-LayerCake Moonshot reference release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from .layercake_product_host import LayerCakeProductHost


CERTIFICATE_FORMAT = "abi-layercake-moonshot-release-certificate/2"
CERTIFICATE_FORMATS = {
    "abi-layercake-moonshot-release-certificate/1",
    CERTIFICATE_FORMAT,
}
FINAL_FORMAT = "abi-layercake-moonshot-final-test-evidence/2"
ROOT = Path(__file__).resolve().parents[1]
PRODUCT_DECISION = "ABI_POSTCERT_GENERALIZATION_AUDIT_DECISION.json"


class MoonshotReleaseError(RuntimeError):
    """Raised when a release certificate or referenced component is stale."""


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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MoonshotReleaseError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MoonshotReleaseError(f"JSON must be an object: {path}")
    return value


def _claim_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    claimed = payload.pop("evidence_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha(payload):
        raise MoonshotReleaseError("evidence claim hash mismatch")
    return claimed


def _relative_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MoonshotReleaseError(
            f"release path escapes repository: {relative}"
        ) from exc
    if not path.is_file():
        raise MoonshotReleaseError(f"release file is missing: {relative}")
    return path


def _evidence_reference(
    root: Path,
    relative: str,
    *,
    status: str,
    format_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _relative_file(root, relative)
    value = _read(path)
    if (
        value.get("status") != status
        or (
            format_name is not None
            and value.get("format", value.get("schema_version"))
            != format_name
        )
    ):
        raise MoonshotReleaseError(f"release gate failed: {relative}")
    claimed = _claim_hash(value)
    return value, {
        "path": relative,
        "file_sha256": _sha256_file(path),
        "evidence_sha256": claimed,
        "status": status,
    }


def _git_state(path: Path) -> dict[str, Any]:
    git = ["git", "-c", f"safe.directory={path}"]
    commit = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        [*git, "status", "--porcelain"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "clean": not bool(porcelain.strip()),
        "porcelain_sha256": hashlib.sha256(
            porcelain.encode("utf-8")
        ).hexdigest(),
    }


def build_certificate(
    *,
    output_path: str | Path,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Build the fail-closed top-level certificate from immutable evidence."""

    root = Path(root).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise MoonshotReleaseError(
            f"release certificate is immutable: {output_path}"
        )
    final, final_ref = _evidence_reference(
        root,
        "results/abi_moonshot/final_test_v5/layercake-final-test.json",
        status="PASS",
        format_name=FINAL_FORMAT,
    )
    repair, repair_ref = _evidence_reference(
        root,
        "results/abi_moonshot/rewriting_v5/v47-repair-certificate.json",
        status="PASS",
        format_name="abi-layercake-rewriting-v5-repair-certificate/1",
    )
    hostile, hostile_ref = _evidence_reference(
        root,
        "results/abi_moonshot/rewriting_v5/v47-hostile-reproduction.json",
        status="PASS",
        format_name="abi-layercake-hostile-reproduction-evidence/1",
    )
    packages, packages_ref = _evidence_reference(
        root,
        "results/abi_moonshot/domain_cakes/"
        "package-certification-validation.json",
        status="PASS_VALIDATION_PACKAGE_GATES_FINAL_TEST_UNOPENED",
        format_name="abi-layercake-domain-package-certification-evidence/1",
    )
    artifact_relative = (
        "results/abi_moonshot/native_hosts/"
        "full-budget3-longform-sequence-repaired-seed9824-"
        "v47-v5-rewriting-repair-identity-fixed"
    )
    artifact = (root / artifact_relative).resolve()
    metadata_path = artifact / "metadata.json"
    metadata = _read(metadata_path)
    component_names = (
        "metadata.json",
        metadata["runtime"]["graph"],
        metadata["tokenizer"]["path"],
        metadata["runtime"]["output_vocabulary"]["path"],
        metadata["symbolic_surface"]["path"],
    )
    components = {
        name: {
            "bytes": (artifact / name).stat().st_size,
            "sha256": _sha256_file(artifact / name),
        }
        for name in component_names
    }

    package_rows: dict[str, Any] = {}
    package_by_domain = {
        value["domain"]: value for value in packages["packages"]
    }
    protocol = _read(root / "ABI_MOONSHOT_V5_FINAL_TEST_PROTOCOL.json")
    source_specs = {
        source["id"]: source for source in protocol["sources"]
    }
    for domain, specification in protocol["locked_candidate"][
        "packages"
    ].items():
        package_path = _relative_file(root, specification["path"])
        key_path = _relative_file(root, specification["public_key"])
        qualified = package_by_domain[domain]
        if _sha256_file(package_path) != specification["archive_sha256"]:
            raise MoonshotReleaseError(
                f"qualified package changed: {domain}"
            )
        package_rows[domain] = {
            "cake_id": specification["cake_id"],
            "path": specification["path"],
            "archive_sha256": specification["archive_sha256"],
            "archive_bytes": package_path.stat().st_size,
            "public_key": specification["public_key"],
            "public_key_sha256": _sha256_file(key_path),
            "minimum_tested_passing_budget_index": next(
                value["minimum_tested_passing_budget_index"]
                for value in packages["lineage"]
                if value["domain"] == domain
            ),
            "teacher_tokens": next(
                value["teacher_tokens"]
                for value in packages["lineage"]
                if value["domain"] == domain
            ),
            "parameter_count": next(
                value["parameter_count"]
                for value in packages["lineage"]
                if value["domain"] == domain
            ),
            "tensor_payload_hash": qualified["tensor_payload_hash"],
        }

    budget_path = _relative_file(
        root, "ENGLISH_BUDGET_FRONTIER_DECISION.json"
    )
    budget = _read(budget_path)
    if (
        budget.get("status")
        != "PASS_MINIMUM_AMONG_PREREGISTERED_TESTED_BUDGETS"
        or budget.get("selected_budget", {}).get("budget_index") != 3
        or budget.get("largest_lower_failing_budget", {}).get(
            "budget_index"
        )
        != 2
        or budget.get("final_test_accessed") is not False
    ):
        raise MoonshotReleaseError("English budget frontier is invalid")

    pytest_path = _relative_file(
        root,
        "results/abi_moonshot/final_test_v5/full-pytest-release.xml",
    )
    xml_root = ET.parse(pytest_path).getroot()
    suite = (
        xml_root.find("testsuite")
        if xml_root.tag == "testsuites"
        else xml_root
    )
    if suite is None:
        raise MoonshotReleaseError("pytest evidence has no test suite")
    test_count = int(suite.attrib.get("tests", 0))
    test_failures = int(suite.attrib.get("failures", 0))
    test_errors = int(suite.attrib.get("errors", 0))

    layercake_root = (root / "../layercake_release").resolve()
    layercake_state = _git_state(layercake_root)
    expected_layercake_commit = (
        "04cf2927a16fba686cd640e18a78708e5658bbda"
    )
    final_metrics = final["capability_metrics"]
    english = sorted(
        capability
        for capability, metrics in final_metrics.items()
        if metrics["destination_scope"] == "english_core"
    )
    domain_capabilities = {
        metrics["domain"]: capability
        for capability, metrics in final_metrics.items()
        if metrics["destination_scope"] == "domain_cake"
    }
    implementation_paths = (
        "abi/layercake_host_runtime.py",
        "abi/symbolic_runtime.py",
        "abi/layercake_product_host.py",
        "abi/layercake_domain_worker.py",
        "abi/layercake_domains.py",
    )
    implementation = {
        relative: {
            "bytes": _relative_file(root, relative).stat().st_size,
            "sha256": _sha256_file(_relative_file(root, relative)),
        }
        for relative in implementation_paths
    }
    gates = {
        "final_test_1700_of_1700": (
            final["observation_count"] == 1700
            and final["layercake_passes"] == 1700
            and final["source_passing_regressions"] == 0
            and all(final["gates"].values())
        ),
        "repair_certificate_pass": all(repair["gates"].values()),
        "hostile_reproduction_pass": all(hostile["gates"].values()),
        "qualified_domain_package_count_three": (
            len(package_rows) == 3
        ),
        "candidate_identity_exact": (
            _sha256_file(metadata_path)
            == repair["candidate"]["metadata_file_sha256"]
            and metadata["runtime"]["graph_sha256"]
            == repair["candidate"]["runtime_graph_sha256"]
            and metadata["symbolic_surface"]["sha256"]
            == repair["candidate"]["symbolic_surface_sha256"]
        ),
        "teacher_absent": (
            metadata["host"]["teacher_present_at_inference"] is False
        ),
        "source_transformer_blocks_absent": (
            metadata["host"]["source_transformer_blocks_retained"] == 0
        ),
        "full_test_suite_pass": (
            test_count >= 157
            and test_failures == 0
            and test_errors == 0
        ),
        "sealed_layercake_pristine": (
            layercake_state["commit"] == expected_layercake_commit
            and layercake_state["clean"] is True
        ),
    }
    certificate: dict[str, Any] = {
        "format": CERTIFICATE_FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "release_scope": "BOUNDED_REFERENCE_RELEASE",
        "candidate": {
            "artifact": artifact_relative,
            "metadata_evidence_sha256": metadata["evidence_sha256"],
            "host_manifest_sha256": metadata["host"][
                "deployment_manifest_sha256"
            ],
            "runtime_graph_sha256": metadata["runtime"]["graph_sha256"],
            "active_runtime_model_bytes": repair["performance"][
                "headline"
            ]["candidate_active_runtime_model_bytes"],
            "deployed_artifact_bytes": sum(
                row["bytes"] for row in components.values()
            ),
            "components": components,
            "teacher_present_at_inference": False,
            "source_transformer_blocks_retained": 0,
        },
        "implementation": implementation,
        "acquisition": {
            "source_models": [
                {
                    "model": source_specs[source["source_id"]]["model"],
                    "revision": source_specs[source["source_id"]][
                        "revision"
                    ],
                    "source_manifest_sha256": source[
                        "source_manifest_sha256"
                    ],
                    "final_observations": source["observation_count"],
                    "teacher_tokens": source["accounting"][
                        "teacher_tokens"
                    ],
                }
                for source in final["source_final_test_evidence"]
            ],
            "english_minimum": {
                "claim": budget["claim"],
                **budget["selected_budget"],
                "largest_lower_failure": budget[
                    "largest_lower_failing_budget"
                ],
                "decision_path": "ENGLISH_BUDGET_FRONTIER_DECISION.json",
                "decision_file_sha256": _sha256_file(budget_path),
            },
            "teacher_outputs_deployed": False,
            "logits_deployed": False,
            "hidden_activations_deployed": False,
            "source_parameters_deployed": 0,
        },
        "capabilities": {
            "english_core": english,
            "qualified_domains": domain_capabilities,
            "domain_packages": package_rows,
            "closed_failed_domains": ["mathematics"],
        },
        "final_test": {
            **final_ref,
            "layercake_passes": final["layercake_passes"],
            "source_passes": final["source_passes"],
            "source_passing_regressions": final[
                "source_passing_regressions"
            ],
            "capability_count": len(final_metrics),
        },
        "performance": repair["performance"],
        "evidence": {
            "repair_certificate": repair_ref,
            "hostile_reproduction": hostile_ref,
            "domain_package_certification": packages_ref,
            "full_pytest": {
                "path": str(pytest_path.relative_to(root)),
                "file_sha256": _sha256_file(pytest_path),
                "tests": test_count,
                "failures": test_failures,
                "errors": test_errors,
            },
        },
        "sealed_layercake": {
            "relative_root": "../layercake_release",
            **layercake_state,
        },
        "gates": gates,
        "historical_negative_evidence_preserved": (
            repair["negative_evidence_preserved"]
        ),
        "claim_boundary": (
            "PASS means bounded semantic dominance on the locked disjoint "
            "v2-v5 suites, exact artifact/package identity, three fresh host "
            "initializations, hostile in-repository reproduction, and the "
            "reported CPU gates. It does not mean universal semantic or "
            "mathematical identity, exhaustive extraction of arbitrary model "
            "knowledge, qualification of untested domains, organizationally "
            "independent audit, or proof of a global information minimum."
        ),
    }
    certificate["evidence_sha256"] = _canonical_sha(certificate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return certificate


def verify_certificate(
    certificate_path: str | Path,
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Verify the top-level claim hash and every deployed release component."""

    root = Path(root).resolve()
    certificate_path = Path(certificate_path).resolve()
    certificate = _read(certificate_path)
    if (
        certificate.get("format") not in CERTIFICATE_FORMATS
        or certificate.get("status") != "PASS"
        or not all(certificate.get("gates", {}).values())
    ):
        raise MoonshotReleaseError("release certificate is not passing")
    _claim_hash(certificate)
    verified = []
    for reference in certificate["evidence"].values():
        path = _relative_file(root, reference["path"])
        if _sha256_file(path) != reference["file_sha256"]:
            raise MoonshotReleaseError(
                f"release evidence changed: {reference['path']}"
            )
        verified.append(reference["path"])
    final_reference = certificate["final_test"]
    final_path = _relative_file(root, final_reference["path"])
    final = _read(final_path)
    if (
        _sha256_file(final_path) != final_reference["file_sha256"]
        or _claim_hash(final) != final_reference["evidence_sha256"]
        or final.get("status") != "PASS"
    ):
        raise MoonshotReleaseError("final-test evidence changed")
    artifact = (root / certificate["candidate"]["artifact"]).resolve()
    for name, specification in certificate["candidate"][
        "components"
    ].items():
        path = artifact / name
        if (
            path.stat().st_size != specification["bytes"]
            or _sha256_file(path) != specification["sha256"]
        ):
            raise MoonshotReleaseError(
                f"candidate component changed: {name}"
            )
    for domain, package in certificate["capabilities"][
        "domain_packages"
    ].items():
        if (
            _sha256_file(_relative_file(root, package["path"]))
            != package["archive_sha256"]
            or _sha256_file(_relative_file(root, package["public_key"]))
            != package["public_key_sha256"]
        ):
            raise MoonshotReleaseError(
                f"domain package changed: {domain}"
            )
    for relative, specification in certificate.get(
        "implementation", {}
    ).items():
        path = _relative_file(root, relative)
        if (
            path.stat().st_size != specification["bytes"]
            or _sha256_file(path) != specification["sha256"]
        ):
            raise MoonshotReleaseError(
                f"release implementation changed: {relative}"
            )
    layercake_state = _git_state(
        (root / certificate["sealed_layercake"]["relative_root"]).resolve()
    )
    if (
        layercake_state["commit"]
        != certificate["sealed_layercake"]["commit"]
        or not layercake_state["clean"]
    ):
        raise MoonshotReleaseError("sealed LayerCake checkout changed")
    decision_path = root / PRODUCT_DECISION
    product_status = "NOT_AUDITED"
    broad_product_complete = False
    if decision_path.is_file():
        decision = _read(decision_path)
        _claim_hash(decision)
        product_status = str(decision.get("status"))
        broad_product_complete = bool(
            decision.get("effect_on_prior_evidence", {}).get(
                "broad_product_moonshot_complete"
            )
        )
        for side in ("locked_layercake", "frozen_source"):
            reference = decision[side]
            if (
                _sha256_file(_relative_file(root, reference["path"]))
                != reference["file_sha256"]
            ):
                raise MoonshotReleaseError(
                    f"post-certificate audit changed: {side}"
                )
    return {
        "status": "PASS_BOUNDED_CERTIFICATE",
        "broad_product_status": product_status,
        "broad_product_moonshot_complete": broad_product_complete,
        "certificate_evidence_sha256": certificate["evidence_sha256"],
        "verified_evidence_files": len(verified) + 1,
        "verified_candidate_components": len(
            certificate["candidate"]["components"]
        ),
        "verified_domain_packages": len(
            certificate["capabilities"]["domain_packages"]
        ),
        "verified_implementation_files": len(
            certificate.get("implementation", {})
        ),
        "final_test_layercake_passes": final["layercake_passes"],
        "final_test_source_passes": final["source_passes"],
        "source_passing_regressions": final[
            "source_passing_regressions"
        ],
    }


def generate(
    *,
    certificate_path: str | Path,
    prompt: str,
    domain: str | None,
    device: str,
    threads: int,
    max_new_tokens: int,
    allow_bounded_reference: bool = False,
) -> dict[str, Any]:
    """Run one verified teacher-free English or explicitly selected domain call."""

    verification = verify_certificate(certificate_path)
    if (
        not verification["broad_product_moonshot_complete"]
        and not allow_bounded_reference
    ):
        raise MoonshotReleaseError(
            "general-purpose generation is blocked by the post-certificate "
            "English generalization failure; pass --allow-bounded-reference "
            "only to reproduce the explicitly bounded research result"
        )
    certificate = _read(Path(certificate_path).resolve())
    artifact = (ROOT / certificate["candidate"]["artifact"]).resolve()
    layercake_root = (
        ROOT / certificate["sealed_layercake"]["relative_root"]
    ).resolve()
    with tempfile.TemporaryDirectory(prefix="abi-release-host-") as registry:
        with LayerCakeProductHost(
            english_artifact=artifact,
            layercake_root=layercake_root,
            registry_root=registry,
            threads=threads,
        ) as host:
            cake_id = None
            if domain is not None:
                try:
                    package = certificate["capabilities"][
                        "domain_packages"
                    ][domain]
                except KeyError as exc:
                    raise MoonshotReleaseError(
                        f"domain is not certified: {domain}"
                    ) from exc
                host.install(
                    ROOT / package["path"],
                    ROOT / package["public_key"],
                )
                cake_id = package["cake_id"]
            result = host.generate(
                prompt,
                cake_id=cake_id,
                domain_device=device,
                max_new_tokens=max_new_tokens,
            )
    return {
        "status": "COMPLETED_BOUNDED_UNSCORED",
        "engine": result.engine,
        "domain": domain,
        "cake_id": result.cake_id,
        "output": result.output,
        "output_sha256": result.output_sha256,
        "teacher_present_at_inference": False,
        "quality_claim": (
            "No quality pass is implied by an ad hoc generation call. "
            "The broad product gate remains controlled by "
            f"{PRODUCT_DECISION}."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-certificate")
    build.add_argument(
        "--output", default="ABI_MOONSHOT_CERTIFICATE_V2.json"
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument(
        "--certificate", default="ABI_MOONSHOT_CERTIFICATE_V2.json"
    )
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument(
        "--certificate", default="ABI_MOONSHOT_CERTIFICATE_V2.json"
    )
    run = subparsers.add_parser("generate")
    run.add_argument(
        "--certificate", default="ABI_MOONSHOT_CERTIFICATE_V2.json"
    )
    run.add_argument("--prompt", required=True)
    run.add_argument("--domain", choices=("chemistry", "civics", "python"))
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--threads", type=int, default=16)
    run.add_argument("--max-new-tokens", type=int, default=96)
    run.add_argument(
        "--allow-bounded-reference",
        action="store_true",
        help=(
            "reproduce the bounded research artifact despite the controlling "
            "generalization failure"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-certificate":
            result = build_certificate(output_path=args.output)
        elif args.command == "verify":
            result = verify_certificate(args.certificate)
        elif args.command == "inspect":
            verification = verify_certificate(args.certificate)
            certificate = _read(Path(args.certificate).resolve())
            result = {
                "status": certificate["status"],
                "release_scope": certificate["release_scope"],
                "candidate": certificate["candidate"],
                "capabilities": certificate["capabilities"],
                "final_test": certificate["final_test"],
                "performance": certificate["performance"],
                "claim_boundary": certificate["claim_boundary"],
                "product_decision": {
                    "status": verification["broad_product_status"],
                    "broad_product_moonshot_complete": verification[
                        "broad_product_moonshot_complete"
                    ],
                },
            }
        else:
            result = generate(
                certificate_path=args.certificate,
                prompt=args.prompt,
                domain=args.domain,
                device=args.device,
                threads=args.threads,
                max_new_tokens=args.max_new_tokens,
                allow_bounded_reference=args.allow_bounded_reference,
            )
    except MoonshotReleaseError as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "error": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
