"""Fail-closed verifier for the ABI host-independence release layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_bytes, sha256_bytes
from .verify_release import verify as verify_v2

HOSTS = ("layercake", "qwen2", "pythia")
CAPABILITIES = ("english", "python", "chemistry", "civics")
REQUIRED_DIRECTORIES = (
    "frozen_v1",
    "structural_analysis",
    "canonical_abi",
    "family_a",
    "family_b",
    "family_c",
    "family_d",
    "synthesis",
    "host_certification",
    "capability_matrix",
    "semantic_retention",
    "mathematical_portability",
    "isolation",
    "economics",
    "performance",
    "hostile_audit",
    "external_validation",
)
EXPECTED_ADAPTERS = {
    "layercake": "d1f3a9d67581032473055711773f784e3b0a028fb68fab58c3a9a63b94317f04",
    "qwen2": "b13a75b82acf6df412c2569edaf36654a105b8c565164236fbeb5813b630291f",
    "pythia": "df3598b6ae28f43a0ccff34aef3e59d99e415a5a3cfee15619fefa1dc74ceafa",
}
EXPECTED_PACKAGES = {
    "english": "acb787b3ffa0153c57d88cd37ba81c3f00b370d4ca4937e659cd4c775851f25d",
    "python": "f1defaef2771ced336a332572a2d2f0e1e542399c877d182c48a6cd2e199231d",
    "chemistry": "f9c9b2668fda5ef6b92844c1b7097fbdf8ff0daaae51f5b86f72d4a49000abeb",
    "civics": "634ce66958859ec36dc1fbdf5ef34d6d2a9949d10cf2348a68c245d8c325d604",
}
EXPECTED_TENSORS = {
    "english": "18ad787696cf8737578253035ad84b7e0145d995f028c6392ba338b6117d5aba",
    "python": "ac58835feee49441aef5fdfddd315523458ac95c53be8f41b8f309a086383242",
    "chemistry": "b70dec838c5d3d119333e0f12ccc26392fad56b97cb4841045d0780cfb5f96a2",
    "civics": "531f4cf14ae8823314cec77c919eba27fe136ba57ea7f4467ac53a69276a38dd",
}


class HostIndependenceVerificationError(RuntimeError):
    """Raised when the release layer is incomplete or inconsistent."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HostIndependenceVerificationError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    if not all(isinstance(value, dict) for value in values):
        raise HostIndependenceVerificationError(f"expected JSONL objects: {path}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _verify_bindings(root: Path, certificate: Mapping[str, Any]) -> None:
    bindings = certificate.get("evidence_bindings")
    if not isinstance(bindings, dict) or len(bindings) < 10:
        raise HostIndependenceVerificationError("release evidence bindings missing")
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise HostIndependenceVerificationError(f"invalid binding: {name}")
        relative = Path(str(binding.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise HostIndependenceVerificationError(f"unsafe binding path: {name}")
        path = root / relative
        if not path.is_file() or _sha256(path) != binding.get("sha256"):
            raise HostIndependenceVerificationError(f"evidence changed: {name}")


def _verify_selected_only(root: Path) -> None:
    path = (
        root
        / "results/abi_capability_compiler_phase6_composition/run_v1032/"
        "seed104729/observations.jsonl"
    )
    rows = _jsonl(path)
    selected_rows = [row for row in rows if row.get("selected_only_execution") is True]
    if len(rows) != 500 or len(selected_rows) != 300:
        raise HostIndependenceVerificationError("selected-only depth changed")
    for row in selected_rows:
        selected = row.get("selected")
        telemetry = row.get("telemetry_delta")
        if row.get("selected_only_execution") is not True or not isinstance(
            selected, list
        ):
            raise HostIndependenceVerificationError("selected-only declaration changed")
        if len(selected) != 1 or not isinstance(telemetry, dict):
            raise HostIndependenceVerificationError("selected capability is ambiguous")
        for capability, counters in telemetry.items():
            active = capability == selected[0]
            if not isinstance(counters, dict):
                raise HostIndependenceVerificationError("invalid execution telemetry")
            calls = int(counters.get("prefill_calls", 0)) + int(
                counters.get("decode_step_calls", 0)
            )
            if active != (calls > 0):
                raise HostIndependenceVerificationError(
                    "non-selected capability consumed active computation"
                )


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    release_root = root / "results/abi_host_independence"
    for name in REQUIRED_DIRECTORIES:
        if not (release_root / name).is_dir():
            raise HostIndependenceVerificationError(f"required evidence directory missing: {name}")

    v2_certificate = verify_v2(root, check_existing=True)
    if not all(v2_certificate.get("technical_gates", {}).values()):
        raise HostIndependenceVerificationError("underlying ABI V2 gate failed")

    certificate = _json(release_root / "release_certificate.json")
    if (
        certificate.get("status")
        != "TECHNICALLY_PROVEN_EXTERNAL_VALIDATION_PENDING"
        or certificate.get("technical_moonshot") != "ABI TECHNICAL MOONSHOT: PROVEN"
        or certificate.get("winning_family")
        != "FAMILY_A_OBSERVABLE_CANONICAL_STATE"
        or certificate.get("evidence_sha256") != _evidence_hash(certificate)
        or not all(certificate.get("technical_gates", {}).values())
        or any(certificate.get("external_gates", {}).values())
    ):
        raise HostIndependenceVerificationError("release certificate gate failed")
    if any(certificate.get("claim_flags", {}).values()):
        raise HostIndependenceVerificationError("unsupported claim enabled")
    _verify_bindings(root, certificate)

    frozen = _json(release_root / "frozen_capability_manifest.json")
    if frozen.get("mutation_allowed") is not False:
        raise HostIndependenceVerificationError("frozen package mutation enabled")
    if sum(
        int(frozen["capabilities"][name]["source_successes"])
        for name in CAPABILITIES
    ) != 1681:
        raise HostIndependenceVerificationError("source-success manifest changed")
    locks = _json(root / frozen["source_success_lock"]["path"])
    for capability in CAPABILITIES:
        value = frozen["capabilities"][capability]
        lock = (
            locks["english"]
            if capability == "english"
            else locks["domains"][capability]
        )
        if (
            value.get("archive_sha256") != EXPECTED_PACKAGES[capability]
            or value.get("tensor_sha256") != EXPECTED_TENSORS[capability]
            or value.get("source_success_ids_sha256")
            != sha256_bytes(canonical_json_bytes(lock["successful_task_ids"]))
        ):
            raise HostIndependenceVerificationError(
                f"frozen capability identity changed: {capability}"
            )

    ownership = _json(release_root / "structural_ownership_map.json")
    mismatches = ownership.get("mismatches")
    allowed_owners = {"HOST-OWNED", "CANONICAL-ABI-OWNED", "CAPABILITY-OWNED"}
    if not isinstance(mismatches, list) or len(mismatches) != 15:
        raise HostIndependenceVerificationError("ownership map is incomplete")
    classifications = [value.get("classification") for value in mismatches]
    if len(set(classifications)) != 15 or any(
        value.get("owner") not in allowed_owners for value in mismatches
    ):
        raise HostIndependenceVerificationError("ownership is ambiguous")

    family_a = _json(release_root / "family_a/decision.json")
    if (
        family_a.get("status") != "PROMOTED_TECHNICAL_PASS"
        or family_a.get("matrix_cells_passed") != 12
        or family_a.get("instrumentation_repair_changed_semantic_outputs") is not False
    ):
        raise HostIndependenceVerificationError("Family A promotion changed")
    for family in ("family_b", "family_c", "family_d"):
        decision = _json(release_root / family / "decision.json")
        if decision.get("status") != "NOT_RUN_AFTER_FAMILY_A_TECHNICAL_PASS":
            raise HostIndependenceVerificationError(f"invalid ladder state: {family}")
    if (
        _json(release_root / "synthesis/decision.json").get("status")
        != "NOT_AUTHORIZED_AFTER_FAMILY_A_TECHNICAL_PASS"
    ):
        raise HostIndependenceVerificationError("invalid synthesis state")

    host_certification = _json(release_root / "host_certification/manifest.json")
    if host_certification.get("status") != "PASS_THREE_CAPABILITY_BLIND_HOST_CERTIFICATIONS":
        raise HostIndependenceVerificationError("host certification summary changed")
    if host_certification.get("cryptographic_ordering", {}).get(
        "proof_kind"
    ) != "HASH_BOUND_PREREGISTRATION":
        raise HostIndependenceVerificationError("certification ordering is unbound")
    for host in HOSTS:
        if host_certification["hosts"][host]["adapter_sha256"] != EXPECTED_ADAPTERS[host]:
            raise HostIndependenceVerificationError(f"adapter changed: {host}")

    economics = _json(release_root / "economics/reuse_economics.json")
    alternatives = economics.get("matched_alternative_comparison", {})
    if (
        economics.get("installation_operation")
        != "VERIFY_PLUS_LOAD_WITH_ZERO_TRAINING_AND_ZERO_CALIBRATION"
        or alternatives.get("quantitative_pareto_superiority_claimed") is not False
    ):
        raise HostIndependenceVerificationError("reuse economics boundary changed")

    task_state = _json(release_root / "task_state.json")
    if (
        task_state.get("technical_declaration") != "ABI TECHNICAL MOONSHOT: PROVEN"
        or task_state.get("current_action")
        != "HAND_OFF_EXTERNAL_VALIDATION; NO FURTHER ARCHITECTURE COMPUTE"
    ):
        raise HostIndependenceVerificationError("campaign task state changed")

    ledger = _jsonl(release_root / "experiment_ledger.jsonl")
    required_fields = {
        "architecture_family",
        "hypothesis",
        "commit",
        "config",
        "host",
        "adapter_hash",
        "capability_hashes",
        "certification_data_hashes",
        "structural_compatibility",
        "semantic_results",
        "mathematical_results",
        "isolation",
        "adapter_overhead",
        "failure_category",
        "next_action",
    }
    if len(ledger) != 1 or not required_fields.issubset(ledger[0]):
        raise HostIndependenceVerificationError("experiment ledger is incomplete")
    _verify_selected_only(root)
    return certificate


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-existing", action="store_true")
    parser.parse_args(argv)
    result = verify(Path.cwd())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
