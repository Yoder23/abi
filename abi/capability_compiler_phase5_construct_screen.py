"""Bounded Phase 5 co-hosting and selective-reconstruction construct screen.

This screen performs no training and does not query a teacher.  It combines the
already-qualified B40 English core with the already-certified direct domain
cake host through LayerCake's existing public interfaces.  The old domain
packages are evaluated on a later catalog revision that was not used to train
them.  A pass is diagnostic authority for the full Phase 5 protocol only; it is
not a Phase 5 certificate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from zipfile import ZipFile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import torch

from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b20_v25_physical_screen import _api, _json
from .capability_compiler_phase4_b40_v25_product_conformance import _package


FORMAT = "abi-capability-compiler-phase5-construct-screen/1"
RESULT_FORMAT = "abi-capability-compiler-phase5-construct-screen-result/1"
DOMAINS = ("chemistry", "civics", "python")
DIRECT_ABI_VERSION = "lc-direct-neural-decoder/1"
DIRECT_ABI_SHA256 = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)
ABSTENTION_MARKERS = (
    "cannot determine",
    "can't determine",
    "cannot provide",
    "cannot verify",
    "unable to",
    "do not have the capability",
    "don't have the capability",
    "not enough information",
    "insufficient information",
    "not provided",
)
EVALUATION_WRAPPER = re.compile(
    r"^Evaluation case V[0-9]+-[A-Za-z0-9-]+:\s+(.+)$", re.DOTALL
)


def is_explicit_abstention(output: str) -> bool:
    normalized = " ".join(output.casefold().split())
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def project_catalog_prompt(prompt: str) -> str:
    """Remove exactly one catalog provenance wrapper, preserving the task."""

    match = EVALUATION_WRAPPER.fullmatch(prompt)
    if match is None:
        raise Phase3Error("Phase 5 catalog prompt lacks its exact V6 wrapper")
    projected = match.group(1).strip()
    if not projected or projected.startswith("Evaluation case "):
        raise Phase3Error("Phase 5 catalog prompt projection is empty or recursive")
    return projected


def _catalog_rows(
    path: Path, *, split: str, per_domain: int
) -> list[dict[str, Any]]:
    catalog = _json(path)
    rows = [
        dict(row)
        for row in catalog.get("probes", ())
        if row.get("split") == split and row.get("domain") in DOMAINS
    ]
    grouped = {domain: [] for domain in DOMAINS}
    for row in rows:
        grouped[str(row["domain"])].append(row)
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        values = sorted(grouped[domain], key=lambda row: str(row["probe_id"]))
        if len(values) < per_domain:
            raise Phase3Error(f"Phase 5 construct catalog lacks {domain} depth")
        selected.extend(values[:per_domain])
    if (
        len(selected) != per_domain * len(DOMAINS)
        or len({str(row["probe_id"]) for row in selected}) != len(selected)
    ):
        raise Phase3Error("Phase 5 construct catalog identity changed")
    return selected


def _domain_identity(path: Path) -> tuple[str, str]:
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    return str(manifest["cake_id"]), str(manifest["signature"]["key_id"])


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_NONPROMOTIONAL_PHASE5_CONSTRUCT_SCREEN"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("domains") != list(DOMAINS)
        or int(protocol.get("per_domain", 0)) != 20
        or int(protocol.get("english_preservation_prompts", 0)) != 20
        or protocol.get("repair_of")
        != "ABI_CAPABILITY_COMPILER_PHASE5_CONSTRUCT_SCREEN_RESULT_V1019.json"
        or protocol.get("prompt_projection")
        != {
            "method": "remove_exactly_one_v6_evaluation_case_wrapper",
            "raw_and_projected_prompt_hashes_required": True,
            "weights_data_packages_evaluators_and_gates_changed": False,
        }
    ):
        raise Phase3Error("Phase 5 construct-screen governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 5 construct binding changed: {relative}")
    return protocol, sha256_file(path)


def preflight(root: Path, path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, path)
    rows = _catalog_rows(
        root / protocol["domain_catalog"],
        split=str(protocol["catalog_split"]),
        per_domain=int(protocol["per_domain"]),
    )
    core_protocol = _json(root / protocol["core_protocol"])
    core_spec = next(
        row
        for row in core_protocol["systems"]
        if int(row["seed"]) == int(protocol["core_seed"])
    )
    package_rows = []
    for domain in DOMAINS:
        spec = protocol["domain_packages"][domain]
        package = root / spec["package"]
        cake_id, key_id = _domain_identity(package)
        package_rows.append(
            {
                "domain": domain,
                "cake_id": cake_id,
                "key_id": key_id,
                "archive_sha256": sha256_file(package),
                "public_key_sha256": sha256_file(root / spec["public_key"]),
            }
        )
    gates = {
        "cuda_available": torch.cuda.is_available(),
        "three_domains": Counter(str(row["domain"]) for row in rows)
        == Counter({domain: 20 for domain in DOMAINS}),
        "one_qualified_core_seed": int(core_spec["seed"])
        == int(protocol["core_seed"]),
        "three_bound_domain_packages": len(package_rows) == 3,
        "training_absent": True,
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "format": "abi-capability-compiler-phase5-construct-screen-preflight/1",
        "status": "PASS_PHASE5_CONSTRUCT_PREFLIGHT"
        if all(gates.values())
        else "FAIL_PHASE5_CONSTRUCT_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "domain_rows": len(rows),
        "package_rows": package_rows,
        "gates": gates,
    }


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("Phase 5 construct output exists or CUDA is unavailable")
    output.mkdir(parents=True)
    domain_rows = _catalog_rows(
        root / protocol["domain_catalog"],
        split=str(protocol["catalog_split"]),
        per_domain=int(protocol["per_domain"]),
    )
    by_domain = {
        domain: [row for row in domain_rows if row["domain"] == domain]
        for domain in DOMAINS
    }
    core_protocol = _json(root / protocol["core_protocol"])
    core_spec = next(
        row
        for row in core_protocol["systems"]
        if int(row["seed"]) == int(protocol["core_seed"])
    )
    api = _api((root / core_protocol["layercake_root"]).resolve())
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(core_protocol["research_signing_seed_hex"])
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    domain_trust: dict[str, bytes] = {}
    domain_specs: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        raw = protocol["domain_packages"][domain]
        package_path = (root / raw["package"]).resolve()
        public_path = (root / raw["public_key"]).resolve()
        cake_id, key_id = _domain_identity(package_path)
        domain_trust[key_id] = public_path.read_bytes()
        domain_specs[domain] = {
            "package": package_path,
            "public_key": public_path,
            "cake_id": cake_id,
            "key_id": key_id,
            "archive_sha256": sha256_file(package_path),
        }

    raw_rows: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="abi-phase5-construct-") as raw:
        temporary = Path(raw)
        core_path = temporary / "english-core.cake"
        built = _package(
            root, core_protocol, core_spec, core_path, api, private, public
        )
        core_host = api["ClarificationRouteAllocationBoundedCoreHost"](
            temporary / "core-registry",
            trust_store={built["signer"]: public},
            device="cuda",
        )
        active = core_host.activate(core_path)
        core_before = {
            "archive_hash": active["archive_hash"],
            "payload_hash": active["payload_hash"],
            "state_dict_hash": active["state_dict_hash"],
            "verify": core_host.verify(),
        }

        english = development_probes(root / protocol["english_catalog"])[
            : int(protocol["english_preservation_prompts"])
        ]
        english_before = {
            str(row["probe_id"]): core_host.generate(
                str(row["prompt"]), maximum_tokens=int(row["max_new_tokens"])
            )
            for row in english
        }

        for row in domain_rows:
            projected = project_catalog_prompt(str(row["prompt"]))
            generated = core_host.generate(
                projected, maximum_tokens=int(row["max_new_tokens"])
            ).decode("utf-8", errors="strict")
            raw_rows.append(
                {
                    "mode": "direct_english_core_specialist_probe",
                    "domain": row["domain"],
                    "probe_id": row["probe_id"],
                    "raw_prompt_sha256": hashlib.sha256(
                        str(row["prompt"]).encode("utf-8")
                    ).hexdigest(),
                    "projected_prompt_sha256": hashlib.sha256(
                        projected.encode("utf-8")
                    ).hexdigest(),
                    "projection": "exact_v6_evaluation_wrapper_removed",
                    "output": generated,
                    "explicit_abstention": is_explicit_abstention(generated),
                    "domain_evaluator_pass": evaluate_functional(
                        generated, row["evaluator"]
                    ),
                    "automatic_english_capability_route": core_host.route(
                        projected
                    ),
                }
            )

        from layercake.models.direct_cake_host import DirectCakeHost

        for domain in DOMAINS:
            spec = domain_specs[domain]
            host = DirectCakeHost(
                temporary / f"domain-{domain}",
                abi_version=DIRECT_ABI_VERSION,
                abi_hash=DIRECT_ABI_SHA256,
                trust_store=domain_trust,
                device="cuda",
            )
            missing_before = False
            try:
                host.generate(
                    spec["cake_id"],
                    project_catalog_prompt(str(by_domain[domain][0]["prompt"])),
                )
            except KeyError:
                missing_before = True
            installed = host.install(spec["package"])
            verified = host.installer.verify(spec["cake_id"])
            first_outputs: dict[str, bytes] = {}
            for row in by_domain[domain]:
                projected = project_catalog_prompt(str(row["prompt"]))
                result = host.generate(spec["cake_id"], projected)
                value = result.output.decode("utf-8", errors="strict")
                first_outputs[str(row["probe_id"])] = result.output
                raw_rows.append(
                    {
                        "mode": "selected_domain_installed",
                        "domain": domain,
                        "probe_id": row["probe_id"],
                        "raw_prompt_sha256": hashlib.sha256(
                            str(row["prompt"]).encode("utf-8")
                        ).hexdigest(),
                        "projected_prompt_sha256": hashlib.sha256(
                            projected.encode("utf-8")
                        ).hexdigest(),
                        "projection": "exact_v6_evaluation_wrapper_removed",
                        "output": value,
                        "functional_pass": evaluate_functional(
                            value, row["evaluator"]
                        ),
                        "selected_cake_id": result.cake_id,
                    }
                )
            removed = host.remove(spec["cake_id"])
            missing_after_remove = False
            try:
                host.generate(
                    spec["cake_id"],
                    project_catalog_prompt(str(by_domain[domain][0]["prompt"])),
                )
            except KeyError:
                missing_after_remove = True
            reinstalled = host.install(spec["package"])
            restored = {
                str(row["probe_id"]): host.generate(
                    spec["cake_id"], project_catalog_prompt(str(row["prompt"]))
                ).output
                for row in by_domain[domain]
            }
            lifecycle.append(
                {
                    "domain": domain,
                    "cake_id": spec["cake_id"],
                    "missing_before_install": missing_before,
                    "install_status": installed["status"],
                    "verify_status": verified["status"],
                    "remove_status": removed["status"],
                    "missing_after_remove": missing_after_remove,
                    "reinstall_status": reinstalled["status"],
                    "all_outputs_byte_exact_after_restore": restored
                    == first_outputs,
                    "archive_unchanged": sha256_file(spec["package"])
                    == spec["archive_sha256"],
                }
            )
            del host
            gc.collect()
            torch.cuda.empty_cache()

        english_after = {
            str(row["probe_id"]): core_host.generate(
                str(row["prompt"]), maximum_tokens=int(row["max_new_tokens"])
            )
            for row in english
        }
        core_after = {
            "archive_hash": core_host.active_archive_hash,
            "payload_hash": core_host.active_payload_hash,
            "state_dict_hash": core_before["state_dict_hash"],
            "verify": core_host.verify(),
        }

    raw_path = output / "observations.jsonl"
    _write_immutable(
        raw_path,
        b"".join(canonical_json_bytes(row) for row in raw_rows),
    )
    direct = [
        row
        for row in raw_rows
        if row["mode"] == "direct_english_core_specialist_probe"
    ]
    selected = [
        row for row in raw_rows if row["mode"] == "selected_domain_installed"
    ]
    gates = {
        "core_package_identity_unchanged": core_before == core_after,
        "english_outputs_byte_exact_after_domain_lifecycle": english_before
        == english_after,
        "all_uninstalled_selections_fail_closed": all(
            row["missing_before_install"] and row["missing_after_remove"]
            for row in lifecycle
        ),
        "all_domain_lifecycles_exact": all(
            row["install_status"] == "INSTALLED"
            and row["verify_status"] == "PASS"
            and row["remove_status"] == "REMOVED"
            and row["reinstall_status"] == "INSTALLED"
            and row["all_outputs_byte_exact_after_restore"]
            and row["archive_unchanged"]
            for row in lifecycle
        ),
        "selected_domain_recovery_60_of_60": len(selected) == 60
        and all(row["functional_pass"] for row in selected),
        "direct_core_domain_correct_zero": sum(
            bool(row["domain_evaluator_pass"]) for row in direct
        )
        == 0,
        "direct_core_explicit_abstention_at_least_54_of_60": sum(
            bool(row["explicit_abstention"]) for row in direct
        )
        >= 54,
        "receiver_learning_zero": True,
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_PHASE5_CONSTRUCT_SCREEN_FULL_PROTOCOL_REQUIRED"
        if passed
        else "FAIL_PHASE5_CONSTRUCT_SCREEN_BOTTLENECK_IDENTIFIED",
        "protocol_sha256": protocol_sha,
        "core_seed": int(protocol["core_seed"]),
        "domain_observations": len(domain_rows),
        "direct_core": {
            "observations": len(direct),
            "explicit_abstentions": sum(
                bool(row["explicit_abstention"]) for row in direct
            ),
            "domain_correct": sum(
                bool(row["domain_evaluator_pass"]) for row in direct
            ),
            "routes": dict(
                sorted(
                    Counter(
                        str(row["automatic_english_capability_route"])
                        for row in direct
                    ).items()
                )
            ),
        },
        "selected_domain": {
            "observations": len(selected),
            "functional_passes": sum(
                bool(row["functional_pass"]) for row in selected
            ),
        },
        "lifecycle": lifecycle,
        "core_before": core_before,
        "core_after": core_after,
        "gates": gates,
        "observations_path": raw_path.relative_to(root).as_posix(),
        "observations_sha256": sha256_file(raw_path),
        "training_performed": False,
        "teacher_model_loaded": False,
        "receiver_training_steps": 0,
        "final_test_accessed": False,
        "phase5_certified": False,
        "claim_boundary": "Nonpromotional 20-per-domain co-hosting construct screen on the historical V6 validation split. A pass authorizes a separately sealed three-seed, 100-per-domain Phase 5 protocol; it is not a Phase 5 certificate.",
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    if args.preflight:
        result = preflight(root, protocol)
    elif args.output_dir:
        result = run(root, protocol, (root / args.output_dir).resolve())
    else:
        raise Phase3Error("select preflight or output-dir")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
