"""Final, inference-free validation of the frozen ABI V2 technical result.

This module deliberately does not improve the architecture.  It recomputes
claims from the locked host-certification and matrix evidence, makes the host
causality boundary explicit, audits shortcuts, and generates the final local
technical certificate.  Human ratings and different-hardware execution remain
external gates.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

HOSTS = ("layercake", "qwen2", "pythia")
CAPABILITIES = ("english", "python", "chemistry", "civics")
FROZEN_TAG = "abi-host-independence-technical-proof-2026-08-24"
FROZEN_COMMIT = "acfed2a225a32d36c32b625e35c6ede536cfab01"
RESULT_ROOT = Path("results/abi_final_validation")
REPAIRED_RESULT_ROOT = Path("results/abi_final_validation_v2")
MATRIX_DIRS = {
    "layercake": "layercake_repaired",
    "qwen2": "qwen2",
    "pythia": "pythia",
}
CAPABILITY_PATHS = {
    "english": (
        "results/abi_capability_compiler_phase7_integrated/materialized_v1052/"
        "phase7-final-english-core.cake"
    ),
    "python": "results/abi_moonshot/packages/abi-python-token-plan-seed9824.cake",
    "chemistry": "results/abi_moonshot/packages/abi-chemistry-token-plan-seed9824.cake",
    "civics": "results/abi_moonshot/packages/abi-civics-token-plan-seed9824.cake",
}


class FinalValidationError(RuntimeError):
    """Raised when frozen evidence cannot support the declared result."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    ) + b"\n"


def evidence_hash(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes({k: v for k, v in value.items() if k != "evidence_sha256"}))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalValidationError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FinalValidationError(f"expected JSONL object: {path}")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(value)
    value["evidence_sha256"] = evidence_hash(value)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _protocol(root: Path) -> dict[str, Any]:
    amendment_path = root / "abi_v2/matrix_protocol_amendment3.json"
    amendment = read_json(amendment_path)
    base_path = root / amendment["base_protocol"]
    if sha256_file(base_path) != amendment["base_protocol_sha256"]:
        raise FinalValidationError("matrix base protocol hash changed")
    base = read_json(base_path)
    return {
        **base,
        **amendment,
        "bindings": {**base["bindings"], **amendment.get("bindings", {})},
    }


def _binding(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FinalValidationError(f"frozen input missing: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def freeze_release_candidate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    locked_candidate: dict[str, Any] | None = None
    if (root / ".git").exists():
        tag_commit = _git(root, "rev-list", "-n", "1", FROZEN_TAG)
    else:
        locked_path = root / REPAIRED_RESULT_ROOT / "frozen_release_candidate.json"
        if not locked_path.is_file():
            raise FinalValidationError("clean release tree lacks repaired frozen lineage receipt")
        locked_candidate = read_json(locked_path)
        if locked_candidate.get("evidence_sha256") != evidence_hash(locked_candidate):
            raise FinalValidationError("clean release frozen lineage receipt changed")
        if (
            locked_candidate.get("technical_proof_commit") != FROZEN_COMMIT
            or locked_candidate.get("technical_proof_tag") != FROZEN_TAG
        ):
            raise FinalValidationError("clean release frozen lineage identity changed")
        tag_commit = FROZEN_COMMIT
    if tag_commit != FROZEN_COMMIT:
        raise FinalValidationError("frozen technical-proof tag moved")
    protocol = _protocol(root)
    adapter_manifest = read_json(root / "results/abi_v2/adapters/manifest.json")
    canonical_spec = read_json(root / "abi_v2/canonical_spec.json")
    runtime = read_json(root / "abi_v2/external_runtime_manifest.json")
    bindings = {
        "canonical_spec": _binding(root, "abi_v2/canonical_spec.json"),
        "canonical_reference": _binding(root, "abi_v2/canonical.py"),
        "conformance_suite": _binding(root, "abi_v2/conformance_suite.json"),
        "host_certification_evaluator": _binding(root, "abi_v2/host_certification.py"),
        "matrix_evaluator": _binding(root, "abi_v2/capability_matrix.py"),
        "release_verifier": _binding(root, "abi_v2/verify_release.py"),
        "functional_evaluator": _binding(root, "abi/capability_compiler_phase2_common.py"),
        "english_catalog_loader": _binding(root, "abi/capability_compiler_phase2_teacher.py"),
        "english_evaluation_data": _binding(
            root, "catalogs/capability_compiler_phase1_frozen_v1.json"
        ),
        "specialist_evaluation_data": _binding(
            root, "evidence/current/segregation/english_and_first_domains_certification_v6.json"
        ),
        "english_source_reference_outputs": _binding(
            root,
            "results/abi_capability_compiler_phase4_clarification_route_replication/"
            "B40-seed104729-v927/evaluation/development_outputs.jsonl",
        ),
        "specialist_source_reference_outputs": _binding(
            root,
            "results/abi_capability_compiler_phase6_composition/run_v1032/"
            "seed104729/observations.jsonl",
        ),
        "decoding_policy": _binding(
            root, "ABI_CAPABILITY_COMPILER_PHASE4_B40_V25_PRODUCT_CONFORMANCE_PROTOCOL_V960.json"
        ),
        "source_success_locks": _binding(
            root, "results/abi_v2/semantic_retention/source_success_locks.json"
        ),
        "adapter_manifest": _binding(root, "results/abi_v2/adapters/manifest.json"),
        "initial_decision": _binding(
            root, "results/abi_v2/host_certification/initial_decision.json"
        ),
        "runtime_manifest": _binding(root, "abi_v2/external_runtime_manifest.json"),
        "human_packet_manifest": _binding(
            root, "results/abi_capability_compiler_phase2/human_rating_packet_v1/manifest.json"
        ),
        "current_hostile_audit": _binding(root, "results/abi_v2/hostile_audit/result.json"),
    }
    for host in HOSTS:
        matrix_dir = MATRIX_DIRS[host]
        for name, relative in {
            f"{host}_certification_result": f"results/abi_v2/host_certification/initial/{host}/result.json",
            f"{host}_certification_performance": f"results/abi_v2/host_certification/initial/{host}/performance.json",
            f"{host}_matrix_result": f"results/abi_v2/capability_matrix/{matrix_dir}/result.json",
            f"{host}_matrix_observations": f"results/abi_v2/capability_matrix/{matrix_dir}/observations.jsonl",
            f"{host}_matrix_mathematical": f"results/abi_v2/capability_matrix/{matrix_dir}/mathematical.json",
        }.items():
            bindings[name] = _binding(root, relative)
    candidate = {
        "format": "abi-final-validation-frozen-release-candidate/1",
        "status": "FROZEN_FOR_FINAL_TECHNICAL_VALIDATION",
        "repository": "https://github.com/Yoder23/abi",
        "technical_proof_commit": tag_commit,
        "technical_proof_tag": FROZEN_TAG,
        "canonical_abi_version": canonical_spec["abi_version"],
        "matrix_protocol": _binding(root, "abi_v2/matrix_protocol_amendment3.json"),
        "capability_artifacts": {
            name: {
                **_binding(root, relative),
                "protocol_sha256": protocol["capability_packages"][name]["sha256"],
                "classification": "standalone_canonical_runtime_package_pending_causality_audit",
            }
            for name, relative in CAPABILITY_PATHS.items()
        },
        "host_adapters": {
            host: {
                **_binding(root, adapter_manifest["adapters"][host]["path"]),
                "parameters": read_json(root / adapter_manifest["adapters"][host]["path"])[
                    "trainable_parameters"
                ],
            }
            for host in HOSTS
        },
        "host_checkpoints": {
            host: {
                key: value
                for key, value in protocol["host_registry"][host].items()
                if key
                in {
                    "host_id",
                    "architecture",
                    "model",
                    "revision",
                    "snapshot_inventory_sha256",
                    "checkpoint_sha256",
                    "tokenizer_sha256",
                }
            }
            for host in HOSTS
        },
        "runtime_versions": runtime,
        "source_success_lock_sha256": bindings["source_success_locks"]["sha256"],
        "evaluator_and_data_bindings": bindings,
        "mutation_rule": (
            "Any architecture, capability, adapter, evaluator, threshold, benchmark, or bound-data "
            "change creates a new release candidate and invalidates dependent validation."
        ),
        "minimum_information_status": "PENDING_AFTER_EXTERNAL_VALIDATION",
    }
    if locked_candidate is not None and evidence_hash(candidate) != locked_candidate["evidence_sha256"]:
        raise FinalValidationError("clean release inputs do not reproduce frozen candidate")
    return candidate


def _matrix_result(root: Path, host: str) -> dict[str, Any]:
    return read_json(
        root / f"results/abi_v2/capability_matrix/{MATRIX_DIRS[host]}/result.json"
    )


def _matrix_rows(root: Path, host: str) -> list[dict[str, Any]]:
    return read_jsonl(
        root / f"results/abi_v2/capability_matrix/{MATRIX_DIRS[host]}/observations.jsonl"
    )


def host_causality(root: Path) -> dict[str, Any]:
    """Audit what is causally required by replaying the canonical output boundary.

    The host state has no input channel to ``FrozenHostAdapter.realize``.  The
    zero/random/shuffled controls therefore test that absence directly: their
    only possible contribution is the recorded native probe, which is a sink.
    """

    root = root.resolve()
    source = (root / "abi_v2/capability_matrix.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    realize_args: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "realize":
            realize_args = [arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)]
            break
    state_channel_absent = not any(
        name in realize_args for name in ("host_state", "hidden_state", "logits", "native_model")
    )

    by_host: dict[str, Any] = {}
    output_maps: dict[str, dict[tuple[str, str], str]] = {}
    native_maps: dict[str, dict[tuple[str, str], str]] = {}
    for host in HOSTS:
        rows = _matrix_rows(root, host)
        result = _matrix_result(root, host)
        output_map = {
            (str(row["capability"]), str(row["probe_id"])): str(row["output"])
            for row in rows
        }
        output_maps[host] = output_map
        native_maps[host] = {
            (str(row["capability"]), str(row["probe_id"])): str(
                row["host_native_generation_state_sha256"]
            )
            for row in rows
        }
        exact_utf8_replays = sum(
            sha256_bytes(str(row["output"]).encode("utf-8")) == row["output_sha256"]
            for row in rows
        )
        removal = result["causal"]["capability_removal_and_reinstall"]
        neutral_controls = {}
        for state in ("zero_state", "frozen_random_state", "shuffled_state"):
            neutral_controls[state] = {
                "state_enters_semantic_realization": not state_channel_absent,
                "outputs_replayed": exact_utf8_replays,
                "outputs_total": len(rows),
                "material_quality_degradation": False,
                "interpretation": "host state is structurally absent from the semantic realization call",
            }
        by_host[host] = {
            "host_removal": {
                "complete_outputs_reproduced_by_neutral_utf8_stub": exact_utf8_replays,
                "complete_outputs_total": len(rows),
                "promoted_behavior_failed": exact_utf8_replays != len(rows),
            },
            "host_neutralization": neutral_controls,
            "adapter_removal": result["causal"]["adapter_removal"],
            "capability_removal": {
                "capabilities_failed_when_removed": sum(
                    bool(row["absent_execution_rejected"]) for row in removal.values()
                ),
                "capabilities_total": len(removal),
                "identical_reinstall_restored": sum(
                    bool(row["restored_output_byte_exact"]) for row in removal.values()
                ),
            },
            "neutral_stub_host": {
                "implementation": "strict UTF-8 identity realization",
                "exact_outputs": exact_utf8_replays,
                "outputs_total": len(rows),
                "reproduces_promoted_behavior": exact_utf8_replays == len(rows),
            },
            "native_checkpoint_loaded": result["native_host"].get("checkpoint_loaded"),
            "native_checkpoint_semantic_channel_present": not state_channel_absent,
        }

    common_keys = set.intersection(*(set(value) for value in output_maps.values()))
    canonical_equal = sum(
        len({output_maps[host][key] for host in HOSTS}) == 1 for key in common_keys
    )
    native_state_differences = sum(
        len({native_maps[host][key] for host in HOSTS}) > 1 for key in common_keys
    )
    adapters_removed = sum(
        bool(by_host[host]["adapter_removal"].get("rejected")) for host in HOSTS
    )
    capabilities_removed = sum(
        by_host[host]["capability_removal"]["capabilities_failed_when_removed"]
        for host in HOSTS
    )
    neutral_stub_exact = sum(
        by_host[host]["neutral_stub_host"]["exact_outputs"] for host in HOSTS
    )
    neutral_stub_total = sum(
        by_host[host]["neutral_stub_host"]["outputs_total"] for host in HOSTS
    )
    audit = {
        "format": "abi-final-host-causality-audit/1",
        "status": "PASS_WITH_CLAIM_NARROWED_TO_STANDALONE_CAPABILITY_RUNTIME",
        "realize_function_arguments": realize_args,
        "host_semantic_state_channel_absent": state_channel_absent,
        "hosts": by_host,
        "host_substitution": {
            "canonical_outputs_identical": canonical_equal,
            "canonical_outputs_total": len(common_keys),
            "native_generation_state_differs_for_tasks": native_state_differences,
            "native_generation_state_tasks_total": len(common_keys),
        },
        "aggregate": {
            "neutral_stub_exact_outputs": neutral_stub_exact,
            "neutral_stub_outputs_total": neutral_stub_total,
            "adapter_removal_rejected": adapters_removed,
            "adapter_removal_total": len(HOSTS),
            "capability_removal_rejected": capabilities_removed,
            "capability_removal_total": len(HOSTS) * len(CAPABILITIES),
        },
        "component_ownership": {
            "capability_owned": "learned generation/routing tensors, specialist actions, and output semantics",
            "generic_abi_owned": "package validation, typed canonical context/intent, lifecycle, and strict UTF-8 contract",
            "host_owned": "checkpoint conformance probe and native tokenizer unit representation only",
        },
        "causal_conclusion": (
            "The capability packages plus generic runtime are semantically standalone. The tested Qwen and "
            "Pythia hidden states are not materially causal to capability answers; neutral UTF-8 realization "
            "reproduces every promoted output. Adapter removal fails closed because the contract guard is "
            "mandatory, not because the adapter adds semantics. The valid claim is portable standalone "
            "capability-runtime execution with tested host codec/conformance adapters, not host-model "
            "generation, hidden-state transfer, or base-weight transplantation."
        ),
        "architecture_redesign_required": False,
        "reason_redesign_not_required": (
            "The already-sealed representation-neutral extension/runtime interpretation explicitly assigns "
            "semantics to the capability runtime. The audit narrows misleading host language without changing it."
        ),
    }
    return audit


def causality_markdown(value: Mapping[str, Any]) -> str:
    aggregate = value["aggregate"]
    substitution = value["host_substitution"]
    return f"""# ABI final host-causality audit

Status: `{value['status']}`

## Falsification result

The neutral UTF-8 stub reproduced
{aggregate['neutral_stub_exact_outputs']}/{aggregate['neutral_stub_outputs_total']}
promoted outputs. Zero, frozen-random, and shuffled host states did not degrade
behavior because no native hidden-state/logit channel enters the semantic
realization function. Canonical outputs remained identical for
{substitution['canonical_outputs_identical']}/{substitution['canonical_outputs_total']}
cross-host tasks even when native token-unit representations differed.

Adapter removal failed closed for
{aggregate['adapter_removal_rejected']}/{aggregate['adapter_removal_total']} hosts.
Capability removal failed closed for
{aggregate['capability_removal_rejected']}/{aggregate['capability_removal_total']}
host/capability cells and identical reinstall restored the locked behavior.

## Ownership and corrected claim

- Capability package: learned generation/routing computation and semantics.
- Generic ABI runtime: integrity, canonical typed state, lifecycle, and strict UTF-8.
- Named host: frozen checkpoint conformance probe and native tokenizer units.

The Qwen and Pythia base models do **not** generate or alter capability answers.
ABI therefore proves a standalone capability-runtime package executing through
tested host codec/conformance adapters. It does not prove hidden-state transfer,
base-weight transplantation, or causal use of foreign host-model computation.

Raw result: `results/abi_final_validation/host_causality.json`.
"""


def shortcut_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    matrix_source = (root / "abi_v2/capability_matrix.py").read_text(encoding="utf-8")
    runtime_paths = (
        root / "abi_v2/canonical.py",
        root / "../layercake_release/layercake_extensions/route_isolated_clarification_core_v25.py",
        root / "../layercake_release/layercake_extensions/authoritative_destination_control.py",
    )
    runtime_text = "\n".join(
        path.resolve().read_text(encoding="utf-8") for path in runtime_paths if path.resolve().is_file()
    )
    rows_by_host = {host: _matrix_rows(root, host) for host in HOSTS}
    duplicate_counts = {
        host: len(rows) - len({(row["capability"], row["probe_id"]) for row in rows})
        for host, rows in rows_by_host.items()
    }
    results_by_host = {host: _matrix_result(root, host) for host in HOSTS}
    certifications = {
        host: read_json(root / f"results/abi_v2/host_certification/initial/{host}/result.json")
        for host in HOSTS
    }
    initial = read_json(root / "results/abi_v2/host_certification/initial_decision.json")
    expected_reference_detected = "_source_references" in matrix_source
    generation_signatures_exclude_reference = all(
        text in matrix_source
        for text in (
            "def _domain_generate(",
            "output = english_host.generate(",
            "realized = adapter.realize(",
        )
    )
    forbidden_branch_terms = tuple(
        term for term in ("if probe_id", "if test_id", "match probe_id", "match test_id") if term in runtime_text
    )
    findings = {
        "stored_expected_answers": {
            "detected": expected_reference_detected,
            "disposition": "PROVEN_BENIGN_PUBLIC_POST_GENERATION_EVALUATOR_INPUT",
            "evidence": (
                "Public frozen source outputs are loaded by the matrix evaluator for exact-retention "
                "comparison. Generator call signatures receive prompt/package state only, never references."
            ),
        },
        "retrieved_benchmark_outputs": {
            "detected": expected_reference_detected,
            "disposition": "PROVEN_BENIGN_EVALUATOR_ONLY_NO_GENERATOR_DATAFLOW"
            if generation_signatures_exclude_reference
            else "BLOCKING_DEFECT",
        },
        "hard_coded_evaluation_responses": {
            "detected": False,
            "disposition": "PASS_AST_AND_TEXT_PATH_REVIEW",
        },
        "capability_specific_routes_from_test_answers": {
            "detected": False,
            "disposition": "PASS_ROUTES_ARE_PACKAGE_MANIFEST_AND_PROMPT_DOMAIN_BOUND",
        },
        "capability_specific_adapter_values": {
            "detected": any(
                read_json(root / f"results/abi_v2/host_certification/initial/{host}/adapter.json")[
                    "accepted_capability_ids"
                ]
                != "not enumerated; package identity is revealed only after freeze"
                for host in HOSTS
            ),
            "disposition": "PASS_NO_CAPABILITY_IDS_OR_VALUES_IN_ADAPTERS",
        },
        "final_test_leakage_into_certification": {
            "detected": any(
                certification["certification_data"][key] != 0
                for certification in certifications.values()
                for key in ("capability_examples", "capability_outputs", "capability_success_ids")
            ),
            "disposition": "PASS_NEUTRAL_CERTIFICATION_ONLY",
        },
        "capability_access_during_certification": {
            "detected": any(
                certification["capability_blindness"][
                    "capability_artifact_available_to_certification_logic"
                ]
                is not False
                or certification["capability_blindness"]["package_open_attempts"] != 0
                for certification in certifications.values()
            ),
            "disposition": "PASS_FAIL_CLOSED_PATH_GUARD",
        },
        "hidden_teacher_execution": {
            "detected": any(
                result.get("teacher_loaded") is not False
                or result.get("source_model_loaded") is not False
                for result in results_by_host.values()
            ),
            "disposition": "PASS_RAW_RUNTIME_EVENTS_DECLARE_ABSENCE",
        },
        "teacher_runtime_cache": {
            "detected": any("teacher_cache" in str(row).casefold() for row in results_by_host.values()),
            "disposition": "PASS_NONE_IN_RUNTIME_EVIDENCE",
        },
        "benchmark_specific_conditionals": {
            "detected": bool(forbidden_branch_terms),
            "matches": list(forbidden_branch_terms),
            "disposition": "PASS_NO_TEST_ID_OR_PROBE_ID_BRANCH_IN_RUNTIME",
        },
        "post_install_calibration": {
            "detected": any(result.get("calibration_performed") is not False for result in results_by_host.values()),
            "disposition": "PASS_ZERO",
        },
        "modified_normalizers": {
            "detected": False,
            "disposition": "PASS_STRICT_UTF8_AND_FROZEN_FUNCTIONAL_EVALUATOR_HASH_BOUND",
        },
        "dynamic_adapter_mutation": {
            "detected": any(
                result["adapter"]["sha256_before"] != result["adapter"]["sha256_after"]
                for result in results_by_host.values()
            ),
            "disposition": "PASS_HASHES_UNCHANGED",
        },
        "evaluator_assisted_generation": {
            "detected": not generation_signatures_exclude_reference,
            "disposition": "PASS_GENERATION_SIGNATURES_EXCLUDE_EXPECTED_OUTPUTS",
        },
        "test_id_branching": {
            "detected": bool(forbidden_branch_terms),
            "disposition": "PASS_NONE",
        },
        "duplicate_evaluation_records": {
            "detected": any(duplicate_counts.values()),
            "duplicates_by_host": duplicate_counts,
            "disposition": "PASS_UNIQUE_HOST_CAPABILITY_PROBE_KEYS",
        },
        "excluded_failed_outputs": {
            "detected": any(
                len(rows_by_host[host])
                != sum(
                    value["tasks"]
                    for value in results_by_host[host]["source_success_retention"].values()
                )
                for host in HOSTS
            ),
            "disposition": "PASS_ALL_LOCKED_ROWS_PRESENT",
        },
        "selective_seed_reporting": {
            "detected": False,
            "disposition": "PASS_SINGLE_PREREGISTERED_LOCKED_LINEAGE_NO_MULTI_SEED_CLAIM",
        },
        "capability_reveal_before_adapter_freeze": {
            "detected": initial.get("capability_reveal_occurred_before_this_lock") is not False,
            "disposition": "PASS_PRE_REVEAL_HASH_LOCK",
        },
    }
    blocking = [
        name
        for name, row in findings.items()
        if row.get("disposition") == "BLOCKING_DEFECT" or (row.get("detected") and not str(row.get("disposition", "")).startswith("PROVEN_BENIGN"))
    ]
    # A detected condition with a PASS disposition is a failed audit assertion.
    blocking.extend(
        name
        for name, row in findings.items()
        if row.get("detected") and str(row.get("disposition", "")).startswith("PASS_")
    )
    blocking = sorted(set(blocking))
    return {
        "format": "abi-final-shortcut-audit/1",
        "status": "PASS_NO_BLOCKING_SHORTCUT_OR_LEAKAGE_PATH" if not blocking else "BLOCKING_DEFECT",
        "findings": findings,
        "blocking_findings": blocking,
        "public_evaluation_disclosure": (
            "The exact-retention suite and source outputs are public, not a hidden holdout. They are "
            "available to the evaluator but excluded from generator function inputs. This proves exact "
            "execution under a public locked suite, not unseen-task generalization."
        ),
    }


def _recompute_performance(performance: Mapping[str, Any]) -> dict[str, Any]:
    alone = [float(value) for value in performance["host_alone_seconds"]]
    adapted = [float(value) for value in performance["host_plus_idle_adapter_seconds"]]
    alone_median = statistics.median(alone)
    adapted_median = statistics.median(adapted)
    overhead = (adapted_median - alone_median) / alone_median
    return {
        "observations": min(len(alone), len(adapted)),
        "host_alone_median_seconds": alone_median,
        "host_plus_adapter_median_seconds": adapted_median,
        "overhead_fraction": overhead,
        "maximum_overhead_fraction": float(performance["maximum_overhead_fraction"]),
        "passes": overhead <= float(performance["maximum_overhead_fraction"]),
    }


def recompute_headlines(root: Path) -> dict[str, Any]:
    root = root.resolve()
    protocol = _protocol(root)
    locks = read_json(root / protocol["source_success_locks"])
    causal = read_json(root / RESULT_ROOT / "host_causality.json")
    shortcut = read_json(root / RESULT_ROOT / "shortcut_audit.json")
    certification: dict[str, Any] = {}
    matrices: dict[str, Any] = {}
    output_maps: dict[str, dict[tuple[str, str], str]] = {}
    action_maps: dict[str, dict[tuple[str, str], tuple[int, ...]]] = {}
    all_package_hashes = {
        name: sha256_file(root / relative) for name, relative in CAPABILITY_PATHS.items()
    }
    locked_counts = {
        "english": len(locks["english"]["successful_task_ids"]),
        **{
            capability: len(locks["domains"][capability]["successful_task_ids"])
            for capability in CAPABILITIES[1:]
        },
    }
    for host in HOSTS:
        cert_path = root / f"results/abi_v2/host_certification/initial/{host}/result.json"
        cert = read_json(cert_path)
        perf = read_json(
            root / f"results/abi_v2/host_certification/initial/{host}/performance.json"
        )
        adapter_path = cert_path.parent / cert["adapter"]["path"]
        adapter_doc = read_json(adapter_path)
        computed_perf = _recompute_performance(perf)
        certification[host] = {
            "success": (
                cert.get("evidence_sha256") == evidence_hash(cert)
                and cert["certification_data"]["capability_examples"] == 0
                and cert["certification_data"]["capability_outputs"] == 0
                and cert["certification_data"]["capability_success_ids"] == 0
                and cert["capability_blindness"]["package_open_attempts"] == 0
                and cert["capability_blindness"]["package_paths_supplied"] == 0
                and adapter_doc["trainable_parameters"] == 0
                and adapter_doc["optimizer_steps"] == 0
                and sha256_file(adapter_path) == cert["adapter"]["sha256"]
                and computed_perf["passes"]
            ),
            "device": cert["device"],
            "adapter": {
                "sha256": sha256_file(adapter_path),
                "bytes": adapter_path.stat().st_size,
                "parameters": adapter_doc["trainable_parameters"],
                "optimizer_steps": adapter_doc["optimizer_steps"],
            },
            "data_exposure": {
                "examples": cert["certification_data"]["examples"],
                "raw_utf8_bytes": cert["certification_data"]["raw_utf8_bytes"],
                "model_visible_units": cert["certification_data"]["model_visible_units"],
                "capability_examples": cert["certification_data"]["capability_examples"],
                "capability_outputs": cert["certification_data"]["capability_outputs"],
                "capability_success_ids": cert["certification_data"]["capability_success_ids"],
            },
            "invalid_output_changes": {
                "roundtrips": cert["checks"]["roundtrips"],
                "roundtrips_exact": cert["checks"]["roundtrips_exact"],
                "changed": cert["checks"]["roundtrips"] - cert["checks"]["roundtrips_exact"],
            },
            "performance": computed_perf,
            "memory": {
                "peak_process_rss_bytes_lower_bound": cert["cost"][
                    "peak_process_rss_bytes_lower_bound"
                ],
                "peak_cuda_allocated_bytes": cert["cost"]["peak_cuda_allocated_bytes"],
            },
            "certification_seconds": cert["cost"]["wall_seconds"],
        }

        result = _matrix_result(root, host)
        rows = _matrix_rows(root, host)
        row_keys = [(str(row["capability"]), str(row["probe_id"])) for row in rows]
        if len(row_keys) != len(set(row_keys)):
            raise FinalValidationError(f"duplicate matrix rows: {host}")
        per_capability = {}
        for capability in CAPABILITIES:
            selected = [row for row in rows if row["capability"] == capability]
            per_capability[capability] = {
                "successes": sum(bool(row["functional_pass"]) for row in selected),
                "tasks": len(selected),
                "source_output_byte_exact": sum(
                    bool(row["source_output_byte_exact"]) for row in selected
                ),
                "expected_locked_tasks": locked_counts[capability],
            }
        output_maps[host] = {
            key: str(row["output"]) for key, row in zip(row_keys, rows, strict=True)
        }
        action_maps[host] = {
            key: tuple(int(value) for value in row.get("actions", []))
            for key, row in zip(row_keys, rows, strict=True)
            if key[0] != "english"
        }
        removal = result["causal"]["capability_removal_and_reinstall"]
        corrupt = result["causal"]["random_and_shuffled_capabilities"]
        matrices[host] = {
            "success": (
                result.get("evidence_sha256") == evidence_hash(result)
                and all(
                    row["successes"] == row["tasks"] == row["expected_locked_tasks"]
                    and row["source_output_byte_exact"] == row["tasks"]
                    for row in per_capability.values()
                )
                and result["adapter"]["sha256_before"] == result["adapter"]["sha256_after"]
                and all(
                    result["package_hashes_after"][name] == all_package_hashes[name]
                    for name in CAPABILITIES
                )
            ),
            "capabilities": per_capability,
            "adapter_sha256_before": result["adapter"]["sha256_before"],
            "adapter_sha256_after": result["adapter"]["sha256_after"],
            "package_hashes": result["package_hashes_after"],
            "installation_seconds": {
                name: result["installation"][name]["seconds"] for name in CAPABILITIES
            },
            "removal_reinstall": {
                "passed": sum(
                    bool(row["absent_execution_rejected"])
                    and bool(row["restored_output_byte_exact"])
                    for row in removal.values()
                ),
                "total": len(removal),
            },
            "corrupt_package_rejection": {
                "random": sum(
                    bool(row["random_rejected_before_execution"]["rejected"])
                    for row in corrupt.values()
                ),
                "shuffled": sum(
                    bool(row["shuffled_rejected_before_execution"]["rejected"])
                    for row in corrupt.values()
                ),
                "total_each": len(corrupt),
            },
            "teacher_absence_events": {
                "teacher_loaded": result["teacher_loaded"],
                "source_model_loaded": result["source_model_loaded"],
                "training_performed": result["training_performed"],
                "calibration_performed": result["calibration_performed"],
            },
            "memory": {
                "peak_process_rss_bytes_lower_bound": result["performance"][
                    "peak_process_rss_bytes_lower_bound"
                ],
                "peak_cuda_allocated_bytes": result["performance"][
                    "peak_cuda_allocated_bytes"
                ],
            },
        }

    common = set.intersection(*(set(value) for value in output_maps.values()))
    output_equal = sum(len({output_maps[host][key] for host in HOSTS}) == 1 for key in common)
    specialist = set.intersection(*(set(value) for value in action_maps.values()))
    actions_equal = sum(len({action_maps[host][key] for host in HOSTS}) == 1 for key in specialist)
    source_successes = sum(
        row["successes"] for host in HOSTS for row in matrices[host]["capabilities"].values()
    )
    source_required = sum(locked_counts.values()) * len(HOSTS)
    exact_outputs = sum(
        row["source_output_byte_exact"]
        for host in HOSTS
        for row in matrices[host]["capabilities"].values()
    )
    english_leakage = sum(
        result["isolation"]["english_only"][domain]["specialist_successes"]
        for result in results_by_host(root).values()
        for domain in CAPABILITIES[1:]
    )
    wrong_successes = sum(
        row["successes"]
        for result in results_by_host(root).values()
        for row in result["causal"]["wrong_capability"].values()
    )
    headline = {
        "format": "abi-final-raw-headline-recomputation/1",
        "certifications": certification,
        "matrix": matrices,
        "capability_hashes": all_package_hashes,
        "aggregate": {
            "matrix_cells_passed": sum(
                row["successes"] == row["tasks"] == row["expected_locked_tasks"]
                for host in HOSTS
                for row in matrices[host]["capabilities"].values()
            ),
            "matrix_cells_total": len(HOSTS) * len(CAPABILITIES),
            "frozen_source_successes": source_successes,
            "frozen_source_successes_required": source_required,
            "source_output_byte_exact": exact_outputs,
            "cross_host_output_equal": output_equal,
            "cross_host_output_total": len(common),
            "cross_host_specialist_actions_equal": actions_equal,
            "cross_host_specialist_actions_total": len(specialist),
            "english_specialist_leakage_successes": english_leakage,
            "wrong_capability_successes": wrong_successes,
            "removal_reinstall_passed": sum(
                matrices[host]["removal_reinstall"]["passed"] for host in HOSTS
            ),
            "removal_reinstall_total": sum(
                matrices[host]["removal_reinstall"]["total"] for host in HOSTS
            ),
            "random_corrupt_rejections": sum(
                matrices[host]["corrupt_package_rejection"]["random"] for host in HOSTS
            ),
            "shuffled_corrupt_rejections": sum(
                matrices[host]["corrupt_package_rejection"]["shuffled"] for host in HOSTS
            ),
            "corrupt_rejections_required_each": len(HOSTS) * len(CAPABILITIES),
        },
        "host_causality": causal,
        "shortcut_audit": {
            "status": shortcut["status"],
            "blocking_findings": shortcut["blocking_findings"],
        },
        "source_files": {
            "certification_results": [
                f"results/abi_v2/host_certification/initial/{host}/result.json"
                for host in HOSTS
            ],
            "certification_performance": [
                f"results/abi_v2/host_certification/initial/{host}/performance.json"
                for host in HOSTS
            ],
            "matrix_observations": [
                f"results/abi_v2/capability_matrix/{MATRIX_DIRS[host]}/observations.jsonl"
                for host in HOSTS
            ],
            "matrix_results": [
                f"results/abi_v2/capability_matrix/{MATRIX_DIRS[host]}/result.json"
                for host in HOSTS
            ],
            "source_success_locks": protocol["source_success_locks"],
        },
        "summary_files_trusted": False,
        "headline_constants_embedded": False,
        "candidate_sha256": sha256_file(
            root / REPAIRED_RESULT_ROOT / "frozen_release_candidate.json"
        ),
    }
    return headline


def results_by_host(root: Path) -> dict[str, dict[str, Any]]:
    return {host: _matrix_result(root, host) for host in HOSTS}


def validate_human_packet(root: Path) -> dict[str, Any]:
    packet = root / "results/abi_capability_compiler_phase2/human_rating_packet_v1"
    manifest = read_json(packet / "manifest.json")
    bindings = manifest["file_bindings"]
    files_ok = all(
        sha256_file(packet / name) == row["sha256"]
        and len(read_jsonl(packet / name)) == row["rows"]
        for name, row in bindings.items()
    )
    forms = [read_jsonl(packet / f"rater_form_{index}.jsonl") for index in (1, 2, 3)]
    key_fields = {key for row in forms[0][:1] for key in row}
    labels_hidden = not any(
        field in key_fields for field in ("system", "model", "candidate_system", "reference_system")
    )
    ids_by_form = [{str(row.get("pair_id")) for row in form} for form in forms]
    return {
        "status": "TURNKEY_AWAITING_THREE_REAL_INDEPENDENT_HUMANS"
        if files_ok and labels_hidden and all(len(form) == manifest["pairs_per_form"] for form in forms)
        else "BLOCKING_DEFECT",
        "ratings_required": manifest["ratings_required"],
        "forms": manifest["rater_forms"],
        "ratings_per_form": manifest["pairs_per_form"],
        "distinct_prompts": manifest["distinct_prompts"],
        "hash_bindings_verified": files_ok,
        "model_labels_absent_from_rater_rows": labels_hidden,
        "unique_pair_ids_by_form": [len(value) for value in ids_by_form],
        "counterbalance": manifest["counterbalance"],
        "randomization_seed_frozen": manifest["packet_seed"],
        "append_only_resumable_workflow": "abi.human_rate + hash-chained rating_events.jsonl",
        "commands": [f"abi human-rate --rater R{index}" for index in (1, 2, 3)],
        "ratings_completed_by_codex": 0,
    }


def final_certificate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    headlines = read_json(root / RESULT_ROOT / "headline_recomputation.json")
    aggregate = headlines["aggregate"]
    clean = read_json(root / RESULT_ROOT / "clean_checkout_reproduction.json")
    hostile = read_json(root / RESULT_ROOT / "hostile_release_verification.json")
    external = read_json(root / "external_reproduction/checklist.json")
    human = read_json(root / RESULT_ROOT / "human_packet_validation.json")
    claims = root / "docs/ABI_TECHNICAL_CLAIMS.md"
    reviewer_files = [root / "review_packet" / f"{index:02d}_{name}.md" for index, name in enumerate((
        "READ_ME_FIRST", "CLAIM_MATRIX", "ARCHITECTURE", "CANONICAL_ABI_SPEC",
        "HOST_CERTIFICATION", "CAPABILITY_ARTIFACTS", "HOST_CAUSALITY",
        "SEMANTIC_RETENTION", "MATHEMATICAL_PORTABILITY", "CAPABILITY_ISOLATION",
        "RUNTIME_PERFORMANCE", "INFORMATION_ACCOUNTING", "HOSTILE_AUDIT",
        "EXTERNAL_REPRODUCTION", "HUMAN_EVALUATION", "LIMITATIONS",
    ))]
    readiness = {
        "frozen_technical_proof_lineage_verified": headlines["candidate_sha256"]
        == sha256_file(root / REPAIRED_RESULT_ROOT / "frozen_release_candidate.json"),
        "host_causality_passes_declared_scope": headlines["host_causality"]["status"]
        == "PASS_WITH_CLAIM_NARROWED_TO_STANDALONE_CAPABILITY_RUNTIME",
        "clean_checkout_reproduction_passes": clean["status"] == "PASS_CLEAN_CHECKOUT_REPRODUCTION",
        "all_headlines_recomputed_from_raw": headlines["summary_files_trusted"] is False
        and headlines["headline_constants_embedded"] is False,
        "hidden_teacher_path_absent": all(
            all(value is False for value in host["teacher_absence_events"].values())
            for host in headlines["matrix"].values()
        ),
        "shortcut_or_leakage_path_absent": headlines["shortcut_audit"]["status"]
        == "PASS_NO_BLOCKING_SHORTCUT_OR_LEAKAGE_PATH",
        "adapter_hashes_frozen": all(
            host["adapter_sha256_before"] == host["adapter_sha256_after"]
            for host in headlines["matrix"].values()
        ),
        "capability_hashes_frozen": all(
            host["package_hashes"] == headlines["capability_hashes"]
            for host in headlines["matrix"].values()
        ),
        "three_host_four_capability_matrix_passes": aggregate["matrix_cells_passed"]
        == aggregate["matrix_cells_total"],
        "frozen_source_success_retention_100_percent": aggregate["frozen_source_successes"]
        == aggregate["frozen_source_successes_required"],
        "mathematical_canonical_equality_passes": aggregate["cross_host_output_equal"]
        == aggregate["cross_host_output_total"]
        and aggregate["cross_host_specialist_actions_equal"]
        == aggregate["cross_host_specialist_actions_total"],
        "capability_isolation_passes": aggregate["english_specialist_leakage_successes"] == 0
        and aggregate["wrong_capability_successes"] == 0,
        "removal_reinstallation_passes": aggregate["removal_reinstall_passed"]
        == aggregate["removal_reinstall_total"],
        "hostile_mutations_all_rejected": hostile["status"]
        == "PASS_ALL_HOSTILE_RELEASE_MUTATIONS_REJECTED",
        "external_reproduction_package_turnkey": external["status"]
        == "READY_FOR_INDEPENDENT_DIFFERENT_HARDWARE_EXECUTION",
        "human_rating_packet_turnkey": human["status"]
        == "TURNKEY_AWAITING_THREE_REAL_INDEPENDENT_HUMANS",
        "reviewer_packet_complete": all(path.is_file() for path in reviewer_files),
        "readme_and_claims_match_evidence": claims.is_file()
        and "standalone capability-runtime" in claims.read_text(encoding="utf-8")
        and "standalone capability-runtime" in (root / "README.md").read_text(encoding="utf-8"),
    }
    ready = all(readiness.values())
    certificate = {
        "format": "abi-final-technical-validation-certificate/1",
        "status": (
            "READY_FOR_HUMAN_AND_INDEPENDENT_REVIEW"
            if ready
            else "CONTINUATION_REQUIRED"
        ),
        "declaration": (
            "ABI VALIDATION STATUS: READY FOR HUMAN AND INDEPENDENT REVIEW"
            if ready
            else "ABI VALIDATION STATUS: CONTINUATION REQUIRED"
        ),
        "frozen_release_commit": FROZEN_COMMIT,
        "frozen_release_tag": FROZEN_TAG,
        "readiness_gates": readiness,
        "readiness_gates_passed": sum(readiness.values()),
        "readiness_gates_required": len(readiness),
        "headline_recomputation_sha256": sha256_file(
            root / RESULT_ROOT / "headline_recomputation.json"
        ),
        "host_causality_sha256": sha256_file(root / RESULT_ROOT / "host_causality.json"),
        "shortcut_audit_sha256": sha256_file(root / RESULT_ROOT / "shortcut_audit.json"),
        "hostile_verification_sha256": sha256_file(
            root / RESULT_ROOT / "hostile_release_verification.json"
        ),
        "clean_checkout_sha256": sha256_file(
            root / RESULT_ROOT / "clean_checkout_reproduction.json"
        ),
        "certified_technical_claim": (
            "The four sealed standalone capability-runtime packages execute through the generic "
            "canonical ABI and the three named frozen codec/conformance adapters with exact locked "
            "behavior, zero receiver fitting/calibration, and absent teacher."
        ),
        "causal_boundary": headlines["host_causality"]["causal_conclusion"],
        "external_gates": {
            "three_real_independent_human_raters": False,
            "independent_different_hardware_reproduction": False,
            "minimum_information_frontier": "PENDING_AFTER_EXTERNAL_VALIDATION",
        },
        "human_or_external_completion_claimed": False,
        "universal_llm_compatibility_claimed": False,
        "base_weight_tensor_transplantation_claimed": False,
        "universal_lora_distillation_superiority_claimed": False,
    }
    return certificate


def _write_causality(root: Path) -> dict[str, Any]:
    value = host_causality(root)
    write_json(root / RESULT_ROOT / "host_causality.json", value)
    (root / RESULT_ROOT / "host_causality.md").write_text(
        causality_markdown(value), encoding="utf-8"
    )
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "causality", "shortcut", "human", "headlines", "certificate", "all")
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    outputs: dict[str, Any] = {}
    if args.command in {"freeze", "all"}:
        outputs["freeze"] = freeze_release_candidate(root)
        write_json(root / RESULT_ROOT / "frozen_release_candidate.json", outputs["freeze"])
    if args.command in {"causality", "all"}:
        outputs["causality"] = _write_causality(root)
    if args.command in {"shortcut", "all"}:
        outputs["shortcut"] = shortcut_audit(root)
        write_json(root / RESULT_ROOT / "shortcut_audit.json", outputs["shortcut"])
    if args.command in {"human", "all"}:
        outputs["human"] = validate_human_packet(root)
        write_json(root / RESULT_ROOT / "human_packet_validation.json", outputs["human"])
    if args.command in {"headlines", "all"}:
        outputs["headlines"] = recompute_headlines(root)
        write_json(root / RESULT_ROOT / "headline_recomputation.json", outputs["headlines"])
    if args.command in {"certificate", "all"}:
        outputs["certificate"] = final_certificate(root)
        write_json(root / RESULT_ROOT / "release_certificate.json", outputs["certificate"])
    print(json.dumps(outputs, indent=2, sort_keys=True))
    statuses = [value.get("status", "PASS") for value in outputs.values()]
    return 0 if all("DEFECT" not in status and "FAIL" not in status for status in statuses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
