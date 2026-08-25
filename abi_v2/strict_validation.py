"""Fail-closed recomputation for the repaired ABI final validation.

No scientific status/gate boolean produced by an experiment is accepted as
evidence.  This verifier derives claims from immutable files, raw observation
rows, hashes, counts, outputs, timings, and live failure records.  Missing,
extra, stale, or unrecomputable inputs are fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from abi.capability_compiler_phase2_common import evaluate_functional

from .canonical import canonical_json_bytes, sha256_bytes, verify_reference
from .capability_matrix import (
    CAPABILITIES,
    DOMAINS,
    FrozenHostAdapter,
    _matrix_records,
    _source_references,
)
from .final_validation import CAPABILITY_PATHS, HOSTS, MATRIX_DIRS, evidence_hash
from .live_causality import CONDITIONS, SAMPLE_SEED, _selected

EVIDENCE_ROOT = Path("results/abi_final_validation_v2")
CERTIFICATION_ROOT = EVIDENCE_ROOT / "isolated_certification_strict"
CAUSALITY_ROOT = EVIDENCE_ROOT / "live_causality"
ISOLATION_ROOT = EVIDENCE_ROOT / "live_isolation"
FORBIDDEN_CAPABILITY_SUFFIXES = {".abi", ".cake", ".abix", ".abicir"}


class StrictValidationError(RuntimeError):
    """Raised whenever a required claim cannot be independently recomputed."""


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise StrictValidationError(f"required file missing: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise StrictValidationError(f"required file unreadable: {path}") from exc
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrictValidationError(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise StrictValidationError(f"required JSON object changed: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise StrictValidationError(f"required JSONL unavailable: {path}") from exc
    if not lines:
        raise StrictValidationError(f"required JSONL is empty: {path}")
    rows = []
    for position, line in enumerate(lines):
        if not line.strip():
            raise StrictValidationError(f"blank raw row at {path}:{position + 1}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StrictValidationError(
                f"invalid raw row at {path}:{position + 1}"
            ) from exc
        if not isinstance(row, dict):
            raise StrictValidationError(f"non-object raw row at {path}:{position + 1}")
        rows.append(row)
    return rows


def read_text(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StrictValidationError(f"required text unavailable: {path}") from exc
    if not value:
        raise StrictValidationError(f"required text is empty: {path}")
    return value


def verify_evidence_hash(value: Mapping[str, Any], *, label: str) -> None:
    if value.get("evidence_sha256") != evidence_hash(value):
        raise StrictValidationError(f"stale or invalid evidence hash: {label}")


def _effective_mount(mountinfo: str, target: str) -> str:
    rows = [
        line
        for line in mountinfo.splitlines()
        if len(line.split()) > 4 and line.split()[4] == target
    ]
    return rows[-1] if rows else ""


def verify_certifications(
    root: Path, certification_root: Path | None = None
) -> dict[str, Any]:
    """Recompute certification solely from capsule and raw measurement bytes."""

    root = root.resolve()
    certification_root = (
        (root / CERTIFICATION_ROOT).resolve()
        if certification_root is None
        else certification_root.resolve()
    )
    adapter_manifest = read_json(root / "results/abi_v2/adapters/manifest.json")
    suite = read_json(root / "abi_v2/conformance_suite.json")
    spec = read_json(root / "abi_v2/canonical_spec.json")
    reference_path = root / suite["reference_vectors"]["path"]
    if sha256_file(reference_path) != suite["reference_vectors"]["sha256"]:
        raise StrictValidationError("generic reference-vector binding changed")
    reference = read_json(reference_path)
    records = reference.get("records")
    if not isinstance(records, list) or len(records) != suite["reference_vectors"]["records"]:
        raise StrictValidationError("generic reference-vector depth changed")
    for record in records:
        verify_reference(record)

    hosts: dict[str, Any] = {}
    for host in HOSTS:
        base = certification_root / host
        launcher = read_json(base / "launcher-receipt.json")
        receipt = read_json(base / "receipt.json")
        isolation = read_json(base / "physical-isolation.json")
        capsule = read_json(base / "certification-capsule-manifest.json")
        result = read_json(base / "certification/result.json")
        performance = read_json(base / "certification/performance.json")
        adapter_path = base / "certification/adapter.json"
        mount_path = base / "mountinfo.txt"
        mountinfo = read_text(mount_path)
        for label, value in (
            ("launcher", launcher),
            ("receipt", receipt),
            ("isolation", isolation),
            ("capsule", capsule),
            ("certification", result),
        ):
            verify_evidence_hash(value, label=f"{host}/{label}")
        bindings = {
            "result_sha256": base / "certification/result.json",
            "adapter_sha256": adapter_path,
            "performance_sha256": base / "certification/performance.json",
            "isolation_evidence_sha256": base / "physical-isolation.json",
            "capsule_manifest_sha256": base / "certification-capsule-manifest.json",
            "mountinfo_sha256": mount_path,
        }
        for field, path in bindings.items():
            if receipt.get(field) != sha256_file(path):
                raise StrictValidationError(f"certification binding changed: {host}/{field}")
        if launcher.get("launcher", {}).get("exit_code") != 0:
            raise StrictValidationError(f"isolated worker exit changed: {host}")

        capsule_files = capsule.get("files")
        if not isinstance(capsule_files, list) or not capsule_files:
            raise StrictValidationError(f"capsule inventory missing: {host}")
        capsule_by_path = {str(row.get("path")): row for row in capsule_files}
        if len(capsule_by_path) != len(capsule_files):
            raise StrictValidationError(f"duplicate capsule inventory path: {host}")
        if any(
            Path(path).suffix.casefold() in FORBIDDEN_CAPABILITY_SUFFIXES
            or "source_success" in path.casefold()
            for path in capsule_by_path
        ):
            raise StrictValidationError(f"forbidden payload entered certification: {host}")
        physical_inventory = isolation.get("capsule", {}).get("inventory")
        if not isinstance(physical_inventory, list):
            raise StrictValidationError(f"physical capsule inventory missing: {host}")
        physical_by_path = {str(row.get("path")): row for row in physical_inventory}
        expected_physical = {
            path: {
                "path": path,
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for path, row in capsule_by_path.items()
        }
        if physical_by_path != expected_physical:
            raise StrictValidationError(f"physical capsule bytes differ from manifest: {host}")
        effective = _effective_mount(mountinfo, "/mnt/c")
        if effective and " - tmpfs " not in effective:
            raise StrictValidationError(f"development drive was visible to worker: {host}")
        if isolation.get("mount", {}).get("mountinfo_sha256") != sha256_file(mount_path):
            raise StrictValidationError(f"raw mount table binding changed: {host}")
        expected_adapter = adapter_manifest["adapters"][host]["sha256"]
        if sha256_file(adapter_path) != expected_adapter:
            raise StrictValidationError(f"adapter bytes changed: {host}")
        adapter = read_json(adapter_path)
        if (
            int(adapter.get("trainable_parameters", -1)) != 0
            or int(adapter.get("optimizer_steps", -1)) != 0
        ):
            raise StrictValidationError(f"adapter structure changed: {host}")
        if result.get("canonical_spec_sha256") != sha256_file(root / "abi_v2/canonical_spec.json"):
            raise StrictValidationError(f"canonical spec binding changed: {host}")
        if result.get("conformance_suite_sha256") != sha256_file(
            root / "abi_v2/conformance_suite.json"
        ):
            raise StrictValidationError(f"conformance suite binding changed: {host}")
        if result.get("reference_implementation_sha256") != sha256_file(
            root / "abi_v2/canonical.py"
        ):
            raise StrictValidationError(f"reference implementation binding changed: {host}")
        if result.get("physical_isolation", {}).get("evidence_sha256") != isolation.get(
            "evidence_sha256"
        ):
            raise StrictValidationError(f"certification/isolation binding changed: {host}")

        checks = result.get("checks")
        if not isinstance(checks, dict):
            raise StrictValidationError(f"raw certification checks missing: {host}")
        roundtrips = checks.get("roundtrip_rows")
        if not isinstance(roundtrips, list) or len(roundtrips) != int(
            result["certification_data"]["examples"]
        ):
            raise StrictValidationError(f"roundtrip raw rows missing: {host}")
        if any(
            row.get("input_utf8_sha256") != row.get("decoded_utf8_sha256")
            or int(row.get("input_utf8_bytes", -1)) != int(row.get("decoded_utf8_bytes", -2))
            for row in roundtrips
        ):
            raise StrictValidationError(f"native roundtrip changed bytes: {host}")
        if result["certification_data"].get("example_sha256") != [
            row["input_utf8_sha256"] for row in roundtrips
        ]:
            raise StrictValidationError(f"certification corpus hashes changed: {host}")
        forwards = checks.get("native_forward_rows")
        if not isinstance(forwards, list):
            raise StrictValidationError(f"native forward raw rows missing: {host}")
        if len(forwards) != int(checks.get("native_forward_records", -1)):
            raise StrictValidationError(f"native forward depth changed: {host}")
        if any(
            int(row.get("finite_values", -1)) != int(row.get("total_values", -2))
            for row in forwards
        ):
            raise StrictValidationError(f"non-finite native forward: {host}")
        if checks.get("native_argmax_id_hashes", []) != [
            row["argmax_id_sha256"] for row in forwards
        ]:
            raise StrictValidationError(f"native forward hashes changed: {host}")

        alone = [float(value) for value in performance.get("host_alone_seconds", [])]
        adapted = [float(value) for value in performance.get("host_plus_idle_adapter_seconds", [])]
        minimum = int(spec["performance_gate"]["minimum_repeated_observations"])
        if len(alone) != len(adapted) or len(alone) < minimum or any(
            not math.isfinite(value) or value <= 0 for value in (*alone, *adapted)
        ):
            raise StrictValidationError(f"performance raw rows incomplete: {host}")
        baseline = statistics.median(alone)
        with_adapter = statistics.median(adapted)
        overhead = with_adapter / baseline - 1.0
        threshold = float(spec["performance_gate"]["maximum_overhead_fraction"])
        if overhead > threshold:
            raise StrictValidationError(f"adapter overhead failed: {host}={overhead}")
        hosts[host] = {
            "capsule_files": len(capsule_files),
            "roundtrip_rows": len(roundtrips),
            "native_forward_rows": len(forwards),
            "adapter_sha256": expected_adapter,
            "performance_observations": len(alone),
            "overhead_fraction": overhead,
        }
    return {
        "hosts": hosts,
        "hosts_verified": len(hosts),
        "physical_capability_archives_present": 0,
        "physical_source_success_ledgers_present": 0,
    }


def verify_locked_matrix_rows(root: Path, matrix_root: Path | None = None) -> dict[str, Any]:
    """Recompute the full quality/retention matrix without trusting result flags."""

    root = root.resolve()
    matrix_root = matrix_root.resolve() if matrix_root is not None else None
    protocol = read_json(root / "abi_v2/matrix_protocol_amendment3.json")
    base = read_json(root / protocol["base_protocol"])
    merged = {**base, **protocol}
    locks = read_json(root / merged["source_success_locks"])
    english_records, domain_records = _matrix_records(root, locks)
    record_maps = {
        "english": {str(row["probe_id"]): row for row in english_records},
        **{
            domain: {str(row["probe_id"]): row for row in domain_records[domain]}
            for domain in DOMAINS
        },
    }
    english_reference, domain_reference = _source_references(root)
    references = {"english": english_reference, **domain_reference}
    expected_keys = {
        (capability, probe_id)
        for capability, records in record_maps.items()
        for probe_id in records
    }
    by_host: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for host in HOSTS:
        path = (
            matrix_root / host / "observations.jsonl"
            if matrix_root is not None
            else root
            / f"results/abi_v2/capability_matrix/{MATRIX_DIRS[host]}/observations.jsonl"
        )
        rows = read_jsonl(path)
        index = {
            (str(row.get("capability")), str(row.get("probe_id"))): row for row in rows
        }
        if len(index) != len(rows) or set(index) != expected_keys:
            raise StrictValidationError(f"full raw matrix row set changed: {host}")
        for key, row in index.items():
            capability, probe_id = key
            output = str(row.get("output"))
            if hashlib.sha256(output.encode("utf-8")).hexdigest() != row.get("output_sha256"):
                raise StrictValidationError(f"matrix output hash changed: {host}/{key}")
            computed_functional = bool(
                evaluate_functional(output, record_maps[capability][probe_id]["evaluator"])
            )
            if not computed_functional:
                raise StrictValidationError(f"matrix functional failure: {host}/{key}")
            if output != references[capability][probe_id]:
                raise StrictValidationError(f"matrix source byte mismatch: {host}/{key}")
            actions = [int(value) for value in row.get("actions", [])]
            if capability != "english" and row.get("actions_sha256") != sha256_bytes(
                canonical_json_bytes(actions)
            ):
                raise StrictValidationError(f"matrix action hash changed: {host}/{key}")
        by_host[host] = index
    cross_output = 0
    cross_actions = 0
    specialist = [key for key in expected_keys if key[0] != "english"]
    for key in expected_keys:
        if len({str(by_host[host][key]["output"]) for host in HOSTS}) != 1:
            raise StrictValidationError(f"cross-host output mismatch: {key}")
        cross_output += 1
    for key in specialist:
        if len(
            {
                tuple(int(value) for value in by_host[host][key].get("actions", []))
                for host in HOSTS
            }
        ) != 1:
            raise StrictValidationError(f"cross-host specialist action mismatch: {key}")
        cross_actions += 1
    return {
        "hosts": len(HOSTS),
        "capabilities": len(CAPABILITIES),
        "rows_per_host": len(expected_keys),
        "rows_verified": len(expected_keys) * len(HOSTS),
        "cross_host_outputs_equal": cross_output,
        "cross_host_specialist_actions_equal": cross_actions,
    }


def verify_live_causality(
    root: Path, causality_root: Path | None = None
) -> dict[str, Any]:
    """Derive causal results from new live raw rows, never prior matrix outputs."""

    root = root.resolve()
    causality_root = (
        (root / CAUSALITY_ROOT).resolve()
        if causality_root is None
        else causality_root.resolve()
    )
    source_path = root / "abi_v2/live_causality.py"
    source_text = read_text(source_path)
    for forbidden in ("_matrix_rows", "_matrix_result", "_source_references"):
        if forbidden in source_text:
            raise StrictValidationError(f"live causality source reads replay evidence: {forbidden}")
    protocol = read_json(root / "abi_v2/matrix_protocol_amendment3.json")
    base = read_json(root / protocol["base_protocol"])
    merged = {**base, **protocol}
    locks = read_json(root / merged["source_success_locks"])
    adapters = read_json(root / merged["adapter_manifest"])["adapters"]
    english_records, domain_records = _matrix_records(root, locks)
    expected_selected = _selected(english_records, domain_records, 32)
    expected_ids = {
        capability: [str(row["probe_id"]) for row in expected_selected[capability]]
        for capability in CAPABILITIES
    }
    current_packages = {
        capability: sha256_file(root / path)
        for capability, path in CAPABILITY_PATHS.items()
    }
    state_channel_supported = "host_state" in inspect.signature(
        FrozenHostAdapter.realize
    ).parameters
    if state_channel_supported:
        raise StrictValidationError("frozen adapter unexpectedly accepts host semantic state")
    all_rows: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    by_host: dict[str, Any] = {}
    positive_conditions = CONDITIONS[:6]
    for host in HOSTS:
        base_path = causality_root / host
        manifest = read_json(base_path / "manifest.json")
        verify_evidence_hash(manifest, label=f"causality/{host}/manifest")
        rows = read_jsonl(base_path / "observations.jsonl")
        if manifest.get("observations_sha256") != sha256_file(
            base_path / "observations.jsonl"
        ):
            raise StrictValidationError(f"live causality raw binding changed: {host}")
        if manifest.get("execution_source_sha256") != sha256_file(source_path):
            raise StrictValidationError(f"stale live causality code binding: {host}")
        if manifest.get("sample_seed") != SAMPLE_SEED or manifest.get(
            "samples_per_capability"
        ) != 32:
            raise StrictValidationError(f"live causal selection changed: {host}")
        if manifest.get("selected_probe_ids") != expected_ids:
            raise StrictValidationError(f"live causal selected IDs changed: {host}")
        if manifest.get("conditions") != list(CONDITIONS):
            raise StrictValidationError(f"live causal interventions changed: {host}")
        if manifest.get("adapter_sha256_before") != adapters[host]["sha256"] or manifest.get(
            "adapter_sha256_after"
        ) != adapters[host]["sha256"]:
            raise StrictValidationError(f"causal adapter changed: {host}")
        if manifest.get("capability_sha256") != current_packages:
            raise StrictValidationError(f"causal capability bytes changed: {host}")
        expected_count = len(CAPABILITIES) * 32 * len(CONDITIONS)
        if len(rows) != expected_count or manifest.get("observations_rows") != expected_count:
            raise StrictValidationError(f"live causal raw row depth changed: {host}")
        index = {
            (str(row.get("condition")), str(row.get("capability")), str(row.get("probe_id"))): row
            for row in rows
        }
        if len(index) != len(rows):
            raise StrictValidationError(f"duplicate live causal row: {host}")
        expected_keys = {
            (condition, capability, probe_id)
            for condition in CONDITIONS
            for capability, ids in expected_ids.items()
            for probe_id in ids
        }
        if set(index) != expected_keys:
            raise StrictValidationError(f"live causal row set changed: {host}")

        for capability, ids in expected_ids.items():
            for probe_id in ids:
                real = index[("real_host", capability, probe_id)]
                real_state = [float(value) for value in real.get("state_vector", [])]
                if not real_state:
                    raise StrictValidationError(f"real causal state invalid: {host}/{capability}/{probe_id}")
                for condition in positive_conditions:
                    row = index[(condition, capability, probe_id)]
                    output = row.get("capability_output")
                    realized = row.get("realized_output")
                    if not isinstance(output, str) or not isinstance(realized, str):
                        raise StrictValidationError(
                            f"live positive execution missing: {host}/{condition}/{capability}/{probe_id}"
                        )
                    if hashlib.sha256(output.encode("utf-8")).hexdigest() != row.get(
                        "capability_output_sha256"
                    ):
                        raise StrictValidationError(f"live capability output hash changed: {host}")
                    if hashlib.sha256(realized.encode("utf-8")).hexdigest() != row.get(
                        "realized_output_sha256"
                    ):
                        raise StrictValidationError(f"live realized output hash changed: {host}")
                    state = [float(value) for value in row.get("state_vector", [])]
                    if row.get("state_sha256") != sha256_bytes(canonical_json_bytes(state)):
                        raise StrictValidationError(f"live state hash changed: {host}")
                    if row.get("actions_sha256") != sha256_bytes(
                        canonical_json_bytes([int(value) for value in row.get("actions", [])])
                    ):
                        raise StrictValidationError(f"live action hash changed: {host}")
                    if realized != real["realized_output"] or output != real["capability_output"]:
                        raise StrictValidationError(
                            f"host-state intervention changed output: {host}/{condition}/{capability}/{probe_id}"
                        )
                    if condition == "neutral_host" and any(value != 0.5 for value in state):
                        raise StrictValidationError(f"neutral state malformed: {host}")
                    if condition == "zero_state" and any(value != 0.0 for value in state):
                        raise StrictValidationError(f"zero state malformed: {host}")
                    if condition == "random_state" and (state == real_state or not state):
                        raise StrictValidationError(f"random state malformed: {host}")
                    if condition == "shuffled_state" and sorted(state) != sorted(real_state):
                        raise StrictValidationError(f"shuffled state malformed: {host}")
                    if condition == "host_removed" and state:
                        raise StrictValidationError(f"host-removed state present: {host}")
                adapter_removed = index[("adapter_removed", capability, probe_id)]
                if (
                    not isinstance(adapter_removed.get("capability_output"), str)
                    or adapter_removed.get("realized_output") is not None
                    or not adapter_removed.get("exception_type")
                ):
                    raise StrictValidationError(f"adapter removal did not fail live: {host}")
                if adapter_removed["capability_output"] != real["capability_output"]:
                    raise StrictValidationError(f"adapter-removal generation was not live: {host}")
                capability_removed = index[("capability_removed", capability, probe_id)]
                if (
                    capability_removed.get("capability_output") is not None
                    or capability_removed.get("realized_output") is not None
                    or not capability_removed.get("exception_type")
                ):
                    raise StrictValidationError(f"capability removal did not fail live: {host}")
        all_rows[host] = index
        by_host[host] = {
            "raw_rows": len(rows),
            "live_positive_executions": len(CAPABILITIES) * 32 * len(positive_conditions),
            "live_adapter_removals": len(CAPABILITIES) * 32,
            "live_capability_removals": len(CAPABILITIES) * 32,
        }

    cross_host = 0
    for capability, ids in expected_ids.items():
        for probe_id in ids:
            values = {
                all_rows[host][("real_host", capability, probe_id)]["realized_output"]
                for host in HOSTS
            }
            if len(values) != 1:
                raise StrictValidationError(
                    f"new live real-host outputs differ: {capability}/{probe_id}"
                )
            cross_host += 1
    return {
        "hosts": by_host,
        "raw_rows": sum(row["raw_rows"] for row in by_host.values()),
        "cross_host_real_outputs_equal": cross_host,
        "state_channel_supported": state_channel_supported,
        "causal_conclusion": (
            "The new live executions confirm that the frozen capability runtime is semantically "
            "standalone: real, neutral, zero, random, shuffled, and removed host states produce "
            "identical capability outputs because the frozen adapter exposes no host-state channel. "
            "Removing the adapter fails realization, while removing a capability fails generation."
        ),
    }


def verify_live_isolation(
    root: Path, isolation_root: Path | None = None
) -> dict[str, Any]:
    """Re-evaluate fresh isolation outputs against the frozen functional evaluators."""

    root = root.resolve()
    isolation_root = (
        (root / ISOLATION_ROOT).resolve()
        if isolation_root is None
        else isolation_root.resolve()
    )
    source_path = root / "abi_v2/live_isolation.py"
    source_text = read_text(source_path)
    for forbidden in ("_source_references", "_matrix_rows", "_matrix_result"):
        if forbidden in source_text:
            raise StrictValidationError(f"live isolation reads evaluator answers: {forbidden}")
    amendment = read_json(root / "abi_v2/matrix_protocol_amendment3.json")
    base_protocol = read_json(root / amendment["base_protocol"])
    protocol = {**base_protocol, **amendment}
    locks = read_json(root / protocol["source_success_locks"])
    adapters = read_json(root / protocol["adapter_manifest"])["adapters"]
    english_records, domain_records = _matrix_records(root, locks)
    english_map = {str(row["probe_id"]): row for row in english_records[:100]}
    domain_maps = {
        domain: {str(row["probe_id"]): row for row in domain_records[domain]}
        for domain in DOMAINS
    }
    expected_keys = set()
    for domain in DOMAINS:
        expected_keys.update(
            ("english_only_specialist_target", domain, probe_id)
            for probe_id in domain_maps[domain]
        )
        expected_keys.update(
            ("wrong_specialist_capability", domain, probe_id)
            for probe_id in domain_maps[domain]
        )
    expected_keys.update(
        ("wrong_specialist_on_english", "english", probe_id)
        for probe_id in english_map
    )
    by_host_rows: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    hosts = {}
    for host in HOSTS:
        base_path = isolation_root / host
        manifest = read_json(base_path / "manifest.json")
        verify_evidence_hash(manifest, label=f"isolation/{host}/manifest")
        rows_path = base_path / "observations.jsonl"
        rows = read_jsonl(rows_path)
        if manifest.get("execution_source_sha256") != sha256_file(source_path):
            raise StrictValidationError(f"stale live isolation source: {host}")
        if manifest.get("observations_sha256") != sha256_file(rows_path):
            raise StrictValidationError(f"live isolation raw binding changed: {host}")
        if manifest.get("adapter_sha256_before") != adapters[host]["sha256"] or manifest.get(
            "adapter_sha256_after"
        ) != adapters[host]["sha256"]:
            raise StrictValidationError(f"live isolation adapter changed: {host}")
        if manifest.get("english_archive_sha256") != sha256_file(
            root / CAPABILITY_PATHS["english"]
        ):
            raise StrictValidationError(f"live isolation English package changed: {host}")
        if manifest.get("domain_archive_sha256") != {
            domain: sha256_file(root / CAPABILITY_PATHS[domain]) for domain in DOMAINS
        }:
            raise StrictValidationError(f"live isolation domain package changed: {host}")
        index = {
            (str(row.get("mode")), str(row.get("target_capability")), str(row.get("probe_id"))): row
            for row in rows
        }
        if len(rows) != 700 or manifest.get("observations_rows") != 700:
            raise StrictValidationError(f"live isolation row depth changed: {host}")
        if len(index) != len(rows) or set(index) != expected_keys:
            raise StrictValidationError(f"live isolation row set changed: {host}")
        successes = 0
        for key, row in index.items():
            mode, target, probe_id = key
            output = row.get("output")
            if not isinstance(output, str) or hashlib.sha256(output.encode("utf-8")).hexdigest() != row.get(
                "output_sha256"
            ):
                raise StrictValidationError(f"live isolation output hash changed: {host}/{key}")
            record = english_map[probe_id] if target == "english" else domain_maps[target][probe_id]
            if hashlib.sha256(str(record["prompt"]).encode("utf-8")).hexdigest() != row.get(
                "prompt_sha256"
            ):
                raise StrictValidationError(f"live isolation prompt binding changed: {host}/{key}")
            actions = [int(value) for value in row.get("actions", [])]
            if mode != "english_only_specialist_target" and row.get(
                "actions_sha256"
            ) != sha256_bytes(canonical_json_bytes(actions)):
                raise StrictValidationError(f"live isolation action hash changed: {host}/{key}")
            successes += int(bool(evaluate_functional(output, record["evaluator"])))
        if successes != 0:
            raise StrictValidationError(f"live capability isolation failed: {host}={successes}/700")
        by_host_rows[host] = index
        hosts[host] = {"raw_rows": len(rows), "target_successes": successes}
    cross_host = 0
    for key in expected_keys:
        if len({by_host_rows[host][key]["output"] for host in HOSTS}) != 1:
            raise StrictValidationError(f"cross-host isolation output changed: {key}")
        cross_host += 1
    return {
        "hosts": hosts,
        "raw_rows": 700 * len(HOSTS),
        "target_successes": 0,
        "cross_host_outputs_equal": cross_host,
    }


def required_input_manifest(root: Path) -> dict[str, Any]:
    """Bind the strict certificate to every file used to derive its claims."""

    root = root.resolve()
    paths = {
        root / "abi/capability_compiler_phase2_common.py",
        root / "abi_v2/canonical.py",
        root / "abi_v2/canonical_spec.json",
        root / "abi_v2/capability_matrix.py",
        root / "abi_v2/conformance_suite.json",
        root / "abi_v2/final_validation.py",
        root / "abi_v2/host_certification.py",
        root / "abi_v2/isolated_certification.py",
        root / "abi_v2/live_causality.py",
        root / "abi_v2/live_isolation.py",
        root / "abi_v2/matrix_protocol.json",
        root / "abi_v2/matrix_protocol_amendment3.json",
        root / "abi_v2/strict_validation.py",
        root / "results/abi_v2/adapters/manifest.json",
    }
    protocol = read_json(root / "abi_v2/matrix_protocol_amendment3.json")
    base = read_json(root / protocol["base_protocol"])
    merged = {**base, **protocol}
    for field in ("source_success_locks", "adapter_manifest"):
        paths.add(root / str(merged[field]))
    suite = read_json(root / "abi_v2/conformance_suite.json")
    paths.add(root / str(suite["reference_vectors"]["path"]))
    paths.update(root / path for path in CAPABILITY_PATHS.values())
    for host in HOSTS:
        cert = root / CERTIFICATION_ROOT / host
        paths.update(
            {
                cert / "launcher-receipt.json",
                cert / "receipt.json",
                cert / "physical-isolation.json",
                cert / "certification-capsule-manifest.json",
                cert / "mountinfo.txt",
                cert / "certification/result.json",
                cert / "certification/performance.json",
                cert / "certification/adapter.json",
                root
                / f"results/abi_v2/capability_matrix/{MATRIX_DIRS[host]}/observations.jsonl",
                root / CAUSALITY_ROOT / host / "manifest.json",
                root / CAUSALITY_ROOT / host / "observations.jsonl",
                root / ISOLATION_ROOT / host / "manifest.json",
                root / ISOLATION_ROOT / host / "observations.jsonl",
            }
        )
    files = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise StrictValidationError(f"required input escaped release root: {path}") from exc
        if not path.is_file():
            raise StrictValidationError(f"required file missing: {path}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise StrictValidationError(f"required file unreadable: {path}") from exc
        files.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "files": files,
        "file_count": len(files),
        "aggregate_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def verify(root: Path) -> dict[str, Any]:
    certification = verify_certifications(root)
    matrix = verify_locked_matrix_rows(root)
    causality = verify_live_causality(root)
    isolation = verify_live_isolation(root)
    return {
        "format": "abi-v2-strict-final-validation/1",
        "status": "PASS_STRICT_RAW_RECOMPUTATION",
        "certification": certification,
        "locked_matrix": matrix,
        "live_causality": causality,
        "live_isolation": isolation,
        "required_inputs": required_input_manifest(root),
        "trusted_scientific_booleans_consumed": 0,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        value = verify(Path(args.root))
    except StrictValidationError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    value["evidence_sha256"] = evidence_hash(value)
    if args.output:
        path = Path(args.root).resolve() / args.output
        if path.exists():
            raise StrictValidationError(f"immutable strict verifier output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
