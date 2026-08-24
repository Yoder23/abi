"""Freeze initial host certifications and pre-receiver source-success locks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json_bytes, sha256_bytes

HOSTS = ("layercake", "qwen2", "pythia")
DOMAINS = ("python", "chemistry", "civics")


class FreezeError(RuntimeError):
    """Raised when preregistered evidence cannot be frozen safely."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreezeError(f"expected object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FreezeError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    certification_root = root / "results/abi_v2/host_certification/initial"
    adapters: dict[str, Any] = {}
    certifications: dict[str, Any] = {}
    for host in HOSTS:
        result_path = certification_root / host / "result.json"
        adapter_path = certification_root / host / "adapter.json"
        result, adapter = _json(result_path), _json(adapter_path)
        if (
            result.get("status") != "PASS_CAPABILITY_BLIND_HOST_CERTIFICATION"
            or not all(result.get("gates", {}).values())
            or result["adapter"]["sha256"] != _sha256(adapter_path)
            or adapter.get("frozen") is not True
            or adapter.get("trainable_parameters") != 0
            or adapter.get("optimizer_steps") != 0
            or adapter.get("capability_examples_seen") != 0
            or adapter.get("capability_outputs_seen") != 0
            or adapter.get("capability_success_ids_seen") != 0
        ):
            raise FreezeError(f"host certification is not releasably frozen: {host}")
        serialized = canonical_json_bytes(adapter).decode("utf-8").casefold()
        if any(domain in serialized for domain in DOMAINS):
            raise FreezeError(f"capability-specific adapter content found: {host}")
        adapters[host] = {
            "path": adapter_path.relative_to(root).as_posix(),
            "bytes": adapter_path.stat().st_size,
            "sha256": _sha256(adapter_path),
            "trainable_parameters": 0,
            "optimizer_steps": 0,
            "frozen_before_capability_reveal": True,
        }
        certifications[host] = {
            "path": result_path.relative_to(root).as_posix(),
            "sha256": _sha256(result_path),
            "evidence_sha256": result["evidence_sha256"],
            "status": result["status"],
            "adapter_overhead_fraction": result["performance"]["overhead_fraction"],
            "certification_examples": result["certification_data"]["examples"],
            "model_visible_units": result["certification_data"]["model_visible_units"],
        }

    decision = {
        "format": "abi-v2-initial-host-certification-decision/1",
        "status": "PASS_ALL_INITIAL_HOST_CERTIFICATIONS_CAPABILITY_REVEAL_AUTHORIZED",
        "hosts": certifications,
        "adapters": adapters,
        "initial_implementation_outcome": "PASS_WITHOUT_BOUNDED_REPAIR",
        "bounded_repairs_consumed": 0,
        "bounded_repairs_remaining": 1,
        "capability_reveal_occurred_before_this_lock": False,
        "post_freeze_adapter_training_allowed": False,
        "post_freeze_adapter_calibration_allowed": False,
        "post_freeze_adapter_mutation_allowed": False,
    }
    decision["evidence_sha256"] = sha256_bytes(canonical_json_bytes(decision))
    decision_path = root / "results/abi_v2/host_certification/initial_decision.json"
    adapter_manifest_path = root / "results/abi_v2/adapters/manifest.json"
    _write_once(decision_path, decision)
    adapter_manifest = {
        "format": "abi-v2-frozen-host-adapter-manifest/1",
        "status": "FROZEN_BEFORE_CAPABILITY_REVEAL",
        "initial_decision_path": decision_path.relative_to(root).as_posix(),
        "initial_decision_sha256": _sha256(decision_path),
        "adapters": adapters,
        "same_adapter_required_across_all_four_capabilities": True,
        "capability_specific_parameters_forbidden": True,
    }
    adapter_manifest["evidence_sha256"] = sha256_bytes(
        canonical_json_bytes(adapter_manifest)
    )
    _write_once(adapter_manifest_path, adapter_manifest)

    english_lock_path = root / "results/abi_final_mile/abi-release/source-success-lock.json"
    english_lock = _json(english_lock_path)
    english_ids = [str(value) for value in english_lock["successful_task_ids"]]
    if len(english_ids) != 1381 or len(english_ids) != len(set(english_ids)):
        raise FreezeError("English source-success lock changed")
    domain_observations_path = (
        root
        / "results/abi_capability_compiler_phase6_composition/run_v1032/seed104729/observations.jsonl"
    )
    selected_rows = [
        row
        for row in _read_jsonl(domain_observations_path)
        if row.get("mode") == "composed_host_selected_domain"
    ]
    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        rows = [row for row in selected_rows if row.get("domain") == domain]
        ids = [str(row["probe_id"]) for row in rows if row.get("functional_pass") is True]
        if len(rows) != 100 or len(ids) != 100 or len(ids) != len(set(ids)):
            raise FreezeError(f"domain source-success depth changed: {domain}")
        outputs = {str(row["probe_id"]): str(row["output"]) for row in rows}
        domains[domain] = {
            "source_host": "layercake-v25-seed104729",
            "success_rule": "functional_pass == true",
            "successful_task_ids": ids,
            "successful_task_count": len(ids),
            "source_outputs_sha256": sha256_bytes(canonical_json_bytes(outputs)),
        }
    locks = {
        "format": "abi-v2-all-capability-source-success-lock/1",
        "status": "FROZEN_AFTER_HOST_CERTIFICATION_AND_BEFORE_RECEIVER_MATRIX",
        "required_receiver_retention": 1.0,
        "english": {
            "source_lock_path": english_lock_path.relative_to(root).as_posix(),
            "source_lock_sha256": _sha256(english_lock_path),
            "successful_task_ids": english_ids,
            "successful_task_count": len(english_ids),
            "success_rule": english_lock["success_rule"],
        },
        "domains": domains,
        "domain_source_observations": {
            "path": domain_observations_path.relative_to(root).as_posix(),
            "sha256": _sha256(domain_observations_path),
        },
        "receiver_outputs_seen_before_lock": False,
        "lock_mutation_allowed": False,
    }
    locks["evidence_sha256"] = sha256_bytes(canonical_json_bytes(locks))
    lock_path = root / "results/abi_v2/semantic_retention/source_success_locks.json"
    _write_once(lock_path, locks)
    return {
        "initial_decision": decision_path.relative_to(root).as_posix(),
        "adapter_manifest": adapter_manifest_path.relative_to(root).as_posix(),
        "source_success_locks": lock_path.relative_to(root).as_posix(),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(json.dumps(freeze(Path.cwd()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
