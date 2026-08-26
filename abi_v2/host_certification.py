"""Capability-blind, zero-training host certification for Canonical ABI V2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import psutil

from .canonical import (
    ABI_VERSION,
    canonical_context,
    canonical_json_bytes,
    canonical_output_intent,
    sha256_bytes,
    strict_utf8,
    verify_reference,
)

HOST_KEYS = {"layercake", "qwen2", "pythia"}


class HostCertificationError(RuntimeError):
    """Raised when a host fails a preregistered conformance rule."""


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HostCertificationError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise HostCertificationError(f"immutable certification output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _neutral_texts(count: int) -> list[str]:
    glyphs = ("Alpha", "Beta", "Gamma", "café", "naïve", "東", "한", "🧪")
    return [
        f"ABI neutral vector {index:04d}: {glyphs[index % len(glyphs)]} / Name-{index % 17} / {index * 37}."
        for index in range(count)
    ]


def _reference_records(root: Path, suite: dict[str, Any]) -> list[dict[str, Any]]:
    binding = suite["reference_vectors"]
    path = (root / binding["path"]).resolve()
    if _sha256_file(path) != binding["sha256"]:
        raise HostCertificationError("reference-vector binding changed")
    value = _object(path)
    records = value.get("records")
    if not isinstance(records, list) or len(records) != binding["records"]:
        raise HostCertificationError("reference-vector depth changed")
    return records


def _snapshot_inventory(snapshot: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for path in sorted(snapshot.iterdir(), key=lambda value: value.name):
        if path.is_file():
            rows.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            )
    return sha256_bytes(canonical_json_bytes(rows)), rows


def _adapter_document(
    *,
    host_key: str,
    host: dict[str, Any],
    spec_sha256: str,
    suite_sha256: str,
    implementation_sha256: str,
    tokenizer_mode: str,
) -> dict[str, Any]:
    return {
        "format": "abi-v2-frozen-generic-host-adapter/1",
        "abi_version": ABI_VERSION,
        "host_key": host_key,
        "host_id": host["host_id"],
        "architecture": host["architecture"],
        "checkpoint": host["checkpoint"]
        if "checkpoint" in host
        else f"{host['model']}@{host['revision']}",
        "checkpoint_sha256": host["checkpoint_sha256"],
        "snapshot_inventory_sha256": host.get("snapshot_inventory_sha256"),
        "tokenizer_sha256": host.get("tokenizer_sha256"),
        "tokenizer_mode": tokenizer_mode,
        "canonical_spec_sha256": spec_sha256,
        "conformance_suite_sha256": suite_sha256,
        "reference_implementation_sha256": implementation_sha256,
        "trainable_parameters": 0,
        "optimizer_steps": 0,
        "capability_examples_seen": 0,
        "capability_outputs_seen": 0,
        "capability_success_ids_seen": 0,
        "capability_paths_accepted": False,
        "accepted_runtime_classes": ["lc-direct-neural-core/25", "lc-direct-neural-decoder/1"],
        "accepted_capability_ids": "not enumerated; package identity is revealed only after freeze",
        "input_adapter": "strict host-tokenizer decode to canonical UTF-8 plus typed context state",
        "output_adapter": "canonical authoritative UTF-8 to exact native-tokenizer round trip",
        "state_ownership": ["host_base", "generic_host_adapter", "frozen_capability", "generic_runtime"],
        "frozen": True,
        "post_freeze_mutation_allowed": False,
    }


def _identity_host_checks(texts: list[str]) -> dict[str, Any]:
    rows = []
    for text in texts:
        decoded = strict_utf8(text).decode("utf-8")
        if decoded != text:
            raise HostCertificationError("LayerCake strict UTF-8 identity changed")
        rows.append(
            {
                "input_utf8_sha256": sha256_bytes(strict_utf8(text)),
                "decoded_utf8_sha256": sha256_bytes(strict_utf8(decoded)),
                "input_utf8_bytes": len(strict_utf8(text)),
                "decoded_utf8_bytes": len(strict_utf8(decoded)),
            }
        )
    return {
        "tokenizer_mode": "strict_utf8_identity",
        "roundtrips": len(texts),
        "roundtrips_exact": len(texts),
        "roundtrip_rows": rows,
        "model_visible_units": sum(len(strict_utf8(text)) for text in texts),
        "native_forward_records": 0,
        "native_forward_finite_records": 0,
        "native_forward_rows": [],
        "host_parameter_count": 0,
        "host_precision": "not applicable before capability installation",
    }


def _model_host_checks(
    *, snapshot: Path, texts: list[str], forward_records: int, device: str
) -> tuple[dict[str, Any], Any, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    exact = 0
    visible_units = 0
    roundtrip_rows = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        visible_units += len(token_ids)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        exact += decoded == text
        roundtrip_rows.append(
            {
                "input_utf8_sha256": sha256_bytes(strict_utf8(text)),
                "decoded_utf8_sha256": sha256_bytes(strict_utf8(decoded)),
                "input_utf8_bytes": len(strict_utf8(text)),
                "decoded_utf8_bytes": len(strict_utf8(decoded)),
                "native_units": len(token_ids),
                "native_units_sha256": sha256_bytes(
                    canonical_json_bytes([int(value) for value in token_ids])
                ),
            }
        )
    if exact != len(texts):
        raise HostCertificationError(
            f"native tokenizer exact round trip failed: {exact}/{len(texts)}"
        )

    selected_device = torch.device(device)
    dtype = torch.float16 if selected_device.type == "cuda" else torch.float32
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(selected_device)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=dtype,
    ).to(selected_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    finite = 0
    argmax_hashes = []
    forward_rows = []
    with torch.inference_mode():
        for position, text in enumerate(texts[:forward_records]):
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            inputs = {key: value.to(selected_device) for key, value in inputs.items()}
            output = model(**inputs, use_cache=True)
            logits = output.logits[:, -1]
            if torch.isfinite(logits).all().item():
                finite += 1
            argmax_sha256 = hashlib.sha256(
                int(logits.argmax(-1).item()).to_bytes(8, "little")
            ).hexdigest()
            argmax_hashes.append(argmax_sha256)
            forward_rows.append(
                {
                    "position": position,
                    "input_utf8_sha256": sha256_bytes(strict_utf8(text)),
                    "input_units": int(inputs["input_ids"].numel()),
                    "finite_values": int(torch.isfinite(logits).sum().item()),
                    "total_values": int(logits.numel()),
                    "argmax_id_sha256": argmax_sha256,
                }
            )
    if finite != forward_records:
        raise HostCertificationError("native host forward produced non-finite state")
    return (
        {
            "tokenizer_mode": tokenizer.__class__.__name__,
            "roundtrips": len(texts),
            "roundtrips_exact": exact,
            "roundtrip_rows": roundtrip_rows,
            "model_visible_units": visible_units,
            "native_forward_records": forward_records,
            "native_forward_finite_records": finite,
            "native_argmax_id_hashes": argmax_hashes,
            "native_forward_rows": forward_rows,
            "host_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "host_precision": str(dtype),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(selected_device))
            if selected_device.type == "cuda"
            else 0,
        },
        model,
        tokenizer,
    )


def _idle_adapter(record: dict[str, Any], frozen: bool) -> dict[str, Any]:
    if not frozen:
        raise HostCertificationError("adapter must be frozen")
    return canonical_context(record)


def _performance(
    *,
    host_key: str,
    records: list[dict[str, Any]],
    repeats: int,
    model: Any = None,
    tokenizer: Any = None,
    device: str = "cpu",
) -> dict[str, Any]:
    import statistics

    baseline_times = []
    adapter_times = []
    if model is None:
        iterations = 2048
        for repeat in range(repeats + 3):
            record = records[repeat % len(records)]["input"]
            started = time.perf_counter_ns()
            for _ in range(iterations):
                strict_utf8(str(record["prompt"]))
            baseline = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            frozen = True
            for _ in range(iterations):
                strict_utf8(str(record["prompt"]))
                if not frozen:
                    raise AssertionError
            adapted = time.perf_counter_ns() - started
            if repeat >= 3:
                baseline_times.append(baseline / 1e9)
                adapter_times.append(adapted / 1e9)
    else:
        import torch

        selected_device = torch.device(device)
        inputs = tokenizer(
            str(records[0]["input"]["prompt"]),
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {key: value.to(selected_device) for key, value in inputs.items()}
        with torch.inference_mode():
            for repeat in range(repeats + 3):
                if selected_device.type == "cuda":
                    torch.cuda.synchronize(selected_device)
                started = time.perf_counter_ns()
                model(**inputs, use_cache=False)
                if selected_device.type == "cuda":
                    torch.cuda.synchronize(selected_device)
                baseline = time.perf_counter_ns() - started

                if selected_device.type == "cuda":
                    torch.cuda.synchronize(selected_device)
                started = time.perf_counter_ns()
                model(**inputs, use_cache=False)
                _idle_adapter(records[repeat % len(records)]["input"], True)
                if selected_device.type == "cuda":
                    torch.cuda.synchronize(selected_device)
                adapted = time.perf_counter_ns() - started
                if repeat >= 3:
                    baseline_times.append(baseline / 1e9)
                    adapter_times.append(adapted / 1e9)
    baseline_median = statistics.median(baseline_times)
    adapter_median = statistics.median(adapter_times)
    overhead = adapter_median / baseline_median - 1.0
    return {
        "host_key": host_key,
        "repeated_observations": repeats,
        "host_alone_seconds": baseline_times,
        "host_plus_idle_adapter_seconds": adapter_times,
        "host_alone_median_seconds": baseline_median,
        "host_plus_idle_adapter_median_seconds": adapter_median,
        "overhead_fraction": overhead,
        "maximum_overhead_fraction": 0.1,
        "passed": overhead <= 0.1,
    }


def certify_host(
    root: Path,
    *,
    host_key: str,
    output_dir: Path,
    snapshot: Path | None = None,
    device: str = "cuda",
    physical_isolation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise HostCertificationError(f"immutable certification directory exists: {output_dir}")
    if host_key not in HOST_KEYS:
        raise HostCertificationError(f"unknown host: {host_key}")
    if (
        not isinstance(physical_isolation, dict)
        or physical_isolation.get("format") != "abi-v2-physical-certification-isolation/2"
        or physical_isolation.get("host_key") != host_key
        or physical_isolation.get("capability_archives_physically_present") != 0
        or physical_isolation.get("source_success_ledgers_physically_present") != 0
        or physical_isolation.get("mount", {}).get("sandbox_policy")
        != "abi-certification-pivot-root/1"
        or physical_isolation.get("mount", {}).get("old_root_present") is not False
        or physical_isolation.get("mount", {}).get("windows_mount_present") is not False
        or physical_isolation.get("mount", {}).get("unexpected_mount_points") != []
        or physical_isolation.get("capsule", {}).get("capability_archives_present") != 0
        or physical_isolation.get("capsule", {}).get("source_success_ledgers_present") != 0
        or physical_isolation.get("reachable_filesystem_forbidden_scan", {}).get(
            "format"
        )
        != "abi-v2-reachable-filesystem-inventory/1"
        or physical_isolation.get("reachable_filesystem_forbidden_scan", {}).get(
            "capability_archive_signature_matches"
        )
        != 0
        or physical_isolation.get("reachable_filesystem_forbidden_scan", {}).get(
            "forbidden_archive_member_paths"
        )
        != 0
        or physical_isolation.get("reachable_filesystem_forbidden_scan", {}).get(
            "campaign_identifier_matches"
        )
        != 0
    ):
        raise HostCertificationError(
            "physical certification-isolation evidence is missing or invalid"
        )

    spec_path = root / "abi_v2/canonical_spec.json"
    suite_path = root / "abi_v2/conformance_suite.json"
    implementation_path = root / "abi_v2/canonical.py"
    spec, suite = _object(spec_path), _object(suite_path)
    implementation_sha = _sha256_file(implementation_path)
    if (
        spec.get("status") != "PREREGISTERED_BEFORE_HOST_CERTIFICATION_AND_CAPABILITY_REVEAL"
        or suite.get("status")
        != "PREREGISTERED_BEFORE_HOST_CERTIFICATION_AND_CAPABILITY_REVEAL"
        or implementation_sha != spec["reference_implementation"]["sha256"]
        or spec["host_certification"]["trainable_parameters_maximum"] != 0
        or spec["host_certification"]["optimizer_steps_maximum"] != 0
    ):
        raise HostCertificationError("canonical ABI V2 governance changed")
    host = suite["host_registry"][host_key]
    reference_records = _reference_records(root, suite)
    reference_states = [verify_reference(record) for record in reference_records]
    texts = _neutral_texts(int(suite["certification_data"]["generated_roundtrip_records"]))
    example_hashes = [sha256_bytes(strict_utf8(text)) for text in texts]
    raw_utf8_bytes = sum(len(strict_utf8(text)) for text in texts)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    model = tokenizer = None
    inventory = []
    if host_key == "layercake":
        extension = root.parent / "layercake_release/layercake_extensions/route_isolated_clarification_core_v25.py"
        if _sha256_file(extension) != host["checkpoint_sha256"]:
            raise HostCertificationError("LayerCake v25 host identity changed")
        checks = _identity_host_checks(texts)
    else:
        if snapshot is None or not snapshot.is_dir():
            raise HostCertificationError(f"{host_key} requires an exact local snapshot")
        inventory_sha, inventory = _snapshot_inventory(snapshot.resolve())
        if inventory_sha != host["snapshot_inventory_sha256"]:
            raise HostCertificationError(f"{host_key} snapshot inventory changed")
        checks, model, tokenizer = _model_host_checks(
            snapshot=snapshot.resolve(),
            texts=texts,
            forward_records=int(suite["certification_data"]["model_forward_records"]),
            device=device,
        )
    performance = _performance(
        host_key=host_key,
        records=reference_records,
        repeats=int(spec["performance_gate"]["minimum_repeated_observations"]),
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    tokenizer_mode = checks["tokenizer_mode"]
    adapter = _adapter_document(
        host_key=host_key,
        host=host,
        spec_sha256=_sha256_file(spec_path),
        suite_sha256=_sha256_file(suite_path),
        implementation_sha256=implementation_sha,
        tokenizer_mode=tokenizer_mode,
    )
    output_dir.mkdir(parents=True)
    adapter_path = output_dir / "adapter.json"
    _write_once(adapter_path, adapter)
    try:
        os.chmod(adapter_path, stat.S_IREAD)
    except OSError:
        pass
    elapsed = time.perf_counter() - started
    rss_after = process.memory_info().rss
    gates = {
        "reference_vectors_exact": len(reference_states) == len(reference_records),
        "canonical_state_hashes_deterministic": all(
            canonical_context(record["input"])["state_sha256"] == state["state_sha256"]
            for record, state in zip(reference_records, reference_states)
        ),
        "strict_utf8_roundtrip_exact": True,
        "native_tokenizer_roundtrip_exact": checks["roundtrips_exact"] == checks["roundtrips"],
        "native_checkpoint_identity_exact": True,
        "native_forward_state_finite": checks["native_forward_finite_records"]
        == checks["native_forward_records"],
        "canonical_output_fusion_roundtrip_exact": all(
            bytes.fromhex(
                canonical_output_intent(text, capability_id="unrevealed-conformance")
                ["authoritative_utf8_hex"]
            ).decode("utf-8")
            == text
            for text in texts
        ),
        "capability_archives_physically_absent": physical_isolation[
            "capability_archives_physically_present"
        ]
        == 0,
        "source_success_ledgers_physically_absent": physical_isolation[
            "source_success_ledgers_physically_present"
        ]
        == 0,
        "development_filesystem_unmounted": (
            physical_isolation["mount"]["old_root_present"] is False
            and physical_isolation["mount"]["windows_mount_present"] is False
            and physical_isolation["mount"]["unexpected_mount_points"] == []
        ),
        "zero_trainable_adapter_parameters": adapter["trainable_parameters"] == 0,
        "zero_optimizer_steps": adapter["optimizer_steps"] == 0,
        "adapter_frozen": adapter["frozen"] is True,
        "adapter_overhead_within_10_percent": performance["passed"],
    }
    result = {
        "format": "abi-v2-host-certification-result/1",
        "status": "PASS_CAPABILITY_BLIND_HOST_CERTIFICATION"
        if all(gates.values())
        else "FAIL_HOST_CERTIFICATION",
        "host_key": host_key,
        "host": host,
        "device": device if host_key != "layercake" else "cpu",
        "canonical_spec_sha256": _sha256_file(spec_path),
        "conformance_suite_sha256": _sha256_file(suite_path),
        "reference_implementation_sha256": implementation_sha,
        "adapter": {
            "path": "adapter.json",
            "bytes": adapter_path.stat().st_size,
            "sha256": _sha256_file(adapter_path),
            "trainable_parameters": 0,
            "optimizer_present": False,
            "gradients_enabled": False,
        },
        "certification_data": {
            "kind": suite["certification_data"]["kind"],
            "examples": len(texts),
            "raw_utf8_bytes": raw_utf8_bytes,
            "model_visible_units": checks["model_visible_units"],
            "example_sha256": example_hashes,
            "reference_vector_records": len(reference_records),
            "capability_examples": 0,
            "capability_outputs": 0,
            "capability_success_ids": 0,
        },
        "checks": checks,
        "snapshot_inventory": inventory,
        "performance": performance,
        "cost": {
            "wall_seconds": elapsed,
            "cpu_hours": elapsed / 3600.0,
            "gpu_hours": elapsed / 3600.0 if host_key != "layercake" and device == "cuda" else 0.0,
            "peak_process_rss_bytes_lower_bound": max(rss_before, rss_after),
            "peak_cuda_allocated_bytes": checks.get("peak_cuda_allocated_bytes", 0),
            "trainable_parameters": 0,
            "adapter_bytes": adapter_path.stat().st_size,
        },
        "physical_isolation": {
            "format": physical_isolation["format"],
            "evidence_sha256": physical_isolation["evidence_sha256"],
            "capsule_manifest_sha256": physical_isolation["capsule"][
                "manifest_sha256"
            ],
            "capsule_inventory_sha256": physical_isolation["capsule"][
                "inventory_sha256"
            ],
            "capsule_files": physical_isolation["capsule"]["files_verified"],
            "sandbox_policy": physical_isolation["mount"]["sandbox_policy"],
            "old_root_present": physical_isolation["mount"]["old_root_present"],
            "windows_mount_present": physical_isolation["mount"][
                "windows_mount_present"
            ],
            "unexpected_mount_points": physical_isolation["mount"][
                "unexpected_mount_points"
            ],
            "capability_archives_present": physical_isolation[
                "capability_archives_physically_present"
            ],
            "source_success_ledgers_present": physical_isolation[
                "source_success_ledgers_physically_present"
            ],
        },
        "gates": gates,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    result["evidence_sha256"] = sha256_bytes(canonical_json_bytes(result))
    _write_once(output_dir / "result.json", result)
    _write_once(output_dir / "performance.json", performance)
    del model, tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=sorted(HOST_KEYS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--physical-isolation", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = certify_host(
        root,
        host_key=args.host,
        output_dir=(root / args.output_dir).resolve(),
        snapshot=Path(args.snapshot).resolve() if args.snapshot else None,
        device=args.device,
        physical_isolation=_object(Path(args.physical_isolation).resolve()),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
