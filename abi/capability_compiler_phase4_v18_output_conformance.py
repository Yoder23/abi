"""Verify exact GPU output conformance of the packaged Phase 3 composite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import psutil
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-v18-output-conformance/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EXACT_GPU_OUTPUT_CONFORMANCE"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v18 output-conformance governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v18 output-conformance binding changed: {relative}")
    return protocol, sha256_file(path)


def compare_row(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "probe_id": actual["probe_id"] == expected["probe_id"],
        "capability": actual["capability"] == expected["capability"],
        "output": actual["output"] == expected["output"],
        "original_output": actual["original_output"] == expected["original_output"],
        "output_token_ids": actual["output_token_ids"] == expected["output_token_ids"],
        "automatic_capability_route": actual["automatic_capability_route"] == expected["automatic_capability_route"],
        "control_residual_route": actual["control_residual_route"] == expected["control_residual_route"],
        "task_route": actual["task_route"] == expected["task_route"],
        "guard_terminated": actual["guard_terminated"] == expected["guard_terminated"],
    }


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("preregistered CUDA device is unavailable")
    reference = _rows(root / protocol["reference_outputs"])
    probes = development_probes(root / protocol["development_catalog"])
    if len(reference) != 1400 or len(probes) != 1400:
        raise Phase3Error("locked conformance population changed")
    if [row["probe_id"] for row in reference] != [row["probe_id"] for row in probes]:
        raise Phase3Error("reference/probe order changed")
    metadata = _json(root / protocol["package_metadata"])
    if metadata.get("interface") != protocol["interface"] or metadata.get("package", {}).get("sha256") != protocol["bindings"][protocol["package"]]:
        raise Phase3Error("package metadata identity changed")
    return {
        "status": "PASS_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "observations": len(probes),
        "distinct_prompts": len({row["probe_id"] for row in probes}),
        "package_sha256": protocol["bindings"][protocol["package"]],
        "reference_outputs_sha256": protocol["bindings"][protocol["reference_outputs"]],
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "training_performed": False,
        "final_test_accessed": False,
    }


def _host_api(layercake_root: Path):
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake_extensions.route_isolated_shallow_sparse_core_v18 import ExactRouteIsolatedShallowSparseCoreHost
    return ExactRouteIsolatedShallowSparseCoreHost


@torch.inference_mode()
def execute(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v18 conformance output exists: {output}")
    check = preflight(root, protocol_path)
    package = root / protocol["package"]
    public = (root / protocol["public_key"]).read_bytes()
    metadata = _json(root / protocol["package_metadata"])
    reference = _rows(root / protocol["reference_outputs"])
    probes = development_probes(root / protocol["development_catalog"])
    Host = _host_api((root / protocol["layercake_root"]).resolve())
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="abi-phase4-v18-conformance-") as raw:
        host = Host(Path(raw), trust_store={metadata["public_key"]["key_id"]: public}, device="cuda")
        activation = host.activate(package)
        for index, (probe, expected) in enumerate(zip(probes, reference)):
            state = host.prefill(str(probe["prompt"]))
            for _ in range(int(probe["max_new_tokens"])):
                if host.decode_step(state) is None:
                    break
            output_text = host.realize(state).decode("utf-8", errors="strict")
            original = host.model_tokenizer.decode(state["generated_ids"])
            actual = {
                "probe_id": str(probe["probe_id"]),
                "capability": str(probe["canonical_capability"]),
                "output": output_text,
                "original_output": original,
                "output_token_ids": host.model_tokenizer.encode(output_text),
                "automatic_capability_route": str(state["capability"]),
                "control_residual_route": None if int(state["weak_route"]) < 0 else int(state["weak_route"]),
                "task_route": int(state["task_route"].item()),
                "guard_terminated": bool(state["terminated_by_guard"]),
            }
            matches = compare_row(actual, expected)
            rows.append({**actual, "matches": matches, "all_exact": all(matches.values())})
            peak_rss = max(peak_rss, process.memory_info().rss)
            if (index + 1) % 100 == 0:
                print(json.dumps({"evaluated": index + 1, "exact": sum(row["all_exact"] for row in rows)}), flush=True)
        verification = host.verify()
    output.mkdir(parents=True)
    raw_path = output / "development_outputs.jsonl"
    _write_immutable(raw_path, b"".join(canonical_json_bytes(row) for row in rows))
    fields = tuple(next(iter(rows))["matches"])
    field_exact = {field: sum(row["matches"][field] for row in rows) for field in fields}
    exact = sum(row["all_exact"] for row in rows)
    result = {
        "format": "abi-capability-compiler-phase4-v18-output-conformance-result/1",
        "status": "PASS_EXACT_GPU_OUTPUT_CONFORMANCE" if exact == 1400 and all(value == 1400 for value in field_exact.values()) else "FAIL_GPU_OUTPUT_CONFORMANCE",
        "protocol_sha256": protocol_sha,
        "interface": protocol["interface"],
        "package_sha256": sha256_file(package),
        "package_hash": activation["archive_hash"],
        "payload_hash": activation["payload_hash"],
        "reference_outputs_sha256": sha256_file(root / protocol["reference_outputs"]),
        "raw_outputs_sha256": sha256_file(raw_path),
        "observations": len(rows),
        "distinct_prompts": len({row["probe_id"] for row in rows}),
        "exact_rows": exact,
        "field_exact": field_exact,
        "receiver_training_steps": activation["receiver_training_steps"],
        "receiver_calibration_runs": activation["receiver_calibration_runs"],
        "package_verification": verification["status"],
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(peak_rss),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "teacher_present": False,
        "source_transformer_blocks": 0,
        "training_performed": False,
        "final_test_accessed": False,
        "claim_boundary": "Exact package-versus-unpackaged Phase 3 development-output conformance only; no Phase 4 information minimum, final-test, matched-baseline, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("execute")
    run.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = root / args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else execute(root, protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
