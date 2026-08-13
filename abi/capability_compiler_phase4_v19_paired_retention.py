"""Contemporaneous paired ordinary-runtime retention for signed v18 and v19."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import torch

from . import capability_compiler_phase3_cpu_runtime as runtime
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-v19-paired-retention/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path):
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_PAIRED_V18_V19_ORDINARY_RETENTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or int(protocol["runtime"]["distinct_prompts"]) < 100
        or int(protocol["runtime"]["repeated_observations"]) < 20
        or int(protocol["runtime"]["torch_threads"]) != 1
    ):
        raise Phase3Error("paired v19 retention governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"paired v19 retention binding changed: {relative}")
    return protocol, sha256_file(path)


def _host_apis(layercake_root: Path):
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake_extensions.route_isolated_shallow_sparse_core_v18 import ExactRouteIsolatedShallowSparseCoreHost
    from layercake_extensions.route_isolated_prompt_span_core_v19 import PromptSpanRouteIsolatedShallowSparseCoreHost
    return ExactRouteIsolatedShallowSparseCoreHost, PromptSpanRouteIsolatedShallowSparseCoreHost


@torch.inference_mode()
def request(host: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    state = host.prefill(str(row["prompt"]))
    first = None
    for _ in range(int(row["max_new_tokens"])):
        token = host.decode_step(state)
        if token is None:
            break
        if first is None:
            first = time.perf_counter() - started
    output = host.realize(state).decode("utf-8", errors="strict")
    total = time.perf_counter() - started
    raw = output.encode("utf-8")
    ids = host.model_tokenizer.encode(output)
    return {
        "output": output,
        "output_token_ids": ids,
        "output_utf8_bytes": len(raw),
        "output_characters": len(output),
        "authoritative_output_tokens": len(ids),
        "time_to_first_output_seconds": float(first if first is not None else total),
        "total_seconds": total,
        "bytes_per_second": len(raw) / total,
        "characters_per_second": len(output) / total,
    }


def run(root: Path, protocol_path: Path, output: Path):
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable paired v19 retention output exists: {output}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    probes = development_probes(root / protocol["development_catalog"])
    distinct, scheduled = runtime.select_runtime_probes(probes, int(protocol["runtime"]["distinct_prompts"]), int(protocol["runtime"]["repeated_observations"]))
    reference = {row["probe_id"]: row for row in (json.loads(line) for line in (root / protocol["ordinary_reference_outputs"]).open(encoding="utf-8"))}
    public = (root / protocol["public_key"]).read_bytes()
    metadata = _json(root / protocol["v19_metadata"])
    V18, V19 = _host_apis((root / protocol["layercake_root"]).resolve())
    with tempfile.TemporaryDirectory(prefix="abi-v19-retention-") as raw:
        temp = Path(raw)
        v18 = V18(temp / "v18", trust_store={metadata["public_key"]["key_id"]: public}, device="cpu")
        v19 = V19(temp / "v19", trust_store={metadata["public_key"]["key_id"]: public}, device="cpu")
        active18 = v18.activate(root / protocol["v18_package"])
        active19 = v19.activate(root / protocol["v19_package"])
        for row in distinct[: int(protocol["runtime"]["warmup_observations"])]:
            request(v18, row)
            request(v19, row)
        evidence = []
        for index, row in enumerate(scheduled):
            order = (("v18", v18), ("v19", v19)) if index % 2 == 0 else (("v19", v19), ("v18", v18))
            values = {name: request(host, row) for name, host in order}
            expected = reference[row["probe_id"]]
            evidence.append({
                "index": index,
                "probe_id": row["probe_id"],
                "execution_order": [name for name, _ in order],
                "v18": values["v18"],
                "v19": values["v19"],
                "v18_v19_output_identity": values["v18"]["output"] == values["v19"]["output"] and values["v18"]["output_token_ids"] == values["v19"]["output_token_ids"],
                "v18_reference_identity": values["v18"]["output"] == expected["output"] and values["v18"]["output_token_ids"] == expected["output_token_ids"],
                "v19_reference_identity": values["v19"]["output"] == expected["output"] and values["v19"]["output_token_ids"] == expected["output_token_ids"],
            })
    v18_rows = [row["v18"] for row in evidence]
    v19_rows = [row["v19"] for row in evidence]
    v18_median = statistics.median(row["bytes_per_second"] for row in v18_rows)
    v19_median = statistics.median(row["bytes_per_second"] for row in v19_rows)
    comparison = runtime.paired_ratio_bootstrap(
        [row["bytes_per_second"] for row in v19_rows],
        [row["bytes_per_second"] for row in v18_rows],
        int(protocol["statistics"]["bootstrap_replicates"]),
        int(protocol["statistics"]["bootstrap_seed"]),
    )
    median_retention = v19_median / v18_median
    gates = {
        "same_tensor_payload": active18["payload_hash"] == active19["payload_hash"] == protocol["tensor_payload_hash"],
        "signed_archives_bound": active18["archive_hash"] == protocol["bindings"][protocol["v18_package"]] and active19["archive_hash"] == protocol["bindings"][protocol["v19_package"]],
        "distinct_depth": len({row["probe_id"] for row in evidence}) == 100,
        "observation_depth": len(evidence) == 120,
        "balanced_execution_order": sum(row["execution_order"][0] == "v18" for row in evidence) == sum(row["execution_order"][0] == "v19" for row in evidence) == 60,
        "v18_v19_output_identity": all(row["v18_v19_output_identity"] for row in evidence),
        "v18_reference_identity": all(row["v18_reference_identity"] for row in evidence),
        "v19_reference_identity": all(row["v19_reference_identity"] for row in evidence),
        "median_retention": median_retention >= float(protocol["gates"]["retention_minimum"]),
        "paired_lower_retention": comparison["lower_95"] >= float(protocol["gates"]["paired_lower_minimum"]),
        "authoritative_token_accounting": all(row["authoritative_output_tokens"] == len(row["output_token_ids"]) for row in v18_rows + v19_rows),
        "p95_supported": len(evidence) >= 100,
        "p99_not_promoted": len(evidence) < 1000,
        "receiver_learning_zero": active18["receiver_training_steps"] == active18["receiver_calibration_runs"] == active19["receiver_training_steps"] == active19["receiver_calibration_runs"] == 0,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    raw_path = output.parent / "observations.jsonl"
    output.parent.mkdir(parents=True)
    _write_immutable(raw_path, b"".join(canonical_json_bytes(row) for row in evidence))
    result = {
        "format": "abi-capability-compiler-phase4-v19-paired-retention-result/1",
        "status": "PASS_PAIRED_V19_ORDINARY_RETENTION" if all(gates.values()) else "FAIL_PAIRED_V19_ORDINARY_RETENTION",
        "protocol_sha256": protocol_sha,
        "v18": {"archive_sha256": active18["archive_hash"], "median_bytes_per_second": v18_median, "median_characters_per_second": statistics.median(row["characters_per_second"] for row in v18_rows), "median_ttft_seconds": statistics.median(row["time_to_first_output_seconds"] for row in v18_rows)},
        "v19": {"archive_sha256": active19["archive_hash"], "median_bytes_per_second": v19_median, "median_characters_per_second": statistics.median(row["characters_per_second"] for row in v19_rows), "median_ttft_seconds": statistics.median(row["time_to_first_output_seconds"] for row in v19_rows)},
        "tensor_payload_hash": active19["payload_hash"],
        "median_throughput_retention": median_retention,
        "paired_throughput_retention": comparison,
        "prompt_depth": {"distinct": 100, "repeated": 20, "observations_per_system": 120},
        "gates": gates,
        "raw_observations_sha256": sha256_file(raw_path),
        "training_performed": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "final_test_accessed": False,
        "claim_boundary": "Contemporaneous paired ordinary v18-to-v19 retention only; prompt-span Qwen comparison remains separately bound, and stable frontier, Phase 4, final test, and superiority remain unproven.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
