"""Measure paired contemporaneous V24-to-V23 ordinary-path CPU retention."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b50_cpu_runtime import (
    _ordinary_request,
    _paired_prompt_throughput,
    _paired_ratio_or_zero,
)
from .capability_compiler_phase4_b50_gpu_runtime import _runtime_metrics, runtime_schedule
from .capability_compiler_phase4_b50_v23_conformance import (
    _api_v23,
    _repackage as _repackage_v23,
)
from .capability_compiler_phase4_b50_v24_conformance import (
    _api_v24,
    _repackage_v24,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase4_v22_b50_rescreen import (
    _api as _api_v22,
    load_protocol as _load_v22_protocol,
)


FORMAT = "abi-capability-compiler-phase4-b50-v24-paired-retention/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-v24-paired-retention-result/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    cfg = protocol.get("runtime", {})
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_PAIRED_EXACT_B50_V24_TO_V23_CPU_RETENTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_generation_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or int(cfg.get("distinct_prompts", 0)) != 100
        or int(cfg.get("repeated_observations", 0)) < 20
        or int(cfg.get("torch_threads", 0)) != 1
    ):
        raise Phase3Error("exact B50 v24 paired-retention governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 v24 retention binding changed: {relative}")
    return protocol, sha256_file(path)


def _identity(
    rows: list[Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]
) -> int:
    return sum(
        str(row["output"]) == str(reference[str(row["probe_id"])]["output"])
        and [int(value) for value in row["output_token_ids"]]
        == [
            int(value)
            for value in reference[str(row["probe_id"])]["output_token_ids"]
        ]
        for row in rows
    )


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B50 v24 retention output exists: {output}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base_runtime = _json(root / str(protocol["runtime_schedule_protocol"]))
    distinct, scheduled = runtime_schedule(root, base_runtime)
    source, _ = _load_v22_protocol(
        root, root / str(protocol["source_candidate_protocol"])
    )
    layercake_root = (root / str(source["layercake_root"])).resolve()
    api_v22 = _api_v22(layercake_root)
    api_v23 = _api_v23(layercake_root)
    api_v24 = _api_v24(layercake_root)
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(source["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api_v24["key_id"](public_pem)
    spec = next(
        row for row in source["systems"]
        if int(row["seed"]) == int(protocol["seed"])
    )
    reference_path = root / str(protocol["ordinary_reference_observations"])
    if sha256_file(reference_path) != protocol["ordinary_reference_sha256"]:
        raise Phase3Error("exact B50 v24 ordinary reference changed")
    reference_rows = [
        row
        for row in (
            json.loads(line)
            for line in reference_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("system") == "layercake_v22_b50"
        and row.get("mode") == "ordinary"
    ]
    reference = {str(row["probe_id"]): row for row in reference_rows}
    if len(reference_rows) != 120 or len(reference) != 100:
        raise Phase3Error("exact B50 v24 ordinary reference depth changed")
    with tempfile.TemporaryDirectory(prefix="abi-b50-v24-retention-") as raw:
        temporary = Path(raw)
        v23_path, v23_package = _repackage_v23(
            root,
            source,
            spec,
            temporary / "v23",
            api_v22,
            api_v23,
            private,
            public_pem,
        )
        v24_path, v24_package = _repackage_v24(
            root,
            source,
            spec,
            temporary / "v24",
            api_v22,
            api_v24,
            private,
            public_pem,
        )
        if (
            v23_package["archive_sha256"] != protocol["v23"]["archive_sha256"]
            or v24_package["archive_sha256"] != protocol["v24"]["archive_sha256"]
            or v23_package["tensor_payload_hash"]
            != v24_package["tensor_payload_hash"]
            != protocol["tensor_payload_hash"]
        ):
            raise Phase3Error("exact B50 v24 retention package identity changed")
        v23_host = api_v23["Host"](
            temporary / "registry-v23", trust_store={signer: public_pem}, device="cpu"
        )
        v24_host = api_v24["Host"](
            temporary / "registry-v24", trust_store={signer: public_pem}, device="cpu"
        )
        v23_active = v23_host.activate(v23_path)
        v24_active = v24_host.activate(v24_path)
        for probe in distinct[: int(protocol["runtime"]["warmup_observations"])]:
            _ordinary_request(v23_host, probe)
            _ordinary_request(v24_host, probe)
        v23_rows: list[dict[str, Any]] = []
        v24_rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for index, probe in enumerate(scheduled):
            order = ("v23", "v24") if index % 2 == 0 else ("v24", "v23")
            values = {}
            for system in order:
                values[system] = _ordinary_request(
                    v23_host if system == "v23" else v24_host, probe
                )
            v23_rows.append(values["v23"])
            v24_rows.append(values["v24"])
        wall = time.perf_counter() - started
        v23_verified = v23_host.verify()
        v24_verified = v24_host.verify()
        del v23_host, v24_host
        gc.collect()
    v23_metrics = _runtime_metrics(v23_rows)
    v24_metrics = _runtime_metrics(v24_rows)
    prompt_v24, prompt_v23 = _paired_prompt_throughput(v24_rows, v23_rows)
    paired = _paired_ratio_or_zero(
        prompt_v24,
        prompt_v23,
        replicates=int(protocol["statistics"]["bootstrap_replicates"]),
        seed=int(protocol["statistics"]["bootstrap_seed"]),
    )
    paired["method"] = "paired_prompt_median_v24_to_v23_throughput_ratio_percentile_bootstrap"
    paired["prompt_pairs"] = len(prompt_v24)
    median_retention = (
        v24_metrics["median_bytes_per_second"]
        / v23_metrics["median_bytes_per_second"]
    )
    v23_identity = _identity(v23_rows, reference)
    v24_identity = _identity(v24_rows, reference)
    cross_identity = sum(
        left["output"] == right["output"]
        and left["output_token_ids"] == right["output_token_ids"]
        for left, right in zip(v23_rows, v24_rows)
    )
    minimum = float(protocol["gates"]["retention_minimum"])
    gates = {
        "same_tensor_payload": v23_active["payload_hash"]
        == v24_active["payload_hash"]
        == protocol["tensor_payload_hash"],
        "package_identity": v23_active["archive_hash"]
        == protocol["v23"]["archive_sha256"]
        and v24_active["archive_hash"] == protocol["v24"]["archive_sha256"],
        "package_verifies": v23_verified["status"] == v24_verified["status"] == "PASS",
        "all_v23_ordinary_outputs_exact": v23_identity == 120,
        "all_v24_ordinary_outputs_exact": v24_identity == 120,
        "all_cross_outputs_exact": cross_identity == 120,
        "median_retention_at_least_95_percent": median_retention >= minimum,
        "paired_lower_retention_at_least_95_percent": paired["lower_95"] is not None
        and paired["lower_95"] >= minimum,
        "authoritative_token_accounting": all(
            row["authoritative_output_tokens"]
            == len(row["retokenized_output_token_ids"])
            for row in v23_rows + v24_rows
        ),
        "depth": len(v23_rows) == len(v24_rows) == 120
        and len({row["probe_id"] for row in v24_rows}) == 100,
        "v24_strict_tensor_adoption": v24_active["strict_assigned_tensor_count"]
        == v24_active["authenticated_tensor_count"]
        == 89
        and v24_active["meta_tensors_after_adoption"] == 0,
        "receiver_learning_zero": v23_active["receiver_training_steps"]
        == v23_active["receiver_calibration_runs"]
        == v24_active["receiver_training_steps"]
        == v24_active["receiver_calibration_runs"]
        == 0,
        "training_absent": True,
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    output.mkdir(parents=True)
    observations = output / "observations.jsonl"
    _write_immutable(
        observations,
        b"".join(
            canonical_json_bytes({"system": system, **row})
            for system, rows in (("v23", v23_rows), ("v24", v24_rows))
            for row in rows
        ),
    )
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_PAIRED_EXACT_B50_V24_TO_V23_CPU_RETENTION"
        if all(gates.values())
        else "FAIL_PAIRED_EXACT_B50_V24_TO_V23_CPU_RETENTION",
        "protocol_sha256": protocol_sha,
        "v23": {"activation": v23_active, "metrics": v23_metrics},
        "v24": {"activation": v24_active, "metrics": v24_metrics},
        "comparisons": {
            "median_throughput_retention": median_retention,
            "paired_throughput_retention": paired,
            "v23_output_identities": v23_identity,
            "v24_output_identities": v24_identity,
            "cross_output_identities": cross_identity,
        },
        "prompt_depth": {"distinct": 100, "repeated": 20, "observations_per_system": 120},
        "interleaved_wall_seconds": wall,
        "gates": gates,
        "observations_sha256": sha256_file(observations),
        "training_performed": False,
        "teacher_query_performed": False,
        "model_inference_performed": True,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Contemporaneous paired exact-payload V24-to-V23 ordinary-path CPU retention only. No Qwen product comparison, final test, Phase 4, or unconditional ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
