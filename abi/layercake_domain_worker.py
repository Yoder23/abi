"""Private JSONL worker for explicitly selected LayerCake domain cakes."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil

from .layercake_domains import (
    DIRECT_ABI_SHA256,
    DIRECT_ABI_VERSION,
    _import_layercake,
)


def _write(value: dict) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True) + "\n")
    sys.stdout.flush()


def main() -> int:
    line = sys.stdin.readline()
    if not line:
        return 2
    configuration = json.loads(line)
    lc = _import_layercake(Path(configuration["layercake_root"]))
    packages = configuration["packages"]
    trust_store = {
        row["key_id"]: Path(row["public_key_path"]).read_bytes()
        for row in packages
    }
    host = lc["DirectCakeHost"](
        configuration["registry_root"],
        abi_version=DIRECT_ABI_VERSION,
        abi_hash=DIRECT_ABI_SHA256,
        trust_store=trust_store,
        device=configuration["device"],
    )
    for row in packages:
        host.install(row["package_path"])
    _write(
        {
            "status": "READY",
            "device": configuration["device"],
            "installed_ids": list(host.installed_ids()),
            "process_id": psutil.Process().pid,
        }
    )
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "close":
            _write({"status": "CLOSED"})
            return 0
        if request.get("command") != "generate":
            _write({"status": "FAIL", "error": "unsupported command"})
            continue
        before_telemetry = host.telemetry()
        started = time.perf_counter_ns()
        result = host.generate(
            request["cake_id"],
            request["prompt"],
            maximum_actions=request.get("maximum_actions"),
        )
        completed = time.perf_counter_ns()
        memory = psutil.Process().memory_info()
        after_telemetry = host.telemetry()
        telemetry_delta = {
            cake_id: {
                key: values[key] - before_telemetry[cake_id][key]
                for key in values
            }
            for cake_id, values in after_telemetry.items()
        }
        _write(
            {
                "status": "PASS",
                "cake_id": result.cake_id,
                "output_base64": base64.b64encode(result.output).decode("ascii"),
                "output_sha256": hashlib.sha256(result.output).hexdigest(),
                "authoritative_generated_actions": list(result.actions),
                "authoritative_generated_action_count": len(result.actions),
                "prefill_calls": result.prefill_calls,
                "decode_step_calls": result.decode_step_calls,
                "total_latency_seconds": (completed - started) / 1e9,
                "process_resident_bytes": int(memory.rss),
                "process_peak_resident_bytes": int(
                    getattr(memory, "peak_wset", memory.rss)
                ),
                "telemetry": after_telemetry,
                "telemetry_delta": telemetry_delta,
                "teacher_present_at_inference": False,
                "source_transformer_blocks_retained": 0,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
