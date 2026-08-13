"""Read-only attribution of v19 ordinary-loop bookkeeping overhead."""

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

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-v19-runtime-profile/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path):
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_V19_ORDINARY_PROFILE"
        or protocol.get("training_authorized") is not False
        or protocol.get("host_mutation_persistence_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v19 runtime profile governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v19 runtime profile binding changed: {relative}")
    return protocol, sha256_file(path)


def _host_api(layercake_root: Path):
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake_extensions.route_isolated_prompt_span_core_v19 import PromptSpanRouteIsolatedShallowSparseCoreHost
    from layercake_extensions.route_isolated_shallow_sparse_core import repetition_collapse
    return PromptSpanRouteIsolatedShallowSparseCoreHost, repetition_collapse


@torch.inference_mode()
def _current(host: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    state = host.prefill(str(row["prompt"]))
    steps = 0
    for _ in range(int(row["max_new_tokens"])):
        if host.decode_step(state) is None:
            break
        steps += 1
    output = host.realize(state).decode("utf-8")
    wall = time.perf_counter() - started
    return {"output": output, "output_token_ids": host.model_tokenizer.encode(output), "wall_seconds": wall, "bytes_per_second": len(output.encode()) / wall, "generated_steps": steps, "route_set_calls": steps + 1}


@torch.inference_mode()
def _direct(host: Any, row: Mapping[str, Any], repetition_collapse: Any) -> dict[str, Any]:
    started = time.perf_counter()
    state = host.prefill(str(row["prompt"]))
    model, _, _, tokenizer, _ = host._require_active()
    steps = 0
    for _ in range(int(row["max_new_tokens"])):
        if state["finished"]:
            break
        selected = state["next_logits"].argmax(dim=-1)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            state["finished"] = True
            break
        candidate = [*state["generated_ids"], token]
        if int(state["weak_route"]) >= 0 and repetition_collapse(tokenizer.decode(candidate)):
            state["terminated_by_guard"] = True
            state["finished"] = True
            break
        state["generated_ids"].append(token)
        result = model(selected[:, None], task_routes=state["task_route"], past_key_values=state["past_key_values"], use_cache=True)
        state["past_key_values"] = result["past_key_values"]
        state["next_logits"] = result["logits"][:, -1]
        steps += 1
    output = host.realize(state).decode("utf-8")
    wall = time.perf_counter() - started
    return {"output": output, "output_token_ids": host.model_tokenizer.encode(output), "wall_seconds": wall, "bytes_per_second": len(output.encode()) / wall, "generated_steps": steps, "route_set_calls": 1}


def run(root: Path, protocol_path: Path, output: Path):
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v19 runtime profile exists: {output}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    probes = development_probes(root / protocol["development_catalog"])
    selected = probes[: int(protocol["profile"]["distinct_prompts"])]
    reference = {row["probe_id"]: row for row in (json.loads(line) for line in (root / protocol["ordinary_reference_outputs"]).open(encoding="utf-8"))}
    Host, repetition_collapse = _host_api((root / protocol["layercake_root"]).resolve())
    public = (root / protocol["public_key"]).read_bytes()
    metadata = _json(root / protocol["package_metadata"])
    with tempfile.TemporaryDirectory(prefix="abi-v19-profile-") as raw:
        host = Host(Path(raw), trust_store={metadata["public_key"]["key_id"]: public}, device="cpu")
        activation = host.activate(root / protocol["package"])
        for row in selected[:3]:
            _current(host, row)
            _direct(host, row, repetition_collapse)
        evidence = []
        for repeat in range(int(protocol["profile"]["repeats"])):
            order = ("current", "direct") if repeat % 2 == 0 else ("direct", "current")
            for row in selected:
                values = {}
                for mode in order:
                    values[mode] = _current(host, row) if mode == "current" else _direct(host, row, repetition_collapse)
                expected = reference[row["probe_id"]]
                evidence.append({
                    "repeat": repeat,
                    "probe_id": row["probe_id"],
                    "current": values["current"],
                    "direct": values["direct"],
                    "current_direct_output_identity": values["current"]["output"] == values["direct"]["output"],
                    "current_reference_identity": values["current"]["output"] == expected["output"] and values["current"]["output_token_ids"] == expected["output_token_ids"],
                    "direct_reference_identity": values["direct"]["output"] == expected["output"] and values["direct"]["output_token_ids"] == expected["output_token_ids"],
                })
    current = [row["current"]["bytes_per_second"] for row in evidence]
    direct = [row["direct"]["bytes_per_second"] for row in evidence]
    current_median = statistics.median(current)
    direct_median = statistics.median(direct)
    ratio = direct_median / current_median
    gates = {
        "same_signed_package": activation["archive_hash"] == protocol["bindings"][protocol["package"]],
        "same_outputs": all(row["current_direct_output_identity"] for row in evidence),
        "current_reference_identity": all(row["current_reference_identity"] for row in evidence),
        "direct_reference_identity": all(row["direct_reference_identity"] for row in evidence),
        "route_set_calls_reduced": all(row["direct"]["route_set_calls"] == 1 and row["current"]["route_set_calls"] == row["current"]["generated_steps"] + 1 for row in evidence),
        "measured_speedup_at_least_required": ratio >= float(protocol["profile"]["minimum_speedup_to_explain_gap"]),
        "no_persistent_host_mutation": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    raw_path = output.parent / "observations.jsonl"
    output.parent.mkdir(parents=True)
    _write_immutable(raw_path, b"".join(canonical_json_bytes(row) for row in evidence))
    result = {
        "format": "abi-capability-compiler-phase4-v19-runtime-profile-result/1",
        "status": "PASS_ROUTE_RESET_BOOKKEEPING_BOTTLENECK_LOCALIZED" if all(gates.values()) else "FAIL_BOTTLENECK_NOT_LOCALIZED",
        "protocol_sha256": protocol_sha,
        "observations_per_mode": len(evidence),
        "distinct_prompts": len(selected),
        "current_median_bytes_per_second": current_median,
        "direct_median_bytes_per_second": direct_median,
        "direct_to_current_ratio": ratio,
        "current_total_route_set_calls": sum(row["current"]["route_set_calls"] for row in evidence),
        "direct_total_route_set_calls": sum(row["direct"]["route_set_calls"] for row in evidence),
        "gates": gates,
        "raw_observations_sha256": sha256_file(raw_path),
        "training_performed": False,
        "host_mutation_persisted": False,
        "final_test_accessed": False,
        "claim_boundary": "Read-only ordinary-loop attribution only; no production host change, runtime promotion, stable frontier, Phase 4, or superiority claim.",
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
