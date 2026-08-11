"""Matched CPU runtime, cold-start, TTFT, RSS, and guard-overhead screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import threading
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.request import Request, urlopen

import psutil
from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _write_immutable
from .capability_compiler_phase3_guarded_screen import artifact_markers
from .capability_compiler_phase3_targeted_recovery_bridge import _load_parent, _load_router
from .capability_compiler_phase3_weak_residual import SharedWeakResidual, WEAK_CAPABILITIES, _attach, _set_routes
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase3-matched-cpu-runtime/1"


def _json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_MATCHED_CPU_RUNTIME_TTFT_RSS_SCREEN" or protocol.get("neural_training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("CPU runtime governance changed")
    if int(protocol["runtime"]["distinct_prompts"]) < 100 or int(protocol["runtime"]["repeated_observations"]) < 20 or int(protocol["runtime"]["torch_threads"]) != 1: raise Phase3Error("CPU runtime depth or thread lock changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"CPU runtime binding changed: {relative}")
    return protocol, sha256_file(path)


def select_runtime_probes(probes: list[dict[str, Any]], distinct: int, repeats: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = {capability: [row for row in probes if row["canonical_capability"] == capability] for capability in CAPABILITIES}
    selected: list[dict[str, Any]] = []
    for index in range(100):
        for capability in CAPABILITIES:
            if len(selected) < distinct: selected.append(grouped[capability][index])
    if len(selected) != distinct or len({row["probe_id"] for row in selected}) != distinct: raise Phase3Error("runtime distinct prompt selection changed")
    repeated = [selected[index % len(selected)] for index in range(repeats)]
    return selected, [*selected, *repeated]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values); position = (len(ordered) - 1) * q; low = int(position); high = min(low + 1, len(ordered)); fraction = position - low; return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def paired_ratio_bootstrap(candidate: list[float], baseline: list[float], replicates: int, seed: int) -> dict[str, Any]:
    if len(candidate) != len(baseline) or not candidate or any(value <= 0 for value in baseline): raise Phase3Error("invalid paired throughput values")
    ratios = [left / right for left, right in zip(candidate, baseline)]; generator = random.Random(seed); draws = []
    for _ in range(replicates):
        sample = [ratios[generator.randrange(len(ratios))] for _ in ratios]; draws.append(statistics.mean(sample))
    return {"method": "paired_observation_mean_ratio_percentile_bootstrap", "observations": len(ratios), "replicates": replicates, "seed": seed, "mean_ratio": statistics.mean(ratios), "median_ratio": statistics.median(ratios), "lower_95": percentile(draws, 0.025), "upper_95": percentile(draws, 0.975)}


def _tensor_bytes(module: torch.nn.Module) -> int: return sum(value.numel() * value.element_size() for value in module.parameters()) + sum(value.numel() * value.element_size() for value in module.buffers())


class PeakMonitor:
    def __init__(self, measure: Callable[[], int]): self.measure = measure; self.peak = measure(); self._stop = threading.Event(); self._thread = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._stop.wait(0.01):
            try: self.peak = max(self.peak, self.measure())
            except (psutil.Error, OSError): pass
    def __enter__(self): self._thread.start(); return self
    def __exit__(self, *_): self._stop.set(); self._thread.join(); self.peak = max(self.peak, self.measure())


def _ollama_rss() -> int:
    total = 0
    for process in psutil.process_iter(["name", "memory_info"]):
        try:
            if "ollama" in str(process.info["name"] or "").casefold(): total += int(process.info["memory_info"].rss)
        except (psutil.Error, OSError): pass
    return total


def _post_json(url: str, body: Mapping[str, Any], *, stream: bool = False):
    request = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return urlopen(request, timeout=600 if stream else 30)


def _ollama_unload(base_url: str, model: str) -> float:
    started = time.perf_counter()
    with _post_json(base_url + "/api/generate", {"model": model, "prompt": "", "stream": False, "keep_alive": 0}) as response: json.loads(response.read())
    while time.perf_counter() - started < 30:
        with urlopen(base_url + "/api/ps", timeout=10) as response: loaded = json.loads(response.read()).get("models", [])
        if not any(row.get("name") == model or row.get("model") == model for row in loaded): return time.perf_counter() - started
        time.sleep(0.05)
    raise Phase3Error("Ollama model did not unload")


def _ollama_request(base_url: str, model: str, probe: Mapping[str, Any], keep_alive: str) -> dict[str, Any]:
    started = time.perf_counter(); first = None; fragments: list[str] = []; final: dict[str, Any] = {}
    body = {"model": model, "messages": [{"role": "user", "content": str(probe["prompt"])}], "stream": True, "keep_alive": keep_alive, "options": {"temperature": 0, "seed": 240503, "num_predict": int(probe["max_new_tokens"]), "num_ctx": 1024}}
    with _post_json(base_url + "/api/chat", body, stream=True) as response:
        for line in response:
            if not line.strip(): continue
            event = json.loads(line); content = str(event.get("message", {}).get("content", ""))
            if content and first is None: first = time.perf_counter() - started
            fragments.append(content)
            if event.get("done"): final = event
    total = time.perf_counter() - started; output = "".join(fragments); output_bytes = len(output.encode("utf-8")); output_chars = len(output)
    return {"probe_id": probe["probe_id"], "output": output, "output_utf8_bytes": output_bytes, "output_characters": output_chars, "authoritative_output_tokens": int(final.get("eval_count", 0)), "time_to_first_output_seconds": float(first if first is not None else total), "total_seconds": total, "bytes_per_second": output_bytes / total if total else 0.0, "characters_per_second": output_chars / total if total else 0.0, "load_seconds_reported": int(final.get("load_duration", 0)) / 1e9, "prompt_eval_tokens": int(final.get("prompt_eval_count", 0))}


@torch.inference_mode()
def _layercake_request(model: Any, tokenizer: Any, router: Any, router_tokenizer: Any, router_protocol: Mapping[str, Any], prompt_row: Mapping[str, Any], markers: tuple[str, ...], clause: str, residual_active: bool) -> dict[str, Any]:
    started = time.perf_counter(); prompt = str(prompt_row["prompt"]); capability, details = sparse._route(router, router_tokenizer, router_protocol, prompt); expected = str(prompt_row["canonical_capability"])
    weak = capability in WEAK_CAPABILITIES and residual_active; weak_to_id = {name: index for index, name in enumerate(WEAK_CAPABILITIES)}; _set_routes(model, torch.tensor([weak_to_id[capability] if weak else -1], dtype=torch.long)); forced = torch.tensor([CAPABILITY_TO_ROUTE[capability]], dtype=torch.long) if weak else None
    prompt_ids = [int(value) for value in tokenizer.encode(prompt.rstrip() + "\n", add_special_tokens=False)]; ids = torch.tensor([prompt_ids], dtype=torch.long); result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], dtype=torch.long), task_routes=forced, use_cache=True); route = result["task_routes"].detach().clone(); cache = result["past_key_values"]; logits = result["logits"][:, -1]; generated: list[int] = []; first = None; terminated = False; guard_seconds = 0.0
    for _ in range(int(prompt_row["max_new_tokens"])):
        selected = logits.argmax(dim=-1); token = int(selected.item())
        if token == int(tokenizer.eos_token_id): break
        candidate = [*generated, token]
        if weak:
            guard_started = time.perf_counter(); collapse = repetition_collapse_v2(tokenizer.decode(candidate, skip_special_tokens=True, clean_up_tokenization_spaces=False)); guard_seconds += time.perf_counter() - guard_started
            if collapse: terminated = True; break
        generated.append(token)
        if first is None and capability != "abstention": first = time.perf_counter() - started
        result = model(selected[:, None], task_routes=route, past_key_values=cache, use_cache=True); cache = result["past_key_values"]; logits = result["logits"][:, -1]
    value = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False); prefixed = False
    if weak and capability == "abstention" and not any(marker.casefold() in value.casefold() for marker in markers): value = clause + (" " + value if value else ""); prefixed = True
    total = time.perf_counter() - started
    if first is None or capability == "abstention": first = total
    final_ids = [int(value_id) for value_id in tokenizer.encode(value, add_special_tokens=False)]; output_bytes = len(value.encode("utf-8")); output_chars = len(value)
    return {"probe_id": prompt_row["probe_id"], "capability": expected, "routed_capability": capability, "route_correct": capability == expected, "task_route": int(route.item()), "output": value, "output_token_ids": final_ids, "output_utf8_bytes": output_bytes, "output_characters": output_chars, "authoritative_output_tokens": len(final_ids), "time_to_first_output_seconds": first, "total_seconds": total, "bytes_per_second": output_bytes / total if total else 0.0, "characters_per_second": output_chars / total if total else 0.0, "guard_seconds": guard_seconds, "guard_terminated": terminated, "abstention_clause_prefixed": prefixed, "router_segment_count": len(details)}


def _metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"observations": len(rows), "median_bytes_per_second": statistics.median(row["bytes_per_second"] for row in rows), "median_characters_per_second": statistics.median(row["characters_per_second"] for row in rows), "median_time_to_first_output_seconds": statistics.median(row["time_to_first_output_seconds"] for row in rows), "median_total_seconds": statistics.median(row["total_seconds"] for row in rows), "p95_supported": len(rows) >= 100, "p99_supported": len(rows) >= 1000}


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable CPU runtime output exists: {output}")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    probes = development_probes(root / protocol["development"]["catalog"]); distinct, runtime_probes = select_runtime_probes(probes, int(protocol["runtime"]["distinct_prompts"]), int(protocol["runtime"]["repeated_observations"])); reference = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["development"]["candidate_reference_outputs"]).open(encoding="utf-8"))}
    base_url = str(protocol["transformer_baseline"]["base_url"]); qwen_model = str(protocol["transformer_baseline"]["model"]); idle_rss = _ollama_rss(); unload_seconds = _ollama_unload(base_url, qwen_model); idle_rss = _ollama_rss()
    with PeakMonitor(_ollama_rss) as qwen_monitor:
        qwen_cold = _ollama_request(base_url, qwen_model, distinct[0], str(protocol["runtime"]["keep_alive"])); qwen_cold["single_cold_request"] = True
        for probe in distinct[:int(protocol["runtime"]["warmup_observations"])]: _ollama_request(base_url, qwen_model, probe, str(protocol["runtime"]["keep_alive"]))
        qwen_rows = [_ollama_request(base_url, qwen_model, probe, str(protocol["runtime"]["keep_alive"])) for probe in runtime_probes]
    qwen_peak_active_rss = max(0, qwen_monitor.peak - idle_rss)

    process = psutil.Process(); before_load_rss = process.memory_info().rss; load_started = time.perf_counter()
    screen_protocol = _json(root / protocol["candidate"]["screen_protocol"]); model, tokenizer, _ = _load_parent(root, screen_protocol, torch.device("cpu")); residual = SharedWeakResidual(); residual.load_state_dict(load_file(str(root / screen_protocol["candidate"]["checkpoint"]), device="cpu"), strict=True); residual.eval(); handles = _attach(model, residual); router, router_tokenizer, router_protocol = _load_router(root, screen_protocol); markers = artifact_markers(root / screen_protocol["guard"]["artifact"]); clause = str(screen_protocol["guard"]["canonical_abstention_clause"]); model.eval(); load_seconds = time.perf_counter() - load_started
    with PeakMonitor(lambda: process.memory_info().rss) as candidate_monitor:
        candidate_cold = _layercake_request(model, tokenizer, router, router_tokenizer, router_protocol, distinct[0], markers, clause, True); candidate_cold["single_cold_request"] = True; candidate_cold["model_load_seconds"] = load_seconds; candidate_cold["time_to_first_output_from_cold_start_seconds"] = load_seconds + candidate_cold["time_to_first_output_seconds"]; candidate_cold["total_from_cold_start_seconds"] = load_seconds + candidate_cold["total_seconds"]
        for probe in distinct[:int(protocol["runtime"]["warmup_observations"])]: _layercake_request(model, tokenizer, router, router_tokenizer, router_protocol, probe, markers, clause, True)
        candidate_rows = [_layercake_request(model, tokenizer, router, router_tokenizer, router_protocol, probe, markers, clause, True) for probe in runtime_probes]
    candidate_peak_active_rss = max(0, candidate_monitor.peak - before_load_rss); candidate_active_tensor_bytes = _tensor_bytes(model) + _tensor_bytes(residual) + _tensor_bytes(router)
    for handle in handles: handle.remove()
    for probe in distinct[:int(protocol["runtime"]["warmup_observations"])]: _layercake_request(model, tokenizer, router, router_tokenizer, router_protocol, probe, markers, clause, False)
    parent_rows = [_layercake_request(model, tokenizer, router, router_tokenizer, router_protocol, probe, markers, clause, False) for probe in runtime_probes]
    exact_reference = sum(row["output"] == str(reference[str(row["probe_id"])]["output"]) and row["output_token_ids"] == reference[str(row["probe_id"])]["output_token_ids"] for row in candidate_rows)
    candidate_metrics = _metrics(candidate_rows); qwen_metrics = _metrics(qwen_rows); parent_metrics = _metrics(parent_rows); comparison = paired_ratio_bootstrap([row["bytes_per_second"] for row in candidate_rows], [row["bytes_per_second"] for row in qwen_rows], int(protocol["statistics"]["bootstrap_replicates"]), int(protocol["statistics"]["bootstrap_seed"])); throughput_ratio = candidate_metrics["median_bytes_per_second"] / qwen_metrics["median_bytes_per_second"]; ttft_ratio = candidate_metrics["median_time_to_first_output_seconds"] / qwen_metrics["median_time_to_first_output_seconds"]; retention = candidate_metrics["median_bytes_per_second"] / parent_metrics["median_bytes_per_second"]; guard_fraction = sum(row["guard_seconds"] for row in candidate_rows) / sum(row["total_seconds"] for row in candidate_rows)
    gates_cfg = protocol["gates"]; gates = {"runtime_outputs_exact_to_quality_candidate": exact_reference == len(candidate_rows), "qualified_router_exact": all(row["route_correct"] for row in candidate_rows), "warm_observation_depth": len(candidate_rows) >= 120 and len(qwen_rows) >= 120, "distinct_prompt_depth": len({row["probe_id"] for row in candidate_rows}) >= 100, "candidate_throughput_ratio_at_least_2x": throughput_ratio >= float(gates_cfg["cpu_throughput_ratio_minimum"]), "paired_bootstrap_lower_at_least_2x": comparison["lower_95"] >= float(gates_cfg["paired_bootstrap_lower_minimum"]), "phase2_host_throughput_retention": retention >= float(gates_cfg["phase2_host_throughput_retention_minimum"]), "ttft_advantage": ttft_ratio <= float(gates_cfg["ttft_ratio_maximum"]), "lower_active_tensor_bytes": candidate_active_tensor_bytes < int(protocol["transformer_baseline"]["model_file_bytes"]), "lower_peak_active_rss": candidate_peak_active_rss < qwen_peak_active_rss, "genuine_candidate_cold_single_request": candidate_cold["single_cold_request"] is True, "genuine_qwen_cold_single_request": qwen_cold["single_cold_request"] is True and qwen_cold["load_seconds_reported"] > 0, "guard_overhead_bounded": guard_fraction <= float(gates_cfg["guard_time_fraction_maximum"]), "teacher_absent_at_inference": True, "final_test_not_accessed": True}; passed = all(gates.values())
    output.mkdir(parents=True); raw = output / "observations.jsonl"; raw.write_bytes(b"".join(canonical_json_bytes({"system": system, **row}) for system, rows in (("candidate", candidate_rows), ("parent", parent_rows), ("qwen", qwen_rows)) for row in rows))
    result = {"format": FORMAT, "status": "PASS_CPU_RUNTIME_PHASE3_CERTIFICATE_REVIEW_OPEN" if passed else "FAIL_CPU_RUNTIME_GATE_CLOSED", "protocol_sha256": protocol_sha, "hardware": {"logical_cpu_count": psutil.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}, "prompt_depth": {"distinct": len(distinct), "repeated": int(protocol["runtime"]["repeated_observations"]), "observations_per_system": len(runtime_probes)}, "candidate": {**candidate_metrics, "cold": candidate_cold, "peak_active_rss_delta_bytes": candidate_peak_active_rss, "active_tensor_bytes": candidate_active_tensor_bytes, "outputs_exact_to_v494": exact_reference, "guard_time_fraction": guard_fraction}, "phase2_host_without_bridge": parent_metrics, "optimized_transformer": {**qwen_metrics, "model": qwen_model, "digest": protocol["transformer_baseline"]["digest"], "model_file_bytes": protocol["transformer_baseline"]["model_file_bytes"], "peak_active_rss_delta_bytes": qwen_peak_active_rss, "cold": qwen_cold, "unload_control_seconds": unload_seconds}, "comparisons": {"candidate_to_qwen_median_bytes_per_second_ratio": throughput_ratio, "candidate_to_qwen_median_ttft_ratio": ttft_ratio, "candidate_to_parent_median_throughput_retention": retention, "paired_throughput": comparison}, "gates": gates, "passed": passed, "raw_observations_sha256": sha256_file(raw), "teacher_present_at_inference": False, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False}; result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); _ollama_unload(base_url, qwen_model); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CPU_RUNTIME_PROTOCOL_V503.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_cpu_runtime/runtime_v504"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
