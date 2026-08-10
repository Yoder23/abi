"""Bounded late-layer sequence-conformance feasibility for routed-v16."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_dual_path_local_fit as dual
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import controlled_prompt


FORMAT = "abi-capability-compiler-phase3-routed-v16-sequence-conformance/1"
OFFSET, EOS = 4, 2


def _forward(model, values: torch.Tensor, route: int, first_trainable: int) -> torch.Tensor:
    positions = torch.arange(values.shape[1], device=values.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.token_embedding(values)
        for index in range(first_trainable): hidden, _, _ = model.layers[index].forward_with_cache(hidden, positions, route)
    hidden = hidden.detach()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(first_trainable, len(model.layers)): hidden, _, _ = model.layers[index].forward_with_cache(hidden, positions, route)
        return model.lm_head(model.final_norm(hidden))


def _loss(model, source: list[int], target: list[int], route: int, first_trainable: int) -> torch.Tensor:
    values = source + target[:-1]
    logits = _forward(model, torch.tensor([values], dtype=torch.long, device="cuda"), route, first_trainable)
    selected = logits[0, len(source) - 1 : len(source) - 1 + len(target)]
    return F.cross_entropy(selected.float(), torch.tensor(target, dtype=torch.long, device="cuda"))


def _evaluate(model, rows: list[dict], first_trainable: int) -> dict:
    total = correct = exact = first = routes = 0; per = defaultdict(lambda: [0, 0])
    with torch.inference_mode():
        for row in rows:
            source, target, expected = row["source_ids"], row["target_actions"], row["route"]
            source_tensor = torch.tensor([source], dtype=torch.long, device="cuda"); route = model._select_route(source_tensor)
            logits = _forward(model, source_tensor.new_tensor([source + target[:-1]]), route, first_trainable)
            predicted = logits[0, len(source) - 1 : len(source) - 1 + len(target)].argmax(-1).tolist()
            matches = sum(left == right for left, right in zip(predicted, target)); total += len(target); correct += matches
            sequence = int(predicted == target); exact += sequence; first += int(predicted[0] == target[0]); routes += int(route == expected)
            per[row["capability"]][0] += matches; per[row["capability"]][1] += len(target)
    return {"actions": total, "action_accuracy": correct / total, "exact_sequences": exact, "sequence_accuracy": exact / len(rows), "first_token_accuracy": first / len(rows), "route_correct": routes, "records": len(rows), "per_capability_action_accuracy": {name: values[0] / values[1] for name, values in sorted(per.items())}}


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_BOUNDED_LATE_LAYER_SEQUENCE_CONFORMANCE" or protocol.get("device") != "cuda" or protocol.get("teacher_model_access") != "PROHIBITED" or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("sequence-conformance governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"sequence-conformance binding changed: {name}")
    if output.exists() or not torch.cuda.is_available(): raise Phase3Error("sequence-conformance output exists or CUDA unavailable")
    output.mkdir(parents=True); set_determinism(int(protocol["training"]["seed"])); torch.use_deterministic_algorithms(True)
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8")); artifact = (root / protocol["artifact"]["directory"]).resolve(); config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve(); sys.path.insert(0, str(layercake_root))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"]); original = load_file(str(artifact / "model.safetensors"), device="cpu")
    model = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer); model.load_state_dict(original, strict=True); model = model.cuda()
    first_trainable = int(protocol["training"]["first_trainable_layer"]); trainable_names = {name for name, _ in model.named_parameters() if any(name.startswith(f"layers.{index}.") for index in range(first_trainable, 32))}
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name in trainable_names)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in trainable) != int(protocol["training"]["trainable_parameters"]): raise Phase3Error("sequence-conformance trainable boundary changed")
    examples = dual.field._examples(root, base, tokenizer); cfg = base["calibration"]
    _, calibration_validation, _ = dual._calibration_examples(examples, seed=int(base["training"]["seed"]), train_per_capability=int(cfg["train_records_per_capability"]), validation_per_capability=int(cfg["validation_records_per_capability"]), maximum_tokens=int(cfg["maximum_sequence_tokens"]))
    held = {str(row["record_id"]) for row in calibration_validation}; training_rows = [row for row in examples if str(row["record_id"]) not in held]
    grouped = defaultdict(list)
    for row in training_rows: grouped[row["capability"]].append(row)
    for capability in CAPABILITIES: grouped[capability].sort(key=lambda row: hashlib.sha256(f"{protocol['training']['seed']}:{row['record_id']}".encode()).digest())
    validation_rows = [{"record_id": row["record_id"], "capability": row["capability"], "source_ids": next(value["source_ids"] for value in examples if value["record_id"] == row["record_id"]), "target_actions": next(value["target_actions"] for value in examples if value["record_id"] == row["record_id"]), "route": 1 if row["capability"] == "abstention" else 2 if row["capability"] == "conversation" else 0} for row in calibration_validation]
    teacher_rows = {str(row["probe_id"]): row for row in map(json.loads, (root / protocol["teacher_reference"]).read_text(encoding="utf-8").splitlines())}; development = []
    for probe in development_probes(root / protocol["development_catalog"]):
        capability = str(probe["canonical_capability"]); source, _ = tokenizer.encode_source(controlled_prompt(capability, str(probe["prompt"]))); target = [EOS if int(value) == int(base["source"]["terminal_token_id"]) else OFFSET + int(value) for value in teacher_rows[str(probe["probe_id"])]["output_token_ids"]]
        development.append({"record_id": str(probe["probe_id"]), "capability": capability, "source_ids": source, "target_actions": target, "route": 1 if capability == "abstention" else 2 if capability == "conversation" else 0})
    optimizer = torch.optim.AdamW(trainable, lr=float(protocol["training"]["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(protocol["training"]["weight_decay"])); curves = []; process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats(); started = time.perf_counter(); model.train()
    steps = int(protocol["training"]["steps"]); sequence_digest = hashlib.sha256()
    for step in range(steps):
        capability = CAPABILITIES[step % len(CAPABILITIES)]; rows = grouped[capability]; row = rows[(step // len(CAPABILITIES)) % len(rows)]; sequence_digest.update((row["record_id"] + "\n").encode())
        route = 1 if capability == "abstention" else 2 if capability == "conversation" else 0; optimizer.zero_grad(set_to_none=True); loss = _loss(model, row["source_ids"], row["target_actions"], route, first_trainable); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(trainable, float(protocol["training"]["gradient_clip_norm"])); optimizer.step()
        if not torch.isfinite(loss): raise Phase3Error("nonfinite sequence-conformance loss")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if (step + 1) % int(protocol["training"]["curve_interval"]) == 0: curves.append({"step": step + 1, "loss": float(loss), "gradient_norm": float(norm)}); print(json.dumps(curves[-1]), flush=True)
    model.eval(); calibration = _evaluate(model, validation_rows, first_trainable); development_metrics = _evaluate(model, development, first_trainable)
    frozen_exact = all(torch.equal(parameter.detach().half().cpu(), original[name]) for name, parameter in model.named_parameters() if name not in trainable_names)
    gates = {"calibration_action_accuracy": calibration["action_accuracy"] >= float(protocol["gates"]["calibration_action_accuracy_minimum"]), "development_action_accuracy": development_metrics["action_accuracy"] >= float(protocol["gates"]["development_action_accuracy_minimum"]), "development_per_capability": min(development_metrics["per_capability_action_accuracy"].values()) >= float(protocol["gates"]["development_per_capability_action_accuracy_minimum"]), "development_first_token": development_metrics["first_token_accuracy"] >= float(protocol["gates"]["development_first_token_accuracy_minimum"]), "routes_exact": calibration["route_correct"] == 28 and development_metrics["route_correct"] == 1400, "frozen_exact": frozen_exact}
    passed = all(gates.values()); checkpoint = None
    if passed:
        path = output / "late_layers_29_31.safetensors"; save_file({name: parameter.detach().half().cpu().contiguous() for name, parameter in model.named_parameters() if name in trainable_names}, str(path), metadata={"format": FORMAT}); checkpoint = {"path": path.name, "sha256": sha256_file(path), "tensor_keys": len(trainable_names), "parameters": sum(parameter.numel() for name, parameter in model.named_parameters() if name in trainable_names)}
    result = {"format": FORMAT, "status": "PASS_SEQUENCE_CONFORMANCE_FEASIBILITY" if passed else "FAIL_SEQUENCE_CONFORMANCE_FEASIBILITY", "protocol_sha256": sha256_file(protocol_path), "artifact_model_sha256": protocol["artifact"]["model_sha256"], "training": {"steps": steps, "train_records": len(training_rows), "trainable_parameters": sum(parameter.numel() for parameter in trainable), "record_sequence_sha256": sequence_digest.hexdigest(), "curves": curves, "wall_seconds": time.perf_counter() - started}, "calibration_validation": calibration, "development_teacher_forced": development_metrics, "gates": gates, "passed": passed, "checkpoint": checkpoint, "frozen_parameters_exact": frozen_exact, "teacher_model_loaded": False, "final_test_accessed": False, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "phase3_certified": False, "claim_boundary": "Bounded hybrid weight-transfer plus cached-sequence conformance feasibility; not autonomous quality, runtime, Phase 3, or superiority proof."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"); return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V16_SEQUENCE_CONFORMANCE_PROTOCOL_V337.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_routed_v16/sequence_conformance_v338"); args = parser.parse_args(); root = Path.cwd().resolve(); print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
