"""Explicitly batched repair of routed-v16 late-layer sequence conformance."""

from __future__ import annotations

from collections import defaultdict
import argparse, hashlib, json, os
from pathlib import Path
import sys, time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_routed_v16_sequence_conformance as prior
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import controlled_prompt

FORMAT = "abi-capability-compiler-phase3-routed-v16-sequence-conformance-batched/1"


def _forward_batch(model, values, route, first):
    positions = torch.arange(values.shape[1], device=values.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.token_embedding(values)
        for index in range(first): hidden, _, _ = model.layers[index].forward_with_cache(hidden, positions, route)
    hidden = hidden.detach()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(first, 32): hidden, _, _ = model.layers[index].forward_with_cache(hidden, positions, route)
        return model.lm_head(model.final_norm(hidden))


def _batch_loss(model, rows, route, first):
    sequences = [row["source_ids"] + row["target_actions"][:-1] for row in rows]; maximum = max(map(len, sequences))
    values = torch.zeros((len(rows), maximum), dtype=torch.long, device="cuda")
    for index, sequence in enumerate(sequences): values[index, :len(sequence)] = torch.tensor(sequence, device="cuda")
    logits = _forward_batch(model, values, route, first); selected, targets = [], []
    for index, row in enumerate(rows):
        start = len(row["source_ids"]) - 1; target = row["target_actions"]; selected.append(logits[index, start:start + len(target)]); targets.extend(target)
    return F.cross_entropy(torch.cat(selected).float(), torch.tensor(targets, dtype=torch.long, device="cuda"))


def _evaluate(model, rows, first, batch_size):
    total = correct = exact = first_correct = route_correct = 0; per = defaultdict(lambda: [0, 0])
    grouped = defaultdict(list)
    for row in rows: grouped[row["route"]].append(row)
    with torch.inference_mode():
        for route, values in sorted(grouped.items()):
            for offset in range(0, len(values), batch_size):
                batch = values[offset:offset + batch_size]; sequences = [row["source_ids"] + row["target_actions"][:-1] for row in batch]; maximum = max(map(len, sequences)); tensor = torch.zeros((len(batch), maximum), dtype=torch.long, device="cuda")
                for index, sequence in enumerate(sequences): tensor[index, :len(sequence)] = torch.tensor(sequence, device="cuda")
                logits = _forward_batch(model, tensor, route, first)
                for index, row in enumerate(batch):
                    source_tensor = torch.tensor([row["source_ids"]], dtype=torch.long, device="cuda"); route_correct += int(model._select_route(source_tensor) == row["route"])
                    start = len(row["source_ids"]) - 1; target = row["target_actions"]; predicted = logits[index, start:start + len(target)].argmax(-1).tolist(); matches = sum(a == b for a, b in zip(predicted, target)); total += len(target); correct += matches; exact += int(predicted == target); first_correct += int(predicted[0] == target[0]); per[row["capability"]][0] += matches; per[row["capability"]][1] += len(target)
    return {"actions": total, "action_accuracy": correct / total, "exact_sequences": exact, "sequence_accuracy": exact / len(rows), "first_token_accuracy": first_correct / len(rows), "route_correct": route_correct, "records": len(rows), "per_capability_action_accuracy": {name: values[0] / values[1] for name, values in sorted(per.items())}}


def execute(root, protocol_path, output):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_EXPLICIT_BATCHED_SEQUENCE_CONFORMANCE_REPAIR" or protocol.get("teacher_model_access") != "PROHIBITED" or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("batched conformance governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected: raise Phase3Error(f"batched conformance binding changed: {name}")
    if output.exists() or not torch.cuda.is_available(): raise Phase3Error("batched output exists or CUDA unavailable")
    output.mkdir(parents=True); training = protocol["training"]; set_determinism(int(training["seed"])); torch.use_deterministic_algorithms(True)
    base_protocol = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8")); artifact = (root / protocol["artifact"]["directory"]).resolve(); config = json.loads((artifact / "config.json").read_text(encoding="utf-8")); sys.path.insert(0, str((root / protocol["layercake_host"]["repository"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"]); original = load_file(str(artifact / "model.safetensors"), device="cpu"); model = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer); model.load_state_dict(original, strict=True); model = model.cuda()
    first = int(training["first_trainable_layer"]); trainable_names = {name for name, _ in model.named_parameters() if any(name.startswith(f"layers.{index}.") for index in range(first, 32))}
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name in trainable_names)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(value.numel() for value in parameters) != int(training["trainable_parameters"]): raise Phase3Error("batched trainable boundary changed")
    examples = dual.field._examples(root, base_protocol, tokenizer); by_id = {row["record_id"]: row for row in examples}; cfg = base_protocol["calibration"]; _, validation_source, _ = dual._calibration_examples(examples, seed=int(base_protocol["training"]["seed"]), train_per_capability=int(cfg["train_records_per_capability"]), validation_per_capability=int(cfg["validation_records_per_capability"]), maximum_tokens=int(cfg["maximum_sequence_tokens"])); held = {row["record_id"] for row in validation_source}; grouped = defaultdict(list)
    for row in examples:
        if row["record_id"] not in held: grouped[row["capability"]].append(row)
    for capability in CAPABILITIES: grouped[capability].sort(key=lambda row: hashlib.sha256(f"{training['seed']}:{row['record_id']}".encode()).digest())
    def route(capability): return 1 if capability == "abstention" else 2 if capability == "conversation" else 0
    calibration = [{**by_id[row["record_id"]], "route": route(row["capability"])} for row in validation_source]
    teacher_rows = {row["probe_id"]: row for row in map(json.loads, (root / protocol["teacher_reference"]).read_text(encoding="utf-8").splitlines())}; development = []
    terminal = int(base_protocol["source"]["terminal_token_id"])
    for probe in development_probes(root / protocol["development_catalog"]):
        capability = probe["canonical_capability"]; source, _ = tokenizer.encode_source(controlled_prompt(capability, probe["prompt"])); target = [prior.EOS if value == terminal else prior.OFFSET + value for value in teacher_rows[probe["probe_id"]]["output_token_ids"]]; development.append({"record_id": probe["probe_id"], "capability": capability, "source_ids": source, "target_actions": target, "route": route(capability)})
    optimizer = torch.optim.AdamW(parameters, lr=float(training["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(training["weight_decay"])); batch_size = int(training["batch_size"]); steps = int(training["steps"]); curves = []; digest = hashlib.sha256(); process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats(); started = time.perf_counter(); model.train()
    for step in range(steps):
        capability = CAPABILITIES[step % 14]; values = grouped[capability]; block = step // 14; batch = [values[(block * batch_size + index) % len(values)] for index in range(batch_size)]
        for row in batch: digest.update((row["record_id"] + "\n").encode())
        optimizer.zero_grad(set_to_none=True); loss = _batch_loss(model, batch, route(capability), first); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip_norm"])); optimizer.step()
        if not torch.isfinite(loss): raise Phase3Error("nonfinite batched loss")
        peak_rss = max(peak_rss, process.memory_info().rss)
        if (step + 1) % int(training["curve_interval"]) == 0: curves.append({"step": step + 1, "records": (step + 1) * batch_size, "loss": float(loss), "gradient_norm": float(norm)}); print(json.dumps(curves[-1]), flush=True)
    model.eval(); calibration_result = _evaluate(model, calibration, first, batch_size); development_result = _evaluate(model, development, first, batch_size); frozen_exact = all(torch.equal(parameter.detach().half().cpu(), original[name]) for name, parameter in model.named_parameters() if name not in trainable_names)
    gates = {"calibration_action_accuracy": calibration_result["action_accuracy"] >= protocol["gates"]["calibration_action_accuracy_minimum"], "development_action_accuracy": development_result["action_accuracy"] >= protocol["gates"]["development_action_accuracy_minimum"], "development_per_capability": min(development_result["per_capability_action_accuracy"].values()) >= protocol["gates"]["development_per_capability_action_accuracy_minimum"], "development_first_token": development_result["first_token_accuracy"] >= protocol["gates"]["development_first_token_accuracy_minimum"], "routes_exact": calibration_result["route_correct"] == 28 and development_result["route_correct"] == 1400, "frozen_exact": frozen_exact}; passed = all(gates.values()); checkpoint = None
    if passed:
        path = output / "late_layers_29_31.safetensors"; save_file({name: parameter.detach().half().cpu().contiguous() for name, parameter in model.named_parameters() if name in trainable_names}, str(path), metadata={"format": FORMAT}); checkpoint = {"path": path.name, "sha256": sha256_file(path), "tensor_keys": len(trainable_names), "parameters": sum(parameter.numel() for name, parameter in model.named_parameters() if name in trainable_names)}
    result = {"format": FORMAT, "status": "PASS_BATCHED_SEQUENCE_CONFORMANCE" if passed else "FAIL_BATCHED_SEQUENCE_CONFORMANCE", "protocol_sha256": sha256_file(protocol_path), "training": {"steps": steps, "batch_size": batch_size, "record_exposures": steps * batch_size, "trainable_parameters": sum(p.numel() for p in parameters), "record_sequence_sha256": digest.hexdigest(), "curves": curves, "wall_seconds": time.perf_counter() - started}, "calibration_validation": calibration_result, "development_teacher_forced": development_result, "gates": gates, "passed": passed, "checkpoint": checkpoint, "frozen_parameters_exact": frozen_exact, "teacher_model_loaded": False, "final_test_accessed": False, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "phase3_certified": False, "claim_boundary": "Batched hybrid weight-transfer plus cached-sequence conformance feasibility; not autonomous quality, runtime, Phase 3, or superiority proof."}; result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V16_SEQUENCE_CONFORMANCE_BATCHED_PROTOCOL_V339.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_routed_v16/sequence_conformance_batched_v340"); args = parser.parse_args(); root = Path.cwd().resolve(); print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
