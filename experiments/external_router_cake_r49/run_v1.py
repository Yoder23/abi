"""Execute the preregistered R49 external label-router cake experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from abi.english_generalization_evaluation import _collapse_metrics, _source_by_probe
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_host import (
    CAPABILITY_TO_ROUTE,
    _select_next_token,
    _sha256_file,
    _truncate_novel_lexical_repetition,
)
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


CHECKPOINT_SHA256 = "65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b"
METADATA_SHA256 = "9590374b0afd3184dd75bcf08d8b9f0876ed7bcc0db7f7ec1f4abf147093d1d6"
CATALOG_SHA256 = "8992c7de94d3733d66f2083f96ec8d3cba31be3943afe1849f42220f33ff8d08"
SOURCE_SHA256 = (
    "85abf114ffd6455d589d8b55b740becd920bc2e4dd4f3f62fddffe4b1708af2c",
    "fe9af246e28efdf57e234e9384fa04c21974f997fe753ac0975d67b1d0c6ecea",
)
SEED = 49_001
FEATURES = 1_024
STEPS = 300
LEARNING_RATE = 0.05
WEIGHT_DECAY = 0.001


class R49Error(RuntimeError):
    pass


def _feature(prompt: str) -> torch.Tensor:
    words = ["<num>" if token.isdigit() else token for token in re.findall(r"[a-z]+|\d+", prompt.casefold())]
    values = [f"u:{word}" for word in words]
    values.extend(f"b:{left}\0{right}" for left, right in zip(words, words[1:]))
    vector = torch.zeros(FEATURES, dtype=torch.float32)
    for value in values:
        digest = hashlib.sha256(value.encode()).digest()
        slot = int.from_bytes(digest[:8], "big") % FEATURES
        vector[slot] += 1.0 if digest[8] & 1 else -1.0
    norm = float(vector.norm())
    if norm:
        vector /= norm
    return vector


def _matrix(rows: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack([_feature(str(row["prompt"])) for row in rows])
    labels = torch.tensor([CAPABILITY_TO_ROUTE[str(row["capability"])] for row in rows], dtype=torch.long)
    return inputs, labels


def _score(router: torch.nn.Linear, inputs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        predictions = router(inputs).argmax(dim=-1)
    correct = int((predictions == labels).sum())
    rotated = (labels + 1) % 10
    return {
        "rows": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels),
        "rotated_correct": int((predictions == rotated).sum()),
        "rotated_accuracy": float((predictions == rotated).float().mean()),
        "predictions_sha256": hashlib.sha256(bytes(int(value) for value in predictions.cpu())).hexdigest(),
    }


@torch.inference_mode()
def _generate(model: Any, tokenizer: Any, prompt: str, route: int, maximum: int, device: torch.device):
    prompt_ids = tokenizer.encode(prompt + "\n")
    if len(prompt_ids) + maximum > model.config.max_tokens:
        raise R49Error("R49 validation prompt exceeds context")
    route_tensor = torch.tensor([route], dtype=torch.long, device=device)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    result = model(ids, prompt_lengths=torch.tensor([len(prompt_ids)], device=device), task_routes=route_tensor, use_cache=True)
    physical = [tuple(model.last_cake_calls)]
    generated: list[int] = []
    state = {"past_key_values": result["past_key_values"], "next_logits": result["logits"][:, -1]}
    for _ in range(maximum):
        selected = _select_next_token(state["next_logits"][0], generated=generated, no_repeat_ngram_size=0).to(device)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        result = model(selected[:, None], task_routes=route_tensor, past_key_values=state["past_key_values"], use_cache=True)
        physical.append(tuple(model.last_cake_calls))
        state = {"past_key_values": result["past_key_values"], "next_logits": result["logits"][:, -1]}
    elapsed = time.perf_counter() - started
    output = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output = _truncate_novel_lexical_repetition(output, prompt, threshold=1)
    generated = tokenizer.encode(output)
    return output, generated, elapsed, physical


def _bootstrap(candidate: Sequence[bool], source: Sequence[bool]) -> dict[str, Any]:
    differences = [float(a) - float(b) for a, b in zip(candidate, source)]
    rng = random.Random(SEED + 1)
    values = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(5_000)
    )
    return {"point": sum(differences) / len(differences), "lower_95": values[124], "upper_95": values[4874], "replicates": 5_000}


def run(candidate: Path, catalog_path: Path, source_paths: Sequence[Path], layercake_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R49Error(f"immutable R49 output exists: {output}")
    if (
        _sha256_file(candidate / "model.safetensors") != CHECKPOINT_SHA256
        or _sha256_file(candidate / "metadata.json") != METADATA_SHA256
        or _sha256_file(catalog_path) != CATALOG_SHA256
        or tuple(_sha256_file(path) for path in source_paths) != SOURCE_SHA256
    ):
        raise R49Error("R49 frozen input changed")
    if not torch.cuda.is_available():
        raise R49Error("R49 integrated screen requires CUDA")
    catalog = load_probe_catalog(catalog_path)
    search = [dict(row) for row in catalog["probes"] if row["split"] == "search"]
    validation = [dict(row) for row in catalog["probes"] if row["split"] == "validation"]
    if len(search) != 1_400 or len(validation) != 1_400:
        raise R49Error("R49 catalog cardinality changed")
    search_x, search_y = _matrix(search)
    validation_x, validation_y = _matrix(validation)
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    router = torch.nn.Linear(FEATURES, 10).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_x, train_y = search_x.to(device), search_y.to(device)
    trace = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = router(train_x)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0:
            row = {"step": step, "loss": float(loss.detach()), "accuracy": float((logits.argmax(dim=-1) == train_y).float().mean())}
            trace.append(row)
            print(json.dumps(row), flush=True)
    router.eval()
    search_score = _score(router, train_x, train_y)
    validation_score = _score(router, validation_x.to(device), validation_y.to(device))
    router_passed = (
        search_score["accuracy"] == validation_score["accuracy"] == 1.0
        and validation_score["rotated_accuracy"] <= 0.20
    )
    output.mkdir(parents=True)
    router_path = output / "router.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in router.state_dict().items()}, str(router_path), metadata={"format": "abi-r49-sparse-label-router/1"})
    router_receipt = {
        "format": "abi-r49-sparse-label-router/1",
        "checkpoint_sha256": _sha256_file(router_path),
        "checkpoint_bytes": router_path.stat().st_size,
        "parameters": sum(value.numel() for value in router.parameters()),
        "features": FEATURES,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "teacher_calls": 0,
        "response_targets": 0,
        "search": search_score,
        "validation": validation_score,
        "trace": trace,
        "wall_seconds": time.perf_counter() - started,
        "passed": router_passed,
    }
    write_json_once(output / "router.json", router_receipt)
    if not router_passed:
        raise R49Error("R49 router prerequisite failed; integrated screen prohibited")

    source, source_identities = _source_by_probe(source_paths, split="validation")
    if set(source) != {str(row["probe_id"]) for row in validation}:
        raise R49Error("R49 source comparison matrix changed")
    model, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
    model.eval()
    rows = []
    generation_started = time.perf_counter()
    for index, probe in enumerate(validation, 1):
        route = int(router(_feature(str(probe["prompt"])).to(device)).argmax())
        generated, tokens, latency, physical = _generate(
            model, tokenizer, str(probe["prompt"]), route, int(probe["max_new_tokens"]), device
        )
        passed, score = evaluate_output(generated, probe["evaluator"])
        source_row = source[str(probe["probe_id"])]
        rows.append({
            "probe_id": probe["probe_id"],
            "capability": probe["capability"],
            "prompt": probe["prompt"],
            "prompt_sha256": hashlib.sha256(str(probe["prompt"]).encode()).hexdigest(),
            "evaluator": probe["evaluator"],
            "output": generated,
            "output_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            "output_token_ids": tokens,
            "functional_pass": passed,
            "functional_score": score,
            "route": route,
            "expected_route": CAPABILITY_TO_ROUTE[str(probe["capability"])],
            "route_correct": route == CAPABILITY_TO_ROUTE[str(probe["capability"])],
            "maximum_cakes_called_per_model_invocation": max(map(len, physical)),
            "all_physical_calls": len(physical),
            "all_calls_selected_only": all(value == (route,) for value in physical),
            "latency_seconds": latency,
            "collapse": _collapse_metrics(tokens, generated, tokenizer.encode(str(probe["prompt"]) + "\n"), str(probe["prompt"])),
            "generation_error": None,
            "source": source_row,
            "source_passing_regression": bool(source_row["passed"] and not passed),
        })
        if index % 100 == 0:
            print(json.dumps({"evaluated": index, "functional": sum(row["functional_pass"] for row in rows), "collapses": sum(row["collapse"]["collapse_detected"] for row in rows)}), flush=True)
    rows_path = output / "evaluation.jsonl"
    write_jsonl_once(rows_path, rows)
    by_capability = {
        capability: {
            "rows": sum(row["capability"] == capability for row in rows),
            "functional": sum(row["capability"] == capability and row["functional_pass"] for row in rows),
            "source_functional": sum(row["capability"] == capability and row["source"]["passed"] for row in rows),
            "collapses": sum(row["capability"] == capability and row["collapse"]["collapse_detected"] for row in rows),
            "route_correct": sum(row["capability"] == capability and row["route_correct"] for row in rows),
        }
        for capability in sorted(CAPABILITY_TO_ROUTE)
    }
    functional = sum(row["functional_pass"] for row in rows)
    source_functional = sum(row["source"]["passed"] for row in rows)
    regressions = sum(row["source_passing_regression"] for row in rows)
    metrics = {
        "rows": len(rows),
        "functional": functional,
        "source_functional": source_functional,
        "source_passing_regressions": regressions,
        "source_retention": (source_functional - regressions) / source_functional,
        "candidate_minus_source": _bootstrap([row["functional_pass"] for row in rows], [row["source"]["passed"] for row in rows]),
        "collapses": sum(row["collapse"]["collapse_detected"] for row in rows),
        "generation_errors": sum(row["generation_error"] is not None for row in rows),
        "route_correct": sum(row["route_correct"] for row in rows),
        "physical_sparse_rows": sum(row["all_calls_selected_only"] and row["maximum_cakes_called_per_model_invocation"] == 1 for row in rows),
        "by_capability": by_capability,
        "generation_wall_seconds": time.perf_counter() - generation_started,
    }
    gates = {
        "router": router_passed,
        "matrix": len(rows) == 1_400,
        "functional": functional >= 1_260,
        "per_capability": all(value["functional"] >= 65 for value in by_capability.values()),
        "source_noninferior_point": functional >= source_functional,
        "source_retention": metrics["source_retention"] >= 0.94,
        "zero_collapse": metrics["collapses"] == 0,
        "zero_generation_error": metrics["generation_errors"] == 0,
        "route_exact": metrics["route_correct"] == 1_400,
        "physical_sparse": metrics["physical_sparse_rows"] == 1_400,
        "candidate_unchanged": _sha256_file(candidate / "model.safetensors") == CHECKPOINT_SHA256 and _sha256_file(candidate / "metadata.json") == METADATA_SHA256,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-r49-external-router-cake/1",
        "verdict": "PASS_R49_EXTERNAL_ROUTER_CAKE" if passed else "FAIL_R49_EXTERNAL_ROUTER_CAKE",
        "inputs": {"candidate_checkpoint_sha256": CHECKPOINT_SHA256, "candidate_metadata_sha256": METADATA_SHA256, "catalog_sha256": CATALOG_SHA256, "source_sha256": list(SOURCE_SHA256)},
        "router": router_receipt,
        "source_identities": source_identities,
        "metrics": metrics,
        "gates": gates,
        "artifacts": {"router": {"path": router_path.name, "sha256": _sha256_file(router_path), "bytes": router_path.stat().st_size}, "evaluation": {"path": rows_path.name, "sha256": _sha256_file(rows_path), "bytes": rows_path.stat().st_size}},
        "teacher_present_at_inference": False,
        "symbolic_output_calls": 0,
        "planner_calls": 0,
        "claim_ceiling": "BOUNDED_LABELED_ROUTER_PLUS_NEURAL_HOST_NOT_UNRESTRICTED_ENGLISH_OR_MINIMALITY",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.candidate.resolve(), args.catalog.resolve(), [path.resolve() for path in args.source_bundle], args.layercake_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
