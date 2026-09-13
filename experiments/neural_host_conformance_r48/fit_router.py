"""Fit the preregistered R48 label-only LayerCake routing bridge."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from abi.artifacts import _tensor_bytes
from abi.hf_extraction import load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import CAPABILITY_TO_ROUTE, _sha256_file


CHECKPOINT_SHA256 = "65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b"
METADATA_SHA256 = "9590374b0afd3184dd75bcf08d8b9f0876ed7bcc0db7f7ec1f4abf147093d1d6"
CATALOG_SHA256 = "8992c7de94d3733d66f2083f96ec8d3cba31be3943afe1849f42220f33ff8d08"
SEED = 48_001
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.01


class R48Error(RuntimeError):
    pass


def _state_hash(model: Any, *, classifier: bool) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        selected = name.startswith("task_classifier.")
        if selected != classifier:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _batches(rows: list[dict[str, Any]], rng: random.Random) -> Iterable[list[dict[str, Any]]]:
    order = list(rows)
    rng.shuffle(order)
    for start in range(0, len(order), BATCH_SIZE):
        yield order[start : start + BATCH_SIZE]


def _tensorize(rows: list[dict[str, Any]], tokenizer: Any, device: torch.device):
    encoded = [tokenizer.encode(str(row["prompt"]) + "\n") for row in rows]
    maximum = max(map(len, encoded))
    inputs = torch.zeros((len(rows), maximum), dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    for index, values in enumerate(encoded):
        inputs[index, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[index, : len(values)] = 1
    labels = torch.tensor(
        [CAPABILITY_TO_ROUTE[str(row["capability"])] for row in rows],
        dtype=torch.long,
        device=device,
    )
    return inputs, mask, labels


@torch.inference_mode()
def _score(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    correct = 0
    predictions: list[int] = []
    expected: list[int] = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        inputs, mask, labels = _tensorize(batch, tokenizer, device)
        hidden = model.transformer(input_ids=inputs, attention_mask=mask, return_dict=True).last_hidden_state
        summary = model._prompt_summary(hidden, prompt_lengths=None, attention_mask=mask)
        predicted = model.task_classifier(summary).argmax(dim=-1)
        correct += int((predicted == labels).sum())
        predictions.extend(int(value) for value in predicted.cpu())
        expected.extend(int(value) for value in labels.cpu())
    rotated = [(value + 1) % len(model.task_cakes) for value in expected]
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "rotated_correct": sum(a == b for a, b in zip(predictions, rotated)),
        "rotated_accuracy": sum(a == b for a, b in zip(predictions, rotated)) / len(rows),
        "prediction_sha256": hashlib.sha256(bytes(predictions)).hexdigest(),
    }


def run(candidate: Path, catalog_path: Path, layercake_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R48Error(f"immutable R48 output exists: {output}")
    if (
        _sha256_file(candidate / "model.safetensors") != CHECKPOINT_SHA256
        or _sha256_file(candidate / "metadata.json") != METADATA_SHA256
        or _sha256_file(catalog_path) != CATALOG_SHA256
    ):
        raise R48Error("R48 frozen input changed")
    if not torch.cuda.is_available():
        raise R48Error("R48 preregistration requires CUDA")
    catalog = load_probe_catalog(catalog_path)
    search = [dict(row) for row in catalog["probes"] if row["split"] == "search"]
    validation = [dict(row) for row in catalog["probes"] if row["split"] == "validation"]
    if len(search) != 1_400 or len(validation) != 1_400:
        raise R48Error("R48 catalog split changed")
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, tokenizer, metadata = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
    frozen_before = _state_hash(model, classifier=False)
    classifier_before = _state_hash(model, classifier=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.task_classifier.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.task_classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rng = random.Random(SEED)
    trace = []
    started = time.perf_counter()
    model.eval()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        correct = 0
        seen = 0
        for rows in _batches(search, rng):
            inputs, mask, labels = _tensorize(rows, tokenizer, device)
            with torch.no_grad():
                hidden = model.transformer(input_ids=inputs, attention_mask=mask, return_dict=True).last_hidden_state
                summary = model._prompt_summary(hidden, prompt_lengths=None, attention_mask=mask)
            optimizer.zero_grad(set_to_none=True)
            logits = model.task_classifier(summary)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(dim=-1) == labels).sum())
            seen += len(rows)
        row = {"epoch": epoch, "loss": sum(losses) / len(losses), "accuracy": correct / seen}
        trace.append(row)
        print(json.dumps(row), flush=True)
    train_score = _score(model, tokenizer, search, device)
    validation_score = _score(model, tokenizer, validation, device)
    frozen_after = _state_hash(model, classifier=False)
    classifier_after = _state_hash(model, classifier=True)
    passed = (
        frozen_before == frozen_after
        and classifier_before != classifier_after
        and train_score["accuracy"] == 1.0
        and validation_score["accuracy"] >= 0.99
        and validation_score["rotated_accuracy"] <= 0.20
    )
    output.mkdir(parents=True)
    checkpoint = output / "model.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        str(checkpoint),
        metadata={"format": "abi-r48-label-only-router-conformance/1"},
    )
    tokenizer.save_pretrained(output)
    derived = copy.deepcopy(metadata)
    derived["status"] = "R48_ROUTER_CONFORMED_NOT_PROSPECTIVELY_CERTIFIED"
    derived["checkpoint"] = {
        "path": checkpoint.name,
        "sha256": _sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }
    derived["router_conformance"] = {
        "format": "abi-r48-label-only-router-conformance/1",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "trainable_parameters": sum(value.numel() for value in model.task_classifier.parameters()),
        "teacher_calls": 0,
        "response_targets": 0,
        "search": train_score,
        "validation": validation_score,
        "trace": trace,
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "classifier_sha256_before": classifier_before,
        "classifier_sha256_after": classifier_after,
        "wall_seconds": time.perf_counter() - started,
        "passed": passed,
    }
    derived["parent_layercake"] = {
        "path_at_training": str(candidate),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "metadata_sha256": METADATA_SHA256,
        "unchanged_on_disk": (
            _sha256_file(candidate / "model.safetensors") == CHECKPOINT_SHA256
            and _sha256_file(candidate / "metadata.json") == METADATA_SHA256
        ),
    }
    derived["claim_boundary"] = (
        "Label-only router conformance; neural English weights and task cakes are byte-identical. "
        "Integrated and prospective quality remain open."
    )
    derived["manifest_sha256"] = _manifest_sha(derived)
    (output / "metadata.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result = {
        "format": "abi-r48-router-fit-result/1",
        "verdict": "PASS_R48_ROUTER_FIT" if passed else "FAIL_R48_ROUTER_FIT",
        "checkpoint_sha256": derived["checkpoint"]["sha256"],
        "manifest_sha256": derived["manifest_sha256"],
        "frozen_state_unchanged": frozen_before == frozen_after,
        "classifier_parameters": derived["router_conformance"]["trainable_parameters"],
        "search": train_score,
        "validation": validation_score,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    (output / "router_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.candidate.resolve(), args.catalog.resolve(), args.layercake_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
