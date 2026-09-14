"""Train the one preregistered R85 span package from frozen ABI records."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core
from abi.layercake_full_core_acquisition import (
    _manifest_sha,
    load_english_training_rows,
)
from abi.layercake_host import _sha256_file
from experiments.stateful_span_r85.core_v1 import (
    BRIDGE_ARCHITECTURE,
    FEEDFORWARD_WIDTH,
    HEADS,
    LAYERS,
    MAX_SPAN_TOKENS,
    MAX_TOKENS,
    PACKAGE_FORMAT,
    WIDTH,
    StatefulSpanBridge,
    bridge_parameter_count,
    validate_metadata,
)


PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
ARTIFACT_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
SOURCE_RESULT_SHA256 = "35774db7db7aae192792454ddaf8c6c1fd2852026f90de41b3c76d11e3f035cc"
CATALOG_SHA256 = "8ad899dfb15f120ae3ffdbde6b6d8dcb06bbb265883665bcecdeff048eaafc4c"
ROUTE = CAPABILITY_CAKE_ORDER.index("domain_independent_reasoning")


class TrainingError(RuntimeError):
    pass


def _span_example(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"])
    response = str(row["response"])
    rendered = prompt + "\n"
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(int(part) for part in pair) for pair in encoded["offset_mapping"]]
    character_starts = [
        index for index in range(len(prompt)) if prompt.startswith(response, index)
    ]
    spans: list[tuple[int, int]] = []
    for character_start in character_starts:
        character_end = character_start + len(response)
        selected = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < character_end and end > character_start
        ]
        if not selected:
            continue
        start = min(selected)
        length = max(selected) - start + 1
        decoded = tokenizer.decode(
            ids[start : start + length],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if decoded == response:
            spans.append((start, length))
    spans = sorted(set(spans))
    if (
        len(character_starts) != 2
        or len(spans) != 2
        or len({length for _, length in spans}) != 1
        or not 1 <= spans[0][1] <= MAX_SPAN_TOKENS
        or len(ids) > MAX_TOKENS
    ):
        raise TrainingError(f"record is not a locked extractive span: {row['record_id']}")
    return {
        "record_id": str(row["record_id"]),
        "input_ids": ids,
        "valid_starts": [start for start, _ in spans],
        "span_length": spans[0][1],
        "prompt_utf8_bytes": len(rendered.encode("utf-8")),
        "response_utf8_bytes": len(response.encode("utf-8")),
        "teacher_tokens": int(row["teacher_tokens"]),
    }


def _batch(
    examples: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(row["input_ids"]) for row in examples)
    ids = torch.full(
        (len(examples), maximum), pad_token_id, dtype=torch.long, device=device
    )
    attention = torch.zeros(
        (len(examples), maximum), dtype=torch.bool, device=device
    )
    starts = torch.zeros(
        (len(examples), maximum), dtype=torch.bool, device=device
    )
    lengths = torch.empty(len(examples), dtype=torch.long, device=device)
    for index, row in enumerate(examples):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        ids[index, : len(values)] = values
        attention[index, : len(values)] = True
        starts[index, row["valid_starts"]] = True
        lengths[index] = int(row["span_length"]) - 1
    return ids, attention, starts, lengths


def train(
    *,
    artifact: Path,
    parent: Path,
    layercake_root: Path,
    source_result: Path,
    catalog: Path,
    output: Path,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    if output.exists():
        raise TrainingError(f"immutable R85 package exists: {output}")
    if (
        seed != 85_001
        or steps != 4_000
        or batch_size != 32
        or learning_rate != 1.0e-3
        or not torch.cuda.is_available()
    ):
        raise TrainingError("R85 preregistered training contract changed")
    frozen = (
        (artifact, ARTIFACT_SHA256),
        (parent / "model.safetensors", PARENT_SHA256),
        (parent / "metadata.json", PARENT_METADATA_SHA256),
        (source_result, SOURCE_RESULT_SHA256),
        (catalog, CATALOG_SHA256),
    )
    if any(not path.is_file() or _sha256_file(path) != digest for path, digest in frozen):
        raise TrainingError("R85 immutable input changed")
    source = json.loads(source_result.read_text(encoding="utf-8"))
    if (
        source.get("verdict") != "PASS_R85_SOURCE"
        or source.get("candidate_accessed") is not False
        or source.get("metrics", {}).get("prior_corrected_passing") != 1_371
    ):
        raise TrainingError("R85 source gate did not pass")
    rows, budget, bundle = load_english_training_rows(artifact, budget_index=-1)
    if len(rows) != 8_129 or bundle["verification"]["training_eligible"] is not True:
        raise TrainingError("R85 artifact coverage or segregation changed")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    parent_sha_before = _sha256_file(parent / "model.safetensors")
    model, tokenizer, parent_metadata = load_layercake_core(
        parent, layercake_root=layercake_root, device=device
    )
    model.eval()
    model.requires_grad_(False)
    examples = [_span_example(tokenizer, row) for row in rows]
    bridge = StatefulSpanBridge().to(device)
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=learning_rate, weight_decay=0.01
    )
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0
    unique_seen: set[str] = set()
    curves = []
    started = time.perf_counter()
    cpu_before = process.cpu_times()
    for step in range(1, steps + 1):
        selected_indices = []
        while len(selected_indices) < batch_size:
            remaining = batch_size - len(selected_indices)
            take = min(remaining, len(order) - cursor)
            selected_indices.extend(order[cursor : cursor + take])
            cursor += take
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
        selected = [examples[index] for index in selected_indices]
        unique_seen.update(row["record_id"] for row in selected)
        ids, attention, valid_starts, span_lengths = _batch(
            selected, pad_token_id=tokenizer.pad_token_id, device=device
        )
        routes = torch.full(
            (len(selected),), ROUTE, dtype=torch.long, device=device
        )
        with torch.no_grad():
            result = model(
                ids,
                attention_mask=attention,
                prompt_lengths=attention.long().sum(dim=1),
                task_routes=routes,
                use_cache=False,
            )
            hidden = result["hidden"].detach()
        traces = [
            tuple(getattr(block, "_abi_last_deep_adapter_routes", ()))
            for block in model.transformer.h
        ]
        if tuple(model.last_cake_calls) != (ROUTE,) or any(
            trace != (ROUTE,) for trace in traces
        ):
            raise TrainingError("R85 parent sparse execution changed")
        start_logits, length_logits = bridge(hidden, attention, routes)
        start_log_probabilities = F.log_softmax(start_logits.float(), dim=-1)
        valid_log_mass = torch.logsumexp(
            start_log_probabilities.masked_fill(~valid_starts, -torch.inf), dim=-1
        )
        start_loss = -valid_log_mass.mean()
        length_loss = F.cross_entropy(length_logits.float(), span_lengths)
        loss = start_loss + 0.25 * length_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            predicted_starts = start_logits.argmax(dim=-1)
            start_correct = valid_starts.gather(
                1, predicted_starts[:, None]
            ).float().mean()
            length_correct = (
                length_logits.argmax(dim=-1) == span_lengths
            ).float().mean()
            curve = {
                "step": step,
                "loss": float(loss.detach().item()),
                "start_loss": float(start_loss.detach().item()),
                "length_loss": float(length_loss.detach().item()),
                "batch_start_accuracy": float(start_correct.item()),
                "batch_length_accuracy": float(length_correct.item()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    bridge.eval()
    if len(unique_seen) != len(examples):
        raise TrainingError("R85 did not cover every imported record")
    if _sha256_file(parent / "model.safetensors") != parent_sha_before:
        raise TrainingError("R85 changed its frozen parent checkpoint")
    if _sha256_file(artifact) != ARTIFACT_SHA256:
        raise TrainingError("R85 changed its source artifact")

    output.mkdir(parents=True)
    checkpoint_path = output / "bridge.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in bridge.state_dict().items()
        },
        str(checkpoint_path),
    )
    parameter_count = bridge_parameter_count()
    metadata = {
        "format": PACKAGE_FORMAT,
        "status": "TRAINED_NOT_YET_PROSPECTIVELY_CERTIFIED",
        "bridge": {
            "architecture": BRIDGE_ARCHITECTURE,
            "width": WIDTH,
            "layers": LAYERS,
            "heads": HEADS,
            "feedforward_width": FEEDFORWARD_WIDTH,
            "maximum_tokens": MAX_TOKENS,
            "maximum_span_tokens": MAX_SPAN_TOKENS,
            "parameter_count": parameter_count,
            "active_parameter_count": parameter_count,
            "contains_prompt_templates": False,
            "contains_rules": False,
            "contains_output_lookup_table": False,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "parent_layercake": {
            "checkpoint_sha256": PARENT_SHA256,
            "metadata_sha256": PARENT_METADATA_SHA256,
            "architecture": parent_metadata["architecture"]["architecture_version"],
            "all_parameters_frozen": True,
            "changed_on_disk": False,
            "maximum_active_capability_cakes_per_sequence": 1,
            "maximum_active_deep_adapters_per_sequence": 6,
        },
        "imported_artifact": {
            "sha256": ARTIFACT_SHA256,
            "records": len(examples),
            "budget_id": budget["budget_id"],
            "teacher_tokens": sum(row["teacher_tokens"] for row in examples),
            "prompt_utf8_bytes": sum(row["prompt_utf8_bytes"] for row in examples),
            "teacher_output_utf8_bytes": sum(
                row["response_utf8_bytes"] for row in examples
            ),
            "all_records_seen": True,
            "artifact_unchanged": True,
        },
        "source_boundary": {
            "teacher_present_during_training": False,
            "teacher_present_at_inference": False,
            "source_parameters_copied": 0,
            "source_transformer_blocks_retained": 0,
            "source_logits_stored": 0,
            "source_hidden_activations_stored": 0,
            "source_generated_text_in_package": False,
            "source_result_sha256": SOURCE_RESULT_SHA256,
        },
        "training": {
            "seed": seed,
            "device": "cuda",
            "successful_optimizer_steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "unique_records_seen": len(unique_seen),
            "wall_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "cpu_seconds": (
                cpu_after.user
                + cpu_after.system
                - cpu_before.user
                - cpu_before.system
            ),
            "active_parameter_seconds": parameter_count * elapsed,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": process.memory_info().rss,
            "curves": curves,
        },
        "prospective_validation": {
            "catalog_sha256": CATALOG_SHA256,
            "candidate_outputs_observed": 0,
            "source_result_sha256": SOURCE_RESULT_SHA256,
        },
        "implementation": {
            "core_sha256": _sha256_file(Path(__file__).with_name("core_v1.py")),
            "trainer_sha256": _sha256_file(Path(__file__).resolve()),
            "screen_sha256": _sha256_file(Path(__file__).with_name("screen_host_v1.py")),
            "protocol_sha256": _sha256_file(Path(__file__).with_name("PROTOCOL.md")),
        },
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": (
            "Uncertified bounded extractive span package; not unrestricted English, "
            "domains, minimality, or ABI moonshot proof."
        ),
    }
    metadata["manifest_sha256"] = _manifest_sha(metadata)
    metadata_path = output / "metadata.json"
    metadata_path.write_bytes(
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    validate_metadata(metadata, output)
    print(
        json.dumps(
            {
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "metadata_sha256": _sha256_file(metadata_path),
                "parameter_count": parameter_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--layercake-root", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    args = parser.parse_args()
    train(
        artifact=args.artifact.resolve(),
        parent=args.parent.resolve(),
        layercake_root=args.layercake_root.resolve(),
        source_result=args.source_result.resolve(),
        catalog=args.catalog.resolve(),
        output=args.output.resolve(),
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
