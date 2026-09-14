"""Train the preregistered R88 joint-span package from frozen ABI records."""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from safetensors.torch import save_file

from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core
from abi.layercake_full_core_acquisition import (
    _manifest_sha,
    load_english_training_rows,
)
from abi.layercake_host import _sha256_file
from experiments.joint_span_r88.core_v1 import (
    BRIDGE_ARCHITECTURE,
    FEEDFORWARD_WIDTH,
    HEADS,
    LAYERS,
    MAX_SPAN_TOKENS,
    MAX_TOKENS,
    PACKAGE_FORMAT,
    WIDTH,
    JointSpanBridge,
    bridge_parameter_count,
    joint_span_scores,
    validate_metadata,
)
from experiments.stateful_span_r85.train_candidate_v1 import (
    _batch,
    _span_example,
)


PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
ARTIFACT_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
SOURCE_RESULT_SHA256 = "facac7a81732805ad7c2dece8160ce22e74532805d5b5da4fe2f8a8aec1baf4e"
CATALOG_SHA256 = "5dba762eec849f2e6531f5f1283417d38e9470a9b37ca69cf3576bb25286a177"
ROUTE = CAPABILITY_CAKE_ORDER.index("domain_independent_reasoning")
CODE = re.compile(r"\b[A-Z]{3}\d{6}\b")
RESERVED_STEMS = frozenset(
    {
        "REF",
        "BAZ",
        "COF",
        "DUM",
        "FEK",
        "GIQ",
        "HUR",
        "JAV",
        "KOB",
        "LIX",
        "MEZ",
        "NOF",
        "PUK",
        "RIL",
        "SOG",
        "TUM",
        "VAB",
        "WEQ",
        "XIR",
        "YOK",
        "ZUP",
        "BEM",
    }
)


class TrainingError(RuntimeError):
    pass


def _stem(value: int) -> str:
    if not 0 <= value < 26**3:
        raise TrainingError("R88 generated stem index is invalid")
    letters = []
    for divisor in (26**2, 26, 1):
        letters.append(chr(65 + (value // divisor) % 26))
    return "".join(letters)


def _rename_row(
    tokenizer: Any, row: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    prompt = str(row["prompt"])
    response = str(row["response"])
    codes = list(dict.fromkeys(CODE.findall(prompt)))
    if len(codes) != 4 or response not in codes:
        raise TrainingError(f"R88 source row is not four-code extractive: {row['record_id']}")
    stems: list[str] = []
    while len(stems) < 4:
        candidate = _stem(rng.randrange(26**3))
        if candidate not in RESERVED_STEMS and candidate not in stems:
            stems.append(candidate)
    numeric = rng.randrange(200_000, 900_000)
    mapping = {
        code: f"{stem}{numeric:06d}"
        for code, stem in zip(codes, stems, strict=True)
    }
    renamed_prompt = CODE.sub(lambda match: mapping[match.group(0)], prompt)
    renamed_response = mapping[response]
    transformed = {
        **row,
        "record_id": f"{row['record_id']}:{numeric}:{'-'.join(stems)}",
        "prompt": renamed_prompt,
        "response": renamed_response,
    }
    example = _span_example(tokenizer, transformed)
    example["base_record_id"] = str(row["record_id"])
    example["synthetic_prompt_utf8_bytes"] = len(
        renamed_prompt.encode("utf-8")
    )
    example["synthetic_response_utf8_bytes"] = len(
        renamed_response.encode("utf-8")
    )
    return example


def _gold_spans(
    examples: list[dict[str, Any]], tokens: int, device: torch.device
) -> torch.Tensor:
    gold = torch.zeros(
        len(examples), tokens, tokens, dtype=torch.bool, device=device
    )
    for index, row in enumerate(examples):
        length = int(row["span_length"])
        for start in row["valid_starts"]:
            gold[index, int(start), int(start) + length - 1] = True
    if not gold.flatten(1).any(dim=1).all():
        raise TrainingError("R88 batch has no gold joint span")
    return gold


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
        raise TrainingError(f"immutable R88 package exists: {output}")
    if (
        seed != 88_001
        or steps != 3_000
        or batch_size != 32
        or learning_rate != 1.0e-3
        or not torch.cuda.is_available()
    ):
        raise TrainingError("R88 preregistered training contract changed")
    frozen = (
        (artifact, ARTIFACT_SHA256),
        (parent / "model.safetensors", PARENT_SHA256),
        (parent / "metadata.json", PARENT_METADATA_SHA256),
        (source_result, SOURCE_RESULT_SHA256),
        (catalog, CATALOG_SHA256),
    )
    if any(not path.is_file() or _sha256_file(path) != digest for path, digest in frozen):
        raise TrainingError("R88 immutable input changed")
    source = json.loads(source_result.read_text(encoding="utf-8"))
    if (
        source.get("verdict") != "PASS_R88_SOURCE"
        or source.get("candidate_accessed") is not False
        or source.get("metrics", {}).get("prior_corrected_passing", 0) < 1_330
        or not all(source.get("gates", {}).values())
    ):
        raise TrainingError("R88 corrected source gate did not pass")
    rows, budget, bundle = load_english_training_rows(artifact, budget_index=-1)
    if len(rows) != 8_129 or bundle["verification"]["training_eligible"] is not True:
        raise TrainingError("R88 artifact coverage or segregation changed")

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
    # This fail-closed pass proves every immutable teacher record has the base
    # two-occurrence span contract before any stochastic normalization.
    base_examples = [_span_example(tokenizer, row) for row in rows]
    bridge = JointSpanBridge().to(device)
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=learning_rate, weight_decay=0.01
    )
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cursor = 0
    unique_seen: set[str] = set()
    curves = []
    synthetic_prompt_bytes = 0
    synthetic_response_bytes = 0
    synthetic_examples = 0
    started = time.perf_counter()
    cpu_before = process.cpu_times()
    for step in range(1, steps + 1):
        selected_indices = []
        while len(selected_indices) < batch_size:
            take = min(batch_size - len(selected_indices), len(order) - cursor)
            selected_indices.extend(order[cursor : cursor + take])
            cursor += take
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
        selected = [_rename_row(tokenizer, rows[index], rng) for index in selected_indices]
        unique_seen.update(row["base_record_id"] for row in selected)
        synthetic_examples += len(selected)
        synthetic_prompt_bytes += sum(
            row["synthetic_prompt_utf8_bytes"] for row in selected
        )
        synthetic_response_bytes += sum(
            row["synthetic_response_utf8_bytes"] for row in selected
        )
        ids, attention, _, _ = _batch(
            selected, pad_token_id=tokenizer.pad_token_id, device=device
        )
        routes = torch.full((len(selected),), ROUTE, dtype=torch.long, device=device)
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
            raise TrainingError("R88 parent sparse execution changed")
        start_logits, end_logits = bridge(hidden, attention, routes)
        scores = joint_span_scores(start_logits, end_logits, attention).float()
        gold = _gold_spans(selected, scores.shape[1], device)
        normalizer = torch.logsumexp(scores.flatten(1), dim=-1)
        gold_mass = torch.logsumexp(
            scores.masked_fill(~gold, -torch.inf).flatten(1), dim=-1
        )
        loss = (normalizer - gold_mass).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            predicted = scores.flatten(1).argmax(dim=-1)
            correct = gold.flatten(1).gather(1, predicted[:, None]).float().mean()
            curve = {
                "step": step,
                "loss": float(loss.detach().item()),
                "batch_joint_span_accuracy": float(correct.item()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    bridge.eval()
    if len(unique_seen) != len(rows):
        raise TrainingError("R88 did not cover every imported record")
    if _sha256_file(parent / "model.safetensors") != parent_sha_before:
        raise TrainingError("R88 changed its frozen parent checkpoint")
    if _sha256_file(artifact) != ARTIFACT_SHA256:
        raise TrainingError("R88 changed its source artifact")

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
            "records": len(base_examples),
            "budget_id": budget["budget_id"],
            "teacher_tokens": sum(row["teacher_tokens"] for row in base_examples),
            "prompt_utf8_bytes": sum(row["prompt_utf8_bytes"] for row in base_examples),
            "teacher_output_utf8_bytes": sum(
                row["response_utf8_bytes"] for row in base_examples
            ),
            "all_records_seen": True,
            "artifact_unchanged": True,
        },
        "normalization": {
            "method": "deterministic_bijective_nonce_code_renaming",
            "adds_teacher_claims": False,
            "synthetic_teacher_tokens": 0,
            "synthetic_examples_consumed": synthetic_examples,
            "synthetic_prompt_utf8_bytes_consumed": synthetic_prompt_bytes,
            "synthetic_response_utf8_bytes_consumed": synthetic_response_bytes,
            "validation_stems_forbidden": sorted(RESERVED_STEMS),
            "validation_numeric_range_forbidden": [995_000, 996_399],
            "normalized_text_stored_in_package": False,
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
            "cpu_seconds": cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
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
        "claim_boundary": "Uncertified bounded extractive joint-span package; not the full ABI moonshot.",
    }
    metadata["manifest_sha256"] = _manifest_sha(metadata)
    metadata_path = output / "metadata.json"
    metadata_path.write_bytes(
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    validate_metadata(metadata, output)
    print(json.dumps({
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "metadata_sha256": _sha256_file(metadata_path),
        "parameter_count": parameter_count,
    }, indent=2, sort_keys=True))
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
        artifact=args.artifact.resolve(), parent=args.parent.resolve(),
        layercake_root=args.layercake_root.resolve(), source_result=args.source_result.resolve(),
        catalog=args.catalog.resolve(), output=args.output.resolve(), seed=args.seed,
        steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
