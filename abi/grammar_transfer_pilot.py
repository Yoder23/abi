"""Evaluate one bounded teacher-artifact-to-LayerCake grammar transfer pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from .layercake_core_loader import load_layercake_core
from .layercake_host import _canonical_json_bytes, _sha256_file


EVIDENCE_FORMAT = "abi-teacher-to-layercake-grammar-pilot-evidence/1"


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"grammar pilot evidence is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validation_rows(catalog_path: Path) -> list[dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = []
    for probe in catalog.get("probes", []):
        if probe.get("split") != "validation":
            continue
        prompt = str(probe["prompt"])
        marker = "\nSentence: "
        if prompt.count(marker) != 1:
            raise RuntimeError("grammar prompt does not expose one incorrect sentence")
        evaluator = probe.get("evaluator")
        if not isinstance(evaluator, Mapping) or evaluator.get("kind") != "exact":
            raise RuntimeError("grammar validation evaluator changed")
        rows.append(
            {
                "probe_id": str(probe["probe_id"]),
                "prompt": prompt,
                "correct": str(evaluator["value"]),
                "incorrect": prompt.split(marker, 1)[1],
                "structure": str(probe["probe_id"]).split("-validation-")[0],
            }
        )
    rows.sort(key=lambda row: row["probe_id"])
    if len(rows) != 64 or len({row["prompt"] for row in rows}) != 64:
        raise RuntimeError("locked 64-row grammar validation split changed")
    return rows


def _mean_completion_log_probability(
    model,
    tokenizer,
    *,
    prompt: str,
    completion: str,
    device: torch.device,
    route: int,
) -> tuple[float, int]:
    prompt_ids = tokenizer.encode(prompt + "\n")
    completion_ids = tokenizer.encode(completion) + [tokenizer.eos_token_id]
    sequence = prompt_ids + completion_ids
    if len(sequence) > int(model.config.max_tokens):
        raise RuntimeError("grammar evaluation sequence exceeds model context")
    ids = torch.tensor([sequence], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    route_tensor = torch.tensor([route], dtype=torch.long, device=device)
    result = model(
        ids,
        attention_mask=attention,
        prompt_lengths=torch.tensor(
            [len(prompt_ids)], dtype=torch.long, device=device
        ),
        task_routes=route_tensor,
        use_cache=False,
    )
    logits = result["logits"][:, :-1].float()
    targets = ids[:, 1:]
    start = len(prompt_ids) - 1
    selected_logits = logits[:, start : start + len(completion_ids)]
    selected_targets = targets[:, start : start + len(completion_ids)]
    token_log_probabilities = -F.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        selected_targets.reshape(-1),
        reduction="none",
    )
    return float(token_log_probabilities.mean().item()), len(completion_ids)


@torch.inference_mode()
def _generate_forced_route(
    model,
    tokenizer,
    *,
    prompt: str,
    device: torch.device,
    route: int,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    prompt_ids = tokenizer.encode(prompt + "\n")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    route_tensor = torch.tensor([route], dtype=torch.long, device=device)
    result = model(
        ids,
        prompt_lengths=torch.tensor(
            [len(prompt_ids)], dtype=torch.long, device=device
        ),
        task_routes=route_tensor,
        use_cache=True,
    )
    cache = result["past_key_values"]
    logits = result["logits"][:, -1]
    generated: list[int] = []
    for _ in range(max_new_tokens):
        selected = logits.argmax(dim=-1)
        token_id = int(selected.item())
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        result = model(
            selected[:, None],
            task_routes=route_tensor,
            past_key_values=cache,
            use_cache=True,
        )
        cache = result["past_key_values"]
        logits = result["logits"][:, -1]
    return (
        tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        generated,
    )


def _repetition_collapse(token_ids: Sequence[int]) -> bool:
    if len(token_ids) < 12:
        return False
    windows = [tuple(token_ids[index : index + 4]) for index in range(len(token_ids) - 3)]
    return len(set(windows)) / len(windows) < 0.5


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if not differences or replicates < 100:
        raise ValueError("paired bootstrap requires rows and at least 100 replicates")
    rng = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(replicates)
    )
    lower = estimates[int(0.025 * replicates)]
    upper = estimates[min(replicates - 1, int(0.975 * replicates))]
    return {
        "mean": statistics.fmean(differences),
        "lower_95": lower,
        "upper_95": upper,
    }


def _evaluate_model(
    *,
    name: str,
    model_path: Path,
    layercake_root: Path,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    route: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model, tokenizer, metadata = load_layercake_core(
        model_path,
        layercake_root=layercake_root,
        device=device,
    )
    observations = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            correct_log_probability, correct_tokens = (
                _mean_completion_log_probability(
                    model,
                    tokenizer,
                    prompt=str(row["prompt"]),
                    completion=str(row["correct"]),
                    device=device,
                    route=route,
                )
            )
            incorrect_log_probability, incorrect_tokens = (
                _mean_completion_log_probability(
                    model,
                    tokenizer,
                    prompt=str(row["prompt"]),
                    completion=str(row["incorrect"]),
                    device=device,
                    route=route,
                )
            )
            output, generated = _generate_forced_route(
                model,
                tokenizer,
                prompt=str(row["prompt"]),
                device=device,
                route=route,
                max_new_tokens=48,
            )
            observations.append(
                {
                    "probe_id": row["probe_id"],
                    "structure": row["structure"],
                    "correct_mean_log_probability": correct_log_probability,
                    "incorrect_mean_log_probability": incorrect_log_probability,
                    "margin": correct_log_probability - incorrect_log_probability,
                    "positive_margin": correct_log_probability > incorrect_log_probability,
                    "correct_completion_tokens": correct_tokens,
                    "incorrect_completion_tokens": incorrect_tokens,
                    "generated_output": output,
                    "generated_output_sha256": hashlib.sha256(
                        output.encode("utf-8")
                    ).hexdigest(),
                    "autonomous_exact_after_outer_whitespace_normalization": (
                        output.strip() == str(row["correct"]).strip()
                    ),
                    "empty_output": not bool(output.strip()),
                    "repetition_collapse": _repetition_collapse(generated),
                    "generated_tokens": len(generated),
                }
            )
            if (index + 1) % 16 == 0:
                print(json.dumps({"model": name, "evaluated": index + 1}), flush=True)
    aggregate = {
        "model": name,
        "path": str(model_path),
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "rows": len(observations),
        "mean_correct_log_probability": statistics.fmean(
            row["correct_mean_log_probability"] for row in observations
        ),
        "mean_incorrect_log_probability": statistics.fmean(
            row["incorrect_mean_log_probability"] for row in observations
        ),
        "mean_margin": statistics.fmean(row["margin"] for row in observations),
        "positive_margin_accuracy": statistics.fmean(
            float(row["positive_margin"]) for row in observations
        ),
        "autonomous_exact_accuracy": statistics.fmean(
            float(row["autonomous_exact_after_outer_whitespace_normalization"])
            for row in observations
        ),
        "empty_outputs": sum(bool(row["empty_output"]) for row in observations),
        "repetition_collapses": sum(
            bool(row["repetition_collapse"]) for row in observations
        ),
        "wall_seconds": time.perf_counter() - started,
        "observations": observations,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return aggregate, metadata


def _state_isolation(
    parent_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    parent = load_file(str(parent_path / "model.safetensors"), device="cpu")
    candidate = load_file(str(candidate_path / "model.safetensors"), device="cpu")
    if set(parent) != set(candidate):
        raise RuntimeError("candidate tensor topology differs from parent")
    changed = sorted(
        name for name in parent if not torch.equal(parent[name], candidate[name])
    )
    allowed_prefix = "task_cakes.0."
    return {
        "changed_tensor_count": len(changed),
        "changed_tensors": changed,
        "route0_changed": bool(changed),
        "all_changes_confined_to_route0": bool(changed)
        and all(name.startswith(allowed_prefix) for name in changed),
        "unchanged_tensor_count": len(parent) - len(changed),
        "tensor_topology_exact": True,
    }


def evaluate_pilot(
    *,
    protocol_path: Path,
    catalog_path: Path,
    source_artifact_path: Path,
    layercake_root: Path,
    parent_path: Path,
    real_path: Path,
    shuffled_path: Path,
    output_path: Path,
    device_name: str,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_BEFORE_COMPOSITION_TRAINING_OR_LAYERCAKE_EVALUATION":
        raise RuntimeError("grammar pilot protocol is not preregistered")
    if _sha256_file(catalog_path) != protocol["catalog"]["sha256"]:
        raise RuntimeError("grammar catalog identity changed")
    if _sha256_file(source_artifact_path) != protocol["source_artifact"]["sha256"]:
        raise RuntimeError("source artifact identity changed")
    rows = _validation_rows(catalog_path)
    device = torch.device(device_name)
    evaluated: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for name, path in (
        ("parent", parent_path),
        ("real", real_path),
        ("shuffled", shuffled_path),
    ):
        evaluated[name], metadata[name] = _evaluate_model(
            name=name,
            model_path=path,
            layercake_root=layercake_root,
            rows=rows,
            device=device,
            route=0,
        )
    observations = {
        name: {row["probe_id"]: row for row in aggregate["observations"]}
        for name, aggregate in evaluated.items()
    }
    ids = [row["probe_id"] for row in rows]
    comparisons = {}
    for metric in ("correct_mean_log_probability", "margin"):
        for baseline in ("parent", "shuffled"):
            key = f"real_minus_{baseline}_{metric}"
            comparisons[key] = paired_bootstrap_interval(
                [
                    float(observations["real"][probe_id][metric])
                    - float(observations[baseline][probe_id][metric])
                    for probe_id in ids
                ],
                replicates=10000,
                seed=87004,
            )
    real_isolation = _state_isolation(parent_path, real_path)
    shuffled_isolation = _state_isolation(parent_path, shuffled_path)
    real_foreign = metadata["real"].get("foreign_source_boundary", {})
    real_target_control = metadata["real"].get("training", {}).get("target_control")
    shuffled_target_control = metadata["shuffled"].get("training", {}).get("target_control")
    checks = {
        "locked_validation_depth_64": len(rows) == 64,
        "real_correct_logprob_beats_parent_ci": comparisons[
            "real_minus_parent_correct_mean_log_probability"
        ]["lower_95"] > 0.0,
        "real_correct_logprob_beats_shuffled_ci": comparisons[
            "real_minus_shuffled_correct_mean_log_probability"
        ]["lower_95"] > 0.0,
        "real_margin_beats_parent_ci": comparisons[
            "real_minus_parent_margin"
        ]["lower_95"] > 0.0,
        "real_margin_beats_shuffled_ci": comparisons[
            "real_minus_shuffled_margin"
        ]["lower_95"] > 0.0,
        "real_positive_margin_accuracy_at_least_075": evaluated["real"][
            "positive_margin_accuracy"
        ] >= 0.75,
        "real_changes_only_route0": real_isolation["all_changes_confined_to_route0"],
        "shuffled_changes_only_route0": shuffled_isolation[
            "all_changes_confined_to_route0"
        ],
        "real_target_control_identity": isinstance(real_target_control, Mapping)
        and real_target_control.get("mode") == "identity",
        "shuffled_target_control_is_complete_derangement": isinstance(
            shuffled_target_control, Mapping
        )
        and shuffled_target_control.get("mode") == "deterministic_derangement"
        and shuffled_target_control.get("all_targets_changed") is True,
        "teacher_absent_from_real_candidate": real_foreign.get(
            "teacher_present_at_inference"
        )
        is False
        and real_foreign.get("source_transformer_blocks_retained") == 0
        and real_foreign.get("source_parameters_copied") == 0
        and real_foreign.get("teacher_tokenizer_required_at_inference") is False,
        "source_artifact_unchanged": _sha256_file(source_artifact_path)
        == protocol["source_artifact"]["sha256"],
        "final_test_absent": True,
    }
    status = "PASS_BOUNDED_GRAMMAR_TRANSFER" if all(checks.values()) else "FAIL_BOUNDED_GRAMMAR_TRANSFER"
    evidence: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "status": status,
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256_file(protocol_path),
        "catalog_path": str(catalog_path),
        "catalog_sha256": _sha256_file(catalog_path),
        "source_artifact_path": str(source_artifact_path),
        "source_artifact_sha256": _sha256_file(source_artifact_path),
        "validation_split": "validation",
        "validation_rows": len(rows),
        "final_test_accessed": False,
        "route": 0,
        "models": evaluated,
        "paired_bootstrap_comparisons": comparisons,
        "state_isolation": {
            "real": real_isolation,
            "shuffled": shuffled_isolation,
        },
        "checks": checks,
        "secondary_generation_gate": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    _write_immutable(output_path, evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--shuffled", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    evidence = evaluate_pilot(
        protocol_path=Path(args.protocol).resolve(),
        catalog_path=Path(args.catalog).resolve(),
        source_artifact_path=Path(args.source_artifact).resolve(),
        layercake_root=Path(args.layercake_root).resolve(),
        parent_path=Path(args.parent).resolve(),
        real_path=Path(args.real).resolve(),
        shuffled_path=Path(args.shuffled).resolve(),
        output_path=Path(args.output).resolve(),
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "evidence_sha256": evidence["evidence_sha256"],
                "checks": evidence["checks"],
                "aggregates": {
                    name: {
                        key: value
                        for key, value in model.items()
                        if key
                        in {
                            "mean_correct_log_probability",
                            "mean_margin",
                            "positive_margin_accuracy",
                            "autonomous_exact_accuracy",
                            "empty_outputs",
                            "repetition_collapses",
                        }
                    }
                    for name, model in evidence["models"].items()
                },
                "paired_bootstrap_comparisons": evidence[
                    "paired_bootstrap_comparisons"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if evidence["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
