"""Qualify grammar pairs directly from frozen-source likelihoods on GPU.

Free generation can confound grammatical knowledge with rewrite adherence.
This tool presents each preregistered wrong/correct pair in both A/B orders
and scores the one-token choices from the frozen source logits.  A row passes
only when the source prefers the correct sentence in both orders.  No source
weights, activations, or full logits are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .hf_extraction import HuggingFaceCausalSource, load_probe_catalog


class ContrastiveGrammarQualificationError(RuntimeError):
    """Raised when contrastive source evidence is incomplete or inconsistent."""


FORMAT = "abi-contrastive-grammar-source-qualification/1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_lower_bound(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (center - spread) / denominator


def _structure_from_probe_id(probe_id: str, split: str) -> str:
    prefix = "natural-grammar-"
    marker = f"-{split}-"
    if not probe_id.startswith(prefix) or marker not in probe_id:
        raise ContrastiveGrammarQualificationError(
            f"unsupported grammar probe identity: {probe_id}"
        )
    return probe_id.removeprefix(prefix).split(marker, 1)[0]


def _pair_from_probe(probe: Mapping[str, Any]) -> tuple[str, str]:
    evaluator = probe.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator.get("kind") != "exact":
        raise ContrastiveGrammarQualificationError(
            "contrastive grammar probes require an exact evaluator"
        )
    correct = evaluator.get("value")
    prompt = probe.get("prompt")
    marker = "\nSentence: "
    if not isinstance(correct, str) or not correct:
        raise ContrastiveGrammarQualificationError("exact correction is missing")
    if not isinstance(prompt, str) or marker not in prompt:
        raise ContrastiveGrammarQualificationError("source sentence is missing")
    wrong = prompt.split(marker, 1)[1]
    if not wrong or wrong == correct:
        raise ContrastiveGrammarQualificationError(
            "grammar pair must contain distinct wrong and correct sentences"
        )
    return wrong, correct


def _choice_prompt(first: str, second: str) -> str:
    return (
        "Which sentence has standard subject-verb agreement? Reply with only "
        f"A or B.\nA: {first}\nB: {second}\nAnswer:"
    )


def _score_requests(
    source: HuggingFaceCausalSource,
    requests: Sequence[Mapping[str, str]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Score exact completion tokens and count ephemeral full-logit elements."""

    if batch_size < 1:
        raise ContrastiveGrammarQualificationError("batch_size must be positive")
    torch = source._torch
    tokenizer = source.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ContrastiveGrammarQualificationError(
            "source tokenizer has no padding or EOS token"
        )
    prepared: list[dict[str, Any]] = []
    for request in requests:
        prompt = str(request["prompt"])
        completion = str(request["completion"])
        rendered = source.rendered_prompt(prompt)
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=True)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        if not prompt_ids or not completion_ids:
            raise ContrastiveGrammarQualificationError(
                "source produced an empty prompt or completion tokenization"
            )
        prepared.append(
            {
                "request_id": str(request["request_id"]),
                "prompt": prompt,
                "rendered_prompt": rendered,
                "completion": completion,
                "prompt_ids": [int(value) for value in prompt_ids],
                "completion_ids": [int(value) for value in completion_ids],
                "full_ids": [int(value) for value in prompt_ids + completion_ids],
            }
        )

    scores: list[dict[str, Any]] = []
    ephemeral_logit_elements = 0
    total_input_tokens = 0
    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        maximum = max(len(row["full_ids"]) for row in chunk)
        input_rows = []
        mask_rows = []
        for row in chunk:
            padding = maximum - len(row["full_ids"])
            input_rows.append(row["full_ids"] + [int(pad_id)] * padding)
            mask_rows.append([1] * len(row["full_ids"]) + [0] * padding)
            total_input_tokens += len(row["full_ids"])
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=source.device)
        attention_mask = torch.tensor(
            mask_rows, dtype=torch.long, device=source.device
        )
        with torch.inference_mode():
            logits = source.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        ephemeral_logit_elements += int(logits.numel())
        for row_index, row in enumerate(chunk):
            prompt_length = len(row["prompt_ids"])
            completion_ids = row["completion_ids"]
            positions = logits[
                row_index,
                prompt_length - 1 : prompt_length - 1 + len(completion_ids),
            ].float()
            targets = torch.tensor(
                completion_ids, dtype=torch.long, device=positions.device
            )
            token_log_probabilities = (
                torch.log_softmax(positions, dim=-1)
                .gather(-1, targets.unsqueeze(-1))
                .squeeze(-1)
                .detach()
                .cpu()
                .tolist()
            )
            scores.append(
                {
                    "request_id": row["request_id"],
                    "prompt": row["prompt"],
                    "rendered_prompt_sha256": hashlib.sha256(
                        row["rendered_prompt"].encode("utf-8")
                    ).hexdigest(),
                    "completion": row["completion"],
                    "completion_token_ids": completion_ids,
                    "completion_token_count": len(completion_ids),
                    "token_log_probabilities": [
                        float(value) for value in token_log_probabilities
                    ],
                    "sum_log_probability": float(sum(token_log_probabilities)),
                }
            )
        del logits, input_ids, attention_mask
    return scores, ephemeral_logit_elements, total_input_tokens


def _observation_from_scores(
    *,
    probe: Mapping[str, Any],
    wrong: str,
    correct: str,
    scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    probe_id = str(probe["probe_id"])
    ab_correct = scores[f"{probe_id}:ab:correct"]
    ab_incorrect = scores[f"{probe_id}:ab:incorrect"]
    ba_correct = scores[f"{probe_id}:ba:correct"]
    ba_incorrect = scores[f"{probe_id}:ba:incorrect"]
    ab_margin = float(ab_correct["sum_log_probability"]) - float(
        ab_incorrect["sum_log_probability"]
    )
    ba_margin = float(ba_correct["sum_log_probability"]) - float(
        ba_incorrect["sum_log_probability"]
    )
    passed = ab_margin > 0.0 and ba_margin > 0.0
    split = str(probe["split"])
    observation = {
        "probe_id": probe_id,
        "split": split,
        "structure": _structure_from_probe_id(probe_id, split),
        "wrong_sentence": wrong,
        "correct_sentence": correct,
        "prompt_contract_sha256": str(
            probe["evaluator"]["prompt_contract_sha256"]
        ),
        "ab": {
            "correct_label": "B",
            "correct_log_probability": float(ab_correct["sum_log_probability"]),
            "incorrect_log_probability": float(
                ab_incorrect["sum_log_probability"]
            ),
            "margin": ab_margin,
            "correct_completion_token_ids": ab_correct["completion_token_ids"],
            "incorrect_completion_token_ids": ab_incorrect[
                "completion_token_ids"
            ],
            "rendered_prompt_sha256": ab_correct["rendered_prompt_sha256"],
        },
        "ba": {
            "correct_label": "A",
            "correct_log_probability": float(ba_correct["sum_log_probability"]),
            "incorrect_log_probability": float(
                ba_incorrect["sum_log_probability"]
            ),
            "margin": ba_margin,
            "correct_completion_token_ids": ba_correct["completion_token_ids"],
            "incorrect_completion_token_ids": ba_incorrect[
                "completion_token_ids"
            ],
            "rendered_prompt_sha256": ba_correct["rendered_prompt_sha256"],
        },
        "minimum_counterbalanced_margin": min(ab_margin, ba_margin),
        "passed": passed,
    }
    observation["observation_sha256"] = hashlib.sha256(
        _canonical_bytes(observation)
    ).hexdigest()
    return observation


def _aggregate(
    observations: Sequence[Mapping[str, Any]],
    *,
    expected_records: int,
    minimum_total_passes: int,
    minimum_search_passes: int,
    minimum_validation_pass_rate: float,
    minimum_validation_wilson_lower_bound: float,
    minimum_search_passes_per_structure: int,
    minimum_validation_passes_per_structure: int,
) -> tuple[dict[str, Any], dict[str, bool], list[str]]:
    if len(observations) != expected_records:
        raise ContrastiveGrammarQualificationError(
            f"expected {expected_records} observations, found {len(observations)}"
        )
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_structure: dict[str, Counter[str]] = defaultdict(Counter)
    for observation in observations:
        split = str(observation["split"])
        structure = str(observation["structure"])
        by_split[split]["total"] += 1
        by_structure[structure][f"{split}_total"] += 1
        if bool(observation["passed"]):
            by_split[split]["passes"] += 1
            by_structure[structure][f"{split}_passes"] += 1
    total_passes = sum(bool(item["passed"]) for item in observations)
    validation_total = by_split["validation"]["total"]
    validation_passes = by_split["validation"]["passes"]
    validation_rate = (
        validation_passes / validation_total if validation_total else None
    )
    validation_wilson = (
        _wilson_lower_bound(validation_passes, validation_total)
        if validation_total
        else None
    )
    checks = {
        "exact_expected_record_count": len(observations) == expected_records,
        "minimum_total_passes_met": total_passes >= minimum_total_passes,
        "minimum_search_passes_met": (
            by_split["search"]["passes"] >= minimum_search_passes
        ),
        "minimum_validation_pass_rate_met": (
            minimum_validation_pass_rate <= 0.0
            or (
                validation_rate is not None
                and validation_rate >= minimum_validation_pass_rate
            )
        ),
        "minimum_validation_wilson_lower_bound_met": (
            minimum_validation_wilson_lower_bound <= 0.0
            or (
                validation_wilson is not None
                and validation_wilson >= minimum_validation_wilson_lower_bound
            )
        ),
        "minimum_search_passes_per_structure_met": all(
            counts["search_passes"] >= minimum_search_passes_per_structure
            for counts in by_structure.values()
        ),
        "minimum_validation_passes_per_structure_met": (
            minimum_validation_passes_per_structure <= 0
            or all(
                counts["validation_passes"]
                >= minimum_validation_passes_per_structure
                for counts in by_structure.values()
            )
        ),
        "both_counterbalanced_orders_present": all(
            "ab" in item and "ba" in item for item in observations
        ),
        "all_margins_finite": all(
            math.isfinite(float(item["ab"]["margin"]))
            and math.isfinite(float(item["ba"]["margin"]))
            for item in observations
        ),
        "unique_probe_ids": (
            len({str(item["probe_id"]) for item in observations})
            == len(observations)
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    summary = {
        "records": len(observations),
        "passes": total_passes,
        "failures": len(observations) - total_passes,
        "search": {
            "records": by_split["search"]["total"],
            "passes": by_split["search"]["passes"],
        },
        "validation": {
            "records": validation_total,
            "passes": validation_passes,
            "pass_rate": validation_rate,
            "wilson_95_lower_bound": validation_wilson,
        },
        "by_structure": {
            structure: dict(counts)
            for structure, counts in sorted(by_structure.items())
        },
    }
    return summary, checks, failures


def run_qualification(
    *,
    source: HuggingFaceCausalSource,
    catalog: Mapping[str, Any],
    catalog_path: Path,
    splits: set[str],
    output_path: Path,
    source_load_seconds: float,
    batch_size: int,
    expected_records: int,
    minimum_total_passes: int,
    minimum_search_passes: int,
    minimum_validation_pass_rate: float,
    minimum_validation_wilson_lower_bound: float,
    minimum_search_passes_per_structure: int,
    minimum_validation_passes_per_structure: int,
) -> dict[str, Any]:
    probes = [
        probe
        for probe in catalog["probes"]
        if probe.get("split") in splits and probe.get("capability") == "grammar"
    ]
    if len(probes) != expected_records:
        raise ContrastiveGrammarQualificationError(
            f"catalog selection expected {expected_records} probes, found {len(probes)}"
        )
    requests: list[dict[str, str]] = []
    pairs: dict[str, tuple[str, str]] = {}
    raw_prompt_bytes = 0
    unique_prompt_bytes: set[bytes] = set()
    for probe in probes:
        probe_id = str(probe["probe_id"])
        wrong, correct = _pair_from_probe(probe)
        pairs[probe_id] = (wrong, correct)
        prompt_ab = _choice_prompt(wrong, correct)
        prompt_ba = _choice_prompt(correct, wrong)
        for prompt in (prompt_ab, prompt_ba):
            encoded = prompt.encode("utf-8")
            raw_prompt_bytes += len(encoded)
            unique_prompt_bytes.add(encoded)
        requests.extend(
            [
                {
                    "request_id": f"{probe_id}:ab:correct",
                    "prompt": prompt_ab,
                    "completion": "B",
                },
                {
                    "request_id": f"{probe_id}:ab:incorrect",
                    "prompt": prompt_ab,
                    "completion": "A",
                },
                {
                    "request_id": f"{probe_id}:ba:correct",
                    "prompt": prompt_ba,
                    "completion": "A",
                },
                {
                    "request_id": f"{probe_id}:ba:incorrect",
                    "prompt": prompt_ba,
                    "completion": "B",
                },
            ]
        )

    if source.device == "cuda":
        source._torch.cuda.reset_peak_memory_stats()
        source._torch.cuda.synchronize()
    inference_started = time.perf_counter()
    score_rows, ephemeral_logit_elements, total_input_tokens = _score_requests(
        source, requests, batch_size=batch_size
    )
    if source.device == "cuda":
        source._torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    score_by_id = {row["request_id"]: row for row in score_rows}
    if len(score_by_id) != len(requests):
        raise ContrastiveGrammarQualificationError(
            "contrastive request identities are not unique"
        )
    observations = [
        _observation_from_scores(
            probe=probe,
            wrong=pairs[str(probe["probe_id"])][0],
            correct=pairs[str(probe["probe_id"])][1],
            scores=score_by_id,
        )
        for probe in probes
    ]
    summary, checks, failures = _aggregate(
        observations,
        expected_records=expected_records,
        minimum_total_passes=minimum_total_passes,
        minimum_search_passes=minimum_search_passes,
        minimum_validation_pass_rate=minimum_validation_pass_rate,
        minimum_validation_wilson_lower_bound=minimum_validation_wilson_lower_bound,
        minimum_search_passes_per_structure=minimum_search_passes_per_structure,
        minimum_validation_passes_per_structure=minimum_validation_passes_per_structure,
    )
    manifest = source.source_manifest
    weight_bytes = sum(int(item["bytes"]) for item in manifest["weight_files"])
    gpu_peak = (
        int(source._torch.cuda.max_memory_allocated())
        if source.device == "cuda"
        else 0
    )
    evidence: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": (
            "This evidence qualifies frozen-source contrastive grammar "
            "preferences only. It is ABI extraction evidence, not proof of "
            "LayerCake transfer, quality, speed, TTFO, memory, or sparsity."
        ),
        "source": {
            "model": manifest["model_id"],
            "revision": manifest["revision"],
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "parameter_count_read": manifest["parameter_count"],
            "source_weight_bytes_read": weight_bytes,
            "runtime": source.source_inference_runtime,
            "hardware": (
                source._torch.cuda.get_device_name(source._torch.cuda.current_device())
                if source.device == "cuda"
                else "cpu"
            ),
        },
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "catalog_id": catalog["catalog_id"],
            "splits": sorted(splits),
            "records": len(probes),
            "final_test_accessed": "final_test" in splits,
        },
        "method": {
            "name": "counterbalanced_forced_choice_log_probability",
            "correct_sentence_scored_in_both_positions": True,
            "pass_rule": "correct sentence has strictly greater completion log probability in both A/B orderings",
            "full_source_logits_stored": False,
            "source_activations_stored": False,
            "source_weights_copied": False,
        },
        "thresholds": {
            "expected_records": expected_records,
            "minimum_total_passes": minimum_total_passes,
            "minimum_search_passes": minimum_search_passes,
            "minimum_validation_pass_rate": minimum_validation_pass_rate,
            "minimum_validation_wilson_95_lower_bound": minimum_validation_wilson_lower_bound,
            "minimum_search_passes_per_structure": minimum_search_passes_per_structure,
            "minimum_validation_passes_per_structure": minimum_validation_passes_per_structure,
        },
        "checks": checks,
        "failures": failures,
        "summary": summary,
        "imported_information_accounting": {
            "raw_source_prompt_count": len(probes) * 2,
            "raw_source_prompt_bytes": raw_prompt_bytes,
            "unique_source_prompt_count": len(unique_prompt_bytes),
            "unique_source_prompt_utf8_bytes": sum(
                len(value) for value in unique_prompt_bytes
            ),
            "source_input_and_completion_tokens_evaluated": total_input_tokens,
            "completion_tokens_scored": sum(
                int(row["completion_token_count"]) for row in score_rows
            ),
            "selected_log_probabilities_stored": len(score_rows),
            "selected_log_probability_storage_bytes_if_float64": len(score_rows)
            * 8,
            "ephemeral_full_logit_elements_materialized": ephemeral_logit_elements,
            "full_logit_elements_stored": 0,
            "hidden_activations_stored": 0,
            "frozen_source_parameters_copied": 0,
            "teacher_generated_output_bytes": 0,
            "teacher_generated_tokens": 0,
            "one_time_source_load_seconds": float(source_load_seconds),
            "source_model_inference_seconds": inference_seconds,
            "peak_cuda_memory_allocated_bytes": gpu_peak,
            "external_hardware_used": source.device == "cuda",
        },
        "observations": observations,
        "layercake_invoked": False,
        "layercake_training_authorized": False,
        "abi_transfer_proven": False,
        "moonshot_complete": False,
    }
    unsigned = dict(evidence)
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_bytes(unsigned)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ContrastiveGrammarQualificationError(
            f"qualification evidence is immutable: {output_path}"
        )
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="search,validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--minimum-total-passes", type=int, required=True)
    parser.add_argument("--minimum-search-passes", type=int, default=0)
    parser.add_argument("--minimum-validation-pass-rate", type=float, default=0.0)
    parser.add_argument(
        "--minimum-validation-wilson-lower-bound", type=float, default=0.0
    )
    parser.add_argument(
        "--minimum-search-passes-per-structure", type=int, default=0
    )
    parser.add_argument(
        "--minimum-validation-passes-per-structure", type=int, default=0
    )
    args = parser.parse_args(argv)
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    if not splits or not splits <= {"search", "validation"}:
        parser.error("contrastive qualification permits search/validation only")
    started = time.perf_counter()
    source = HuggingFaceCausalSource(
        args.model,
        revision=args.revision,
        license_id=args.license,
        device=args.device,
        local_files_only=not args.allow_network,
        load_in_8bit=args.load_in_8bit,
    )
    source_load_seconds = time.perf_counter() - started
    catalog_path = Path(args.catalog)
    catalog = load_probe_catalog(catalog_path)
    evidence = run_qualification(
        source=source,
        catalog=catalog,
        catalog_path=catalog_path,
        splits=splits,
        output_path=Path(args.output),
        source_load_seconds=source_load_seconds,
        batch_size=args.batch_size,
        expected_records=args.expected_records,
        minimum_total_passes=args.minimum_total_passes,
        minimum_search_passes=args.minimum_search_passes,
        minimum_validation_pass_rate=args.minimum_validation_pass_rate,
        minimum_validation_wilson_lower_bound=(
            args.minimum_validation_wilson_lower_bound
        ),
        minimum_search_passes_per_structure=(
            args.minimum_search_passes_per_structure
        ),
        minimum_validation_passes_per_structure=(
            args.minimum_validation_passes_per_structure
        ),
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "status": evidence["status"],
                "passes": evidence["summary"]["passes"],
                "records": evidence["summary"]["records"],
                "source_load_seconds": source_load_seconds,
                "source_inference_seconds": evidence[
                    "imported_information_accounting"
                ]["source_model_inference_seconds"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if evidence["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
