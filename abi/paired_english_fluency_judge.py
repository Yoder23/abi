"""Run the preregistered blinded teacher-relative English quality audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
from typing import Any, Mapping, Sequence

from .capability_pipeline import read_extraction_bundle
from .hf_extraction import HuggingFaceCausalSource, load_probe_catalog


PROTOCOL_ID = "abi-english-paired-fluency-judge-v1"
VALIDATION_PROTOCOL_ID = "abi-english-scale-validation-paired-judge-v1"
EVIDENCE_FORMAT = "abi-paired-english-fluency-judge-evidence/1"
DIMENSIONS = (
    "grammatical_fluency",
    "local_and_global_coherence",
    "prompt_and_context_grounding",
    "instruction_and_format_adherence",
)
FLAGS = (
    "unsupported_factual_detail",
    "repetition_or_collapse",
    "unusable_or_empty",
)
PROMPTS_PER_CAPABILITY = 8
BOOTSTRAP_SAMPLES = 10_000


class PairedJudgeError(RuntimeError):
    """Raised when paired evidence violates the frozen audit contract."""


def _canonical_sha(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_probe_ids(
    catalog: Mapping[str, Any],
    *,
    split: str = "final_test",
    protocol_id: str = PROTOCOL_ID,
    prompts_per_capability: int = PROMPTS_PER_CAPABILITY,
) -> list[str]:
    if split not in {"search", "validation", "final_test"}:
        raise PairedJudgeError("unsupported paired-judge split")
    if prompts_per_capability <= 0:
        raise PairedJudgeError("prompts per capability must be positive")
    by_capability: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for probe in catalog["probes"]:
        if probe["split"] != split:
            continue
        probe_id = str(probe["probe_id"])
        score = hashlib.sha256(
            f"{protocol_id}:{probe_id}".encode("utf-8")
        ).hexdigest()
        by_capability[str(probe["capability"])].append((score, probe_id))
    selected = []
    for capability in sorted(by_capability):
        rows = sorted(by_capability[capability])
        if len(rows) < prompts_per_capability:
            raise PairedJudgeError(
                f"insufficient {split} probes for {capability}"
            )
        selected.extend(
            probe_id for _, probe_id in rows[:prompts_per_capability]
        )
    return selected


def _candidate_outputs(
    path: Path,
    *,
    split: str = "final_test",
) -> tuple[dict[str, str], dict[str, Any]]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("split") != split:
        raise PairedJudgeError(
            f"candidate evidence is not {split} evidence"
        )
    outputs = {
        str(row["probe_id"]): str(row["layercake_output"])
        for row in evidence["observations"]
    }
    return outputs, {
        "path_at_judging": str(path),
        "sha256": _sha256_file(path),
        "evidence_sha256": evidence.get("evidence_sha256"),
        "candidate": evidence.get("candidate"),
    }


def _teacher_outputs(
    path: Path,
    *,
    split: str = "final_test",
) -> tuple[dict[str, str], dict[str, Any]]:
    bundle = read_extraction_bundle(path)
    records_by_id = {
        str(record["record_id"]): record
        for record in bundle["records"]
        if record["split"] == split
    }
    by_probe = {}
    for result in bundle["probe_results"]:
        record = records_by_id.get(str(result["record_id"]))
        if record is not None:
            by_probe[str(result["probe_id"])] = str(record["output"])
    return by_probe, {
        "path_at_judging": str(path),
        "archive_sha256": bundle["verification"]["archive_sha256"],
        "manifest_sha256": bundle["verification"]["manifest_sha256"],
    }


def _candidate_is_a(
    probe_id: str,
    protocol_id: str = PROTOCOL_ID,
) -> bool:
    digest = hashlib.sha256(
        f"{protocol_id}:orientation:{probe_id}".encode("utf-8")
    ).digest()
    return bool(digest[0] & 1)


def _judge_prompt(
    *,
    user_prompt: str,
    output_a: str,
    output_b: str,
) -> str:
    return (
        "You are a strict English response evaluator. Score each response "
        "independently from 0 (unusable) to 4 (excellent) on grammatical "
        "fluency, local and global coherence, grounding in the user's prompt "
        "or supplied context, and instruction/format adherence. Mark each "
        "binary problem true or false: unsupported factual detail, repetition "
        "or collapse, and unusable or empty. Do not reward verbosity. Do not "
        "guess which system made either response.\n"
        "Return exactly one JSON object with this schema and no Markdown:\n"
        '{"A":{"grammatical_fluency":0,"local_and_global_coherence":0,'
        '"prompt_and_context_grounding":0,'
        '"instruction_and_format_adherence":0,'
        '"unsupported_factual_detail":false,'
        '"repetition_or_collapse":false,"unusable_or_empty":false},'
        '"B":{"grammatical_fluency":0,"local_and_global_coherence":0,'
        '"prompt_and_context_grounding":0,'
        '"instruction_and_format_adherence":0,'
        '"unsupported_factual_detail":false,'
        '"repetition_or_collapse":false,"unusable_or_empty":false}}\n'
        f"<user_prompt>\n{user_prompt}\n</user_prompt>\n"
        f"<response_A>\n{output_a}\n</response_A>\n"
        f"<response_B>\n{output_b}\n</response_B>"
    )


def _parse_scores(output: str) -> dict[str, dict[str, Any]] | None:
    match = re.search(r"\{.*\}", output, flags=re.DOTALL)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if set(parsed) != {"A", "B"}:
        return None
    normalized: dict[str, dict[str, Any]] = {}
    for side in ("A", "B"):
        row = parsed.get(side)
        if not isinstance(row, dict):
            return None
        if set(row) != set(DIMENSIONS) | set(FLAGS):
            return None
        values: dict[str, Any] = {}
        for dimension in DIMENSIONS:
            value = row[dimension]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 4
            ):
                return None
            values[dimension] = value
        for flag in FLAGS:
            if not isinstance(row[flag], bool):
                return None
            values[flag] = row[flag]
        normalized[side] = values
    return normalized


def _bootstrap_ratio(
    pairs: Sequence[tuple[float, float]],
    *,
    seed: int = 98_240_117,
) -> dict[str, float]:
    if not pairs:
        raise PairedJudgeError("cannot bootstrap empty paired scores")
    rng = random.Random(seed)
    ratios = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        candidate = sum(row[0] for row in sample)
        teacher = sum(row[1] for row in sample)
        ratios.append(candidate / teacher if teacher else 0.0)
    ratios.sort()
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "lower_95": ratios[int(0.025 * BOOTSTRAP_SAMPLES)],
        "median": ratios[int(0.5 * BOOTSTRAP_SAMPLES)],
        "upper_95": ratios[int(0.975 * BOOTSTRAP_SAMPLES)],
    }


def _summary(rows: Sequence[Mapping[str, Any]], system: str) -> dict[str, Any]:
    selected = [row[system] for row in rows if row["parsed"]]
    if not selected:
        raise PairedJudgeError("judge produced no parseable scores")
    return {
        "observations": len(selected),
        "dimension_means": {
            dimension: statistics.fmean(
                float(row[dimension]) for row in selected
            )
            for dimension in DIMENSIONS
        },
        "flag_rates": {
            flag: sum(bool(row[flag]) for row in selected) / len(selected)
            for flag in FLAGS
        },
        "total_score": sum(
            int(row[dimension])
            for row in selected
            for dimension in DIMENSIONS
        ),
    }


def run_judge(
    *,
    protocol_path: Path,
    catalog_path: Path,
    candidate_evidence_path: Path,
    teacher_bundle_path: Path,
    output_path: Path,
    model: str,
    revision: str,
    license_id: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if output_path.exists():
        raise PairedJudgeError(f"judge evidence is immutable: {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_id = str(protocol.get("protocol_id"))
    split = str(protocol.get("catalog", {}).get("split", "final_test"))
    prompts_per_capability = int(
        protocol.get("catalog", {}).get(
            "prompts_per_capability",
            PROMPTS_PER_CAPABILITY,
        )
    )
    if (
        protocol_id not in {PROTOCOL_ID, VALIDATION_PROTOCOL_ID}
        or protocol["catalog"]["sha256"] != _sha256_file(catalog_path)
        or protocol["judge"]["model_id"] != model
        or protocol["judge"]["revision"] != revision
    ):
        raise PairedJudgeError("paired judge protocol identity changed")
    catalog = load_probe_catalog(catalog_path)
    probes = {
        str(probe["probe_id"]): probe for probe in catalog["probes"]
    }
    selected_ids = _selected_probe_ids(
        catalog,
        split=split,
        protocol_id=protocol_id,
        prompts_per_capability=prompts_per_capability,
    )
    candidate_outputs, candidate_identity = _candidate_outputs(
        candidate_evidence_path,
        split=split,
    )
    teacher_outputs, teacher_identity = _teacher_outputs(
        teacher_bundle_path,
        split=split,
    )
    if set(selected_ids) - set(candidate_outputs):
        raise PairedJudgeError("candidate evidence lacks selected prompts")
    if set(selected_ids) - set(teacher_outputs):
        raise PairedJudgeError("teacher evidence lacks selected prompts")

    requests = []
    orientation: dict[str, bool] = {}
    for probe_id in selected_ids:
        candidate_a = _candidate_is_a(
            probe_id,
            protocol_id=protocol_id,
        )
        orientation[probe_id] = candidate_a
        candidate = candidate_outputs[probe_id]
        teacher = teacher_outputs[probe_id]
        requests.append(
            {
                "prompt": _judge_prompt(
                    user_prompt=str(probes[probe_id]["prompt"]),
                    output_a=candidate if candidate_a else teacher,
                    output_b=teacher if candidate_a else candidate,
                ),
                "max_new_tokens": 256,
                "seed": 0,
                "temperature": 0.0,
            }
        )

    judge = HuggingFaceCausalSource(
        model,
        revision=revision,
        license_id=license_id,
        device=device,
        local_files_only=True,
        trust_remote_code=False,
        use_chat_template=True,
    )
    generated = []
    for start in range(0, len(requests), batch_size):
        generated.extend(
            judge.generate_batch(requests[start : start + batch_size])
        )

    observations = []
    for probe_id, sample in zip(selected_ids, generated, strict=True):
        parsed = _parse_scores(str(sample["output"]))
        candidate_a = orientation[probe_id]
        observation: dict[str, Any] = {
            "probe_id": probe_id,
            "capability": probes[probe_id]["capability"],
            "candidate_was_response_a": candidate_a,
            "judge_output": sample["output"],
            "judge_output_sha256": hashlib.sha256(
                str(sample["output"]).encode("utf-8")
            ).hexdigest(),
            "judge_tokens": sample["teacher_tokens"],
            "judge_token_counter": sample["teacher_token_counter"],
            "parsed": parsed is not None,
        }
        if parsed is not None:
            observation["candidate"] = parsed["A" if candidate_a else "B"]
            observation["teacher"] = parsed["B" if candidate_a else "A"]
        observations.append(observation)

    parse_rate = sum(row["parsed"] for row in observations) / len(observations)
    candidate_summary = _summary(observations, "candidate")
    teacher_summary = _summary(observations, "teacher")
    paired_totals = [
        (
            sum(int(row["candidate"][dimension]) for dimension in DIMENSIONS),
            sum(int(row["teacher"][dimension]) for dimension in DIMENSIONS),
        )
        for row in observations
        if row["parsed"]
    ]
    ratio = (
        candidate_summary["total_score"] / teacher_summary["total_score"]
        if teacher_summary["total_score"]
        else 0.0
    )
    bootstrap = _bootstrap_ratio(paired_totals)
    gates = {
        "minimum_parse_rate": parse_rate >= 0.95,
        "minimum_candidate_grammatical_fluency_mean": (
            candidate_summary["dimension_means"]["grammatical_fluency"]
            >= 3.25
        ),
        "minimum_candidate_coherence_mean": (
            candidate_summary["dimension_means"][
                "local_and_global_coherence"
            ]
            >= 3.25
        ),
        "minimum_candidate_grounding_mean": (
            candidate_summary["dimension_means"][
                "prompt_and_context_grounding"
            ]
            >= 3.0
        ),
        "minimum_candidate_adherence_mean": (
            candidate_summary["dimension_means"][
                "instruction_and_format_adherence"
            ]
            >= 3.0
        ),
        "zero_candidate_repetition_or_collapse": (
            candidate_summary["flag_rates"]["repetition_or_collapse"] == 0
        ),
        "zero_candidate_unusable": (
            candidate_summary["flag_rates"]["unusable_or_empty"] == 0
        ),
        "minimum_candidate_to_teacher_total_score_ratio": ratio >= 0.9,
        "maximum_grounding_mean_loss_vs_teacher": (
            teacher_summary["dimension_means"][
                "prompt_and_context_grounding"
            ]
            - candidate_summary["dimension_means"][
                "prompt_and_context_grounding"
            ]
            <= 0.25
        ),
        "minimum_bootstrap_lower_ratio": bootstrap["lower_95"] >= 0.85,
    }
    evidence: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": {
            "path_at_judging": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
            "protocol_id": protocol_id,
        },
        "catalog": {
            "path_at_judging": str(catalog_path),
            "sha256": _sha256_file(catalog_path),
            "split": split,
        },
        "candidate_identity": candidate_identity,
        "teacher_identity": teacher_identity,
        "judge_identity": judge.source_manifest,
        "selected_prompt_count": len(selected_ids),
        "prompts_per_capability": prompts_per_capability,
        "parse_rate": parse_rate,
        "candidate": candidate_summary,
        "teacher": teacher_summary,
        "candidate_to_teacher_total_score_ratio": ratio,
        "paired_bootstrap_ratio": bootstrap,
        "gates": gates,
        "judge_generated_tokens": sum(
            int(row["judge_tokens"]) for row in observations
        ),
        "observations": observations,
        "claim_boundary": (
            "This is a preregistered blinded automated-model comparison. It "
            "does not replace the separately required blinded human fluency "
            "audit and does not prove mathematical losslessness."
        ),
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="ABI_ENGLISH_PAIRED_FLUENCY_JUDGE_PROTOCOL.json",
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--candidate-evidence", required=True)
    parser.add_argument("--teacher-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    evidence = run_judge(
        protocol_path=Path(args.protocol).resolve(),
        catalog_path=Path(args.catalog).resolve(),
        candidate_evidence_path=Path(args.candidate_evidence).resolve(),
        teacher_bundle_path=Path(args.teacher_bundle).resolve(),
        output_path=Path(args.output).resolve(),
        model=args.model,
        revision=args.revision,
        license_id=args.license,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "parse_rate": evidence["parse_rate"],
                "candidate_to_teacher_total_score_ratio": evidence[
                    "candidate_to_teacher_total_score_ratio"
                ],
                "bootstrap_lower_95": evidence[
                    "paired_bootstrap_ratio"
                ]["lower_95"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
