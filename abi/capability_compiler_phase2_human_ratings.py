"""Lock, unblind, and score completed Phase 2 human-rating forms fail-closed."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    Phase2Error,
    canonical_json_bytes,
    sha256_file,
)


FORM_COUNT = 3
PREFERENCE_VALUES = {"A", "B", "TIE", "BOTH_UNACCEPTABLE"}
RATING_FIELDS = {
    "preference",
    "fluency_A_1_to_5",
    "fluency_B_1_to_5",
    "grounding_and_adherence_A_1_to_5",
    "grounding_and_adherence_B_1_to_5",
    "repetition_or_collapse_A",
    "repetition_or_collapse_B",
    "rater_comment",
}
SCORE_FIELDS = {
    "fluency_A_1_to_5",
    "fluency_B_1_to_5",
    "grounding_and_adherence_A_1_to_5",
    "grounding_and_adherence_B_1_to_5",
}
BOOL_FIELDS = {"repetition_or_collapse_A", "repetition_or_collapse_B"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1729
PREFERENCE_LOWER_BOUND_MINIMUM = 0.45


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2Error(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for physical_line, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase2Error(f"invalid JSONL at {path}:{physical_line}") from exc
        if not isinstance(value, dict):
            raise Phase2Error(f"non-object JSONL row at {path}:{physical_line}")
        rows.append(value)
    return rows


def _form_name(index: int, *, completed: bool) -> str:
    suffix = ".completed" if completed else ""
    return f"rater_form_{index}{suffix}.jsonl"


def _validate_attestations(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("format") != "abi-capability-compiler-phase2-rater-attestations/1":
        raise Phase2Error("rater attestation format changed")
    raters = value.get("raters")
    if not isinstance(raters, list) or len(raters) != FORM_COUNT:
        raise Phase2Error("exactly three rater attestations are required")
    expected_forms = {f"rater_form_{index}" for index in range(1, FORM_COUNT + 1)}
    identities: set[str] = set()
    observed_forms: set[str] = set()
    for row in raters:
        if not isinstance(row, dict):
            raise Phase2Error("invalid rater attestation")
        identity = row.get("rater_id")
        form = row.get("rater_form")
        if not isinstance(identity, str) or not identity.strip() or identity in identities:
            raise Phase2Error("three distinct nonempty rater identifiers are required")
        if form not in expected_forms or form in observed_forms:
            raise Phase2Error("each rater must attest to exactly one distinct form")
        for field in (
            "human_rater",
            "completed_independently",
            "answer_key_not_accessed_before_lock",
            "other_ratings_not_accessed_before_lock",
        ):
            if row.get(field) is not True:
                raise Phase2Error(f"rater attestation is missing true field: {field}")
        completed = row.get("completed_utc")
        if not isinstance(completed, str) or not completed.strip():
            raise Phase2Error("rater completion time is required")
        disclosure = row.get("conflict_disclosure")
        if not isinstance(disclosure, str):
            raise Phase2Error("rater conflict disclosure is required")
        identities.add(identity)
        observed_forms.add(form)
    if observed_forms != expected_forms:
        raise Phase2Error("rater form attestations are incomplete")
    custodian = value.get("custodian")
    if not isinstance(custodian, dict) or not isinstance(custodian.get("custodian_id"), str):
        raise Phase2Error("a packet custodian identifier is required")
    if custodian.get("verified_three_distinct_people") is not True:
        raise Phase2Error("custodian did not verify three distinct people")
    if custodian.get("withheld_answer_key_and_cross_rater_access_until_lock") is not True:
        raise Phase2Error("custodian did not attest to blinded custody")
    return value


def validate_completed_forms(
    *, packet_dir: Path, completed_dir: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate ratings without reading the answer key."""
    manifest_path = packet_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("format") != "abi-capability-compiler-phase2-human-rating-packet/1"
        or manifest.get("rater_forms") != FORM_COUNT
        or manifest.get("status") != "AWAITING_THREE_INDEPENDENT_HUMAN_RATERS"
    ):
        raise Phase2Error("sealed human-rating packet changed")

    bindings: dict[str, dict[str, Any]] = {}
    all_ids: set[str] = set()
    expected_rows = int(manifest["pairs_per_form"])
    for index in range(1, FORM_COUNT + 1):
        template_name = _form_name(index, completed=False)
        completed_name = _form_name(index, completed=True)
        template_path = packet_dir / template_name
        completed_path = completed_dir / completed_name
        template_binding = manifest["file_bindings"][template_name]
        if sha256_file(template_path) != template_binding["sha256"]:
            raise Phase2Error("sealed rater template hash changed")
        templates = _read_jsonl(template_path)
        completed = _read_jsonl(completed_path)
        if len(templates) != expected_rows or len(completed) != expected_rows:
            raise Phase2Error("completed rater form depth changed")
        template_by_id = {str(row.get("rating_id")): row for row in templates}
        completed_by_id = {str(row.get("rating_id")): row for row in completed}
        if len(template_by_id) != expected_rows or set(completed_by_id) != set(template_by_id):
            raise Phase2Error("completed rater form identities changed")
        for rating_id, row in completed_by_id.items():
            template = template_by_id[rating_id]
            if set(row) != set(template):
                raise Phase2Error("completed rater form schema changed")
            if rating_id in all_ids:
                raise Phase2Error("rating identity was reused across forms")
            all_ids.add(rating_id)
            for field in set(template) - RATING_FIELDS:
                if row[field] != template[field]:
                    raise Phase2Error(f"blinded prompt/output field changed: {field}")
            if row["preference"] not in PREFERENCE_VALUES:
                raise Phase2Error("preference is incomplete or invalid")
            for field in SCORE_FIELDS:
                score = row[field]
                if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                    raise Phase2Error(f"invalid required 1-to-5 score: {field}")
            for field in BOOL_FIELDS:
                if not isinstance(row[field], bool):
                    raise Phase2Error(f"invalid required Boolean rating: {field}")
            if row["rater_comment"] is not None and not isinstance(row["rater_comment"], str):
                raise Phase2Error("rater comment must be text or null")
        bindings[completed_name] = {
            "rows": len(completed),
            "sha256": sha256_file(completed_path),
        }
    if len(all_ids) != int(manifest["ratings_required"]):
        raise Phase2Error("completed rating identity depth changed")
    return manifest, bindings


def lock_completed_forms(*, packet_dir: Path, completed_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise Phase2Error("immutable blinded lock receipt already exists")
    manifest, forms = validate_completed_forms(packet_dir=packet_dir, completed_dir=completed_dir)
    attestations_path = completed_dir / "attestations.json"
    attestations = _validate_attestations(attestations_path)
    result = {
        "format": "abi-capability-compiler-phase2-blinded-rating-lock/1",
        "status": "LOCKED_COMPLETE_BEFORE_UNBLINDING",
        "packet_manifest_sha256": sha256_file(packet_dir / "manifest.json"),
        "ratings": int(manifest["ratings_required"]),
        "independent_raters_attested": len(attestations["raters"]),
        "completed_form_bindings": forms,
        "attestations": {
            "path": "attestations.json",
            "sha256": sha256_file(attestations_path),
        },
        "blinding_key_read_by_lock_tool": False,
        "claim_boundary": "Software validates the declarations and distinct identifiers; the named custodian, not software, establishes that the raters are three independent humans and that custody was blinded.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(result))
    return result


def _candidate_credit(preference: str, candidate_is_a: bool) -> float:
    if preference == "TIE":
        return 0.5
    if preference == "BOTH_UNACCEPTABLE":
        return 0.0
    return float((preference == "A") == candidate_is_a)


def _bootstrap(values: Mapping[str, Sequence[float]], *, resamples: int, seed: int) -> dict[str, float | int]:
    if set(values) != set(CAPABILITIES) or any(not rows for rows in values.values()):
        raise Phase2Error("human-rating bootstrap strata changed")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means = []
        for capability in CAPABILITIES:
            array = np.asarray(values[capability], dtype=np.float64)
            means.append(float(rng.choice(array, size=array.size, replace=True).mean()))
        draws[index] = float(np.mean(means))
    observed = float(np.mean([np.mean(rows) for rows in values.values()]))
    return {
        "point": observed,
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def _verify_lock(*, packet_dir: Path, completed_dir: Path, lock_path: Path) -> dict[str, Any]:
    lock = _read_json(lock_path)
    if (
        lock.get("format") != "abi-capability-compiler-phase2-blinded-rating-lock/1"
        or lock.get("status") != "LOCKED_COMPLETE_BEFORE_UNBLINDING"
        or lock.get("blinding_key_read_by_lock_tool") is not False
        or lock.get("packet_manifest_sha256") != sha256_file(packet_dir / "manifest.json")
    ):
        raise Phase2Error("blinded lock receipt changed")
    manifest, forms = validate_completed_forms(packet_dir=packet_dir, completed_dir=completed_dir)
    attestations_path = completed_dir / "attestations.json"
    _validate_attestations(attestations_path)
    if forms != lock.get("completed_form_bindings"):
        raise Phase2Error("completed forms changed after blinded lock")
    if sha256_file(attestations_path) != lock.get("attestations", {}).get("sha256"):
        raise Phase2Error("attestations changed after blinded lock")
    if lock.get("ratings") != manifest["ratings_required"]:
        raise Phase2Error("blinded lock rating count changed")
    return lock


def score_locked_forms(
    *,
    packet_dir: Path,
    completed_dir: Path,
    lock_path: Path,
    output: Path | None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if output is not None and output.exists():
        raise Phase2Error("immutable human-rating score manifest already exists")
    if resamples != BOOTSTRAP_RESAMPLES or seed != BOOTSTRAP_SEED:
        raise Phase2Error("preregistered human-rating bootstrap changed")
    lock = _verify_lock(
        packet_dir=packet_dir, completed_dir=completed_dir, lock_path=lock_path
    )
    packet = _read_json(packet_dir / "manifest.json")
    key_path = packet_dir / "blinding_key.jsonl"
    key_binding = packet["file_bindings"]["blinding_key.jsonl"]
    if sha256_file(key_path) != key_binding["sha256"]:
        raise Phase2Error("blinding key changed")
    key = {str(row["rating_id"]): row for row in _read_jsonl(key_path)}
    if len(key) != int(packet["ratings_required"]):
        raise Phase2Error("blinding key depth changed")

    per_pair: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    diagnostics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for index in range(1, FORM_COUNT + 1):
        path = completed_dir / _form_name(index, completed=True)
        for row in _read_jsonl(path):
            rating_id = str(row["rating_id"])
            binding = key.get(rating_id)
            if binding is None or binding["pair_id"] != row["pair_id"]:
                raise Phase2Error("completed rating does not match blinding key")
            candidate = str(binding["candidate_system"])
            capability = str(row["capability"])
            candidate_is_a = binding["system_A"] == candidate
            if (binding["system_A"] == candidate) == (binding["system_B"] == candidate):
                raise Phase2Error("invalid candidate/reference position binding")
            credit = _candidate_credit(str(row["preference"]), candidate_is_a)
            per_pair[(candidate, capability, str(row["pair_id"]))].append(credit)
            counts[candidate][str(row["preference"])] += 1
            candidate_side = "A" if candidate_is_a else "B"
            reference_side = "B" if candidate_is_a else "A"
            diagnostics[candidate]["candidate_fluency"].append(float(row[f"fluency_{candidate_side}_1_to_5"]))
            diagnostics[candidate]["reference_fluency"].append(float(row[f"fluency_{reference_side}_1_to_5"]))
            diagnostics[candidate]["candidate_grounding"].append(float(row[f"grounding_and_adherence_{candidate_side}_1_to_5"]))
            diagnostics[candidate]["reference_grounding"].append(float(row[f"grounding_and_adherence_{reference_side}_1_to_5"]))
            diagnostics[candidate]["candidate_repetition"].append(float(row[f"repetition_or_collapse_{candidate_side}"]))
            diagnostics[candidate]["reference_repetition"].append(float(row[f"repetition_or_collapse_{reference_side}"]))

    systems: dict[str, Any] = {}
    for candidate in packet["candidate_systems"]:
        strata: dict[str, list[float]] = {capability: [] for capability in CAPABILITIES}
        for (system, capability, _pair_id), credits in per_pair.items():
            if system != candidate:
                continue
            if len(credits) != FORM_COUNT:
                raise Phase2Error("each candidate/prompt pair requires three ratings")
            strata[capability].append(float(np.mean(credits)))
        interval = _bootstrap(strata, resamples=resamples, seed=seed)
        diag = diagnostics[candidate]
        systems[candidate] = {
            "ratings": sum(counts[candidate].values()),
            "paired_prompts": sum(len(rows) for rows in strata.values()),
            "preference_counts": dict(sorted(counts[candidate].items())),
            "candidate_preference_credit": interval,
            "meets_0_45_lower_bound": interval["lower_95"] >= PREFERENCE_LOWER_BOUND_MINIMUM,
            "diagnostics": {name: float(np.mean(values)) for name, values in sorted(diag.items())},
        }

    result = {
        "format": "abi-capability-compiler-phase2-human-rating-score/1",
        "status": "PASS",
        "packet_manifest_sha256": sha256_file(packet_dir / "manifest.json"),
        "blind_lock_sha256": sha256_file(lock_path),
        "independent_raters": FORM_COUNT,
        "ratings": int(packet["ratings_required"]),
        "scoring": {
            "candidate_win_credit": 1.0,
            "tie_credit": 0.5,
            "reference_win_credit": 0.0,
            "both_unacceptable_credit": 0.0,
            "unit": "candidate-prompt cluster averaging three raters",
            "stratification": "fourteen English capabilities",
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
            "confidence_level": 0.95,
            "preference_lower_bound_minimum": PREFERENCE_LOWER_BOUND_MINIMUM,
        },
        "systems": systems,
        "phase2_human_evidence_complete": True,
        "claim_boundary": "PASS means the preregistered human-evidence workflow is complete and valid. Individual baseline promotion thresholds are reported per system and are not forced to pass.",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(result))
    return result


def verify_scored_manifest(*, root: Path, path: Path) -> dict[str, Any]:
    packet_dir = root / "results/abi_capability_compiler_phase2/human_rating_packet_v1"
    completed_dir = path.parent
    lock_path = completed_dir / "blind_lock.json"
    packet = _read_json(packet_dir / "manifest.json")
    value = _read_json(path)
    if (
        value.get("format") != "abi-capability-compiler-phase2-human-rating-score/1"
        or value.get("status") != "PASS"
        or value.get("independent_raters") != FORM_COUNT
        or value.get("ratings") != packet.get("ratings_required")
        or value.get("packet_manifest_sha256") != sha256_file(packet_dir / "manifest.json")
        or value.get("blind_lock_sha256") != sha256_file(lock_path)
        or value.get("phase2_human_evidence_complete") is not True
    ):
        raise Phase2Error("completed human-rating manifest changed")
    expected = score_locked_forms(
        packet_dir=packet_dir,
        completed_dir=completed_dir,
        lock_path=lock_path,
        output=None,
    )
    if expected != value:
        raise Phase2Error("completed human-rating statistics are not reproducible")
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--packet-dir", required=True)
    lock_parser.add_argument("--completed-dir", required=True)
    lock_parser.add_argument("--output", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--packet-dir", required=True)
    score_parser.add_argument("--completed-dir", required=True)
    score_parser.add_argument("--lock", required=True)
    score_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "lock":
        result = lock_completed_forms(
            packet_dir=Path(args.packet_dir).resolve(),
            completed_dir=Path(args.completed_dir).resolve(),
            output=Path(args.output).resolve(),
        )
    else:
        result = score_locked_forms(
            packet_dir=Path(args.packet_dir).resolve(),
            completed_dir=Path(args.completed_dir).resolve(),
            lock_path=Path(args.lock).resolve(),
            output=Path(args.output).resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
