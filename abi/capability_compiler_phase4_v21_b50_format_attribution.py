"""Read-only attribution of B50 seed104729 format-control failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json, _rows
from .capability_compiler_phase4_v19_frontier_verify import _without


FORMAT = "abi-capability-compiler-phase4-v21-b50-format-attribution/1"
TOKEN = re.compile(r"[A-Za-z]+|\d+|[^\w\s]", re.UNICODE)


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def classify_delta(candidate: str, reference: str) -> dict[str, Any]:
    candidate_lf = candidate.replace("\r\n", "\n")
    reference_lf = reference.replace("\r\n", "\n")
    candidate_tokens = TOKEN.findall(candidate_lf)
    reference_tokens = TOKEN.findall(reference_lf)
    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    recalled = sum(
        min(count, candidate_counts[token]) for token, count in reference_counts.items()
    )
    if candidate_lf == reference_lf:
        primary = "exact"
    elif reference_lf.startswith(candidate_lf):
        primary = "truncated_canonical_prefix"
    elif candidate_lf.startswith(reference_lf):
        primary = "extra_suffix"
    elif candidate_lf.endswith(reference_lf):
        primary = "extra_prefix"
    elif candidate_lf.count("\n") != reference_lf.count("\n"):
        primary = "line_structure_mismatch"
    elif " ".join(candidate_lf.split()) == " ".join(reference_lf.split()):
        primary = "whitespace_or_linebreak_only"
    elif candidate_lf.casefold() == reference_lf.casefold():
        primary = "case_only"
    else:
        primary = "lexical_or_identifier_mismatch"
    first_mismatch = next(
        (
            index
            for index, (left, right) in enumerate(zip(candidate_lf, reference_lf))
            if left != right
        ),
        min(len(candidate_lf), len(reference_lf)),
    )
    return {
        "primary": primary,
        "utf8_edit_distance": edit_distance(candidate_lf, reference_lf),
        "first_mismatch_character": first_mismatch,
        "candidate_characters": len(candidate_lf),
        "reference_characters": len(reference_lf),
        "candidate_lines": candidate_lf.count("\n") + 1,
        "reference_lines": reference_lf.count("\n") + 1,
        "reference_token_recall": recalled / len(reference_tokens)
        if reference_tokens
        else 1.0,
        "candidate_is_reference_prefix": reference_lf.startswith(candidate_lf),
        "reference_is_candidate_prefix": candidate_lf.startswith(reference_lf),
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_B50_FORMAT_ATTRIBUTION"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B50 format-attribution governance changed")
    if sorted(int(seed) for seed in protocol["seed_outputs"]) != [104729, 130363, 155921]:
        raise Phase3Error("B50 format-attribution seed set changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B50 format-attribution binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable B50 format attribution exists: {output}")
    source = _json(root / protocol["source_result"])
    source_evidence = hashlib.sha256(
        canonical_json_bytes(_without(source, "evidence_sha256"))
    ).hexdigest()
    probes = {
        str(probe["probe_id"]): probe
        for probe in development_probes(root / protocol["development_catalog"])
        if probe["canonical_capability"] == "format_control"
    }
    teacher = {
        str(row["probe_id"]): row
        for row in _rows(root / protocol["teacher_reference"])
    }
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    flag_recomputations: list[bool] = []
    for seed_text, path in protocol["seed_outputs"].items():
        seed = int(seed_text)
        rows = [
            row
            for row in _rows(root / path)
            if row["capability"] == "format_control"
        ]
        by_seed[seed] = {str(row["probe_id"]): row for row in rows}
        for row in rows:
            probe = probes[str(row["probe_id"])]
            flag_recomputations.append(
                bool(row["functional_pass_v1"])
                == evaluate_functional(str(row["output"]), probe["evaluator"])
            )

    failures = [
        probe_id
        for probe_id, row in by_seed[104729].items()
        if not bool(row["functional_pass_v1"])
    ]
    records: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    both_other_seeds_pass = 0
    any_other_seed_exact_candidate = 0
    for probe_id in sorted(failures):
        probe = probes[probe_id]
        reference = str(teacher[probe_id]["output"])
        candidate = str(by_seed[104729][probe_id]["output"])
        delta = classify_delta(candidate, reference)
        categories[delta["primary"]] += 1
        other = {
            str(seed): {
                "functional_pass_v1": bool(by_seed[seed][probe_id]["functional_pass_v1"]),
                "output": str(by_seed[seed][probe_id]["output"]),
                "exact_teacher_output": str(by_seed[seed][probe_id]["output"])
                == reference,
                "exact_seed104729_output": str(by_seed[seed][probe_id]["output"])
                == candidate,
            }
            for seed in (130363, 155921)
        }
        if all(value["functional_pass_v1"] for value in other.values()):
            both_other_seeds_pass += 1
        if any(value["exact_seed104729_output"] for value in other.values()):
            any_other_seed_exact_candidate += 1
        records.append(
            {
                "probe_id": probe_id,
                "template_family": probe["phase1_template_family"],
                "evaluator": probe["evaluator"],
                "seed104729_output": candidate,
                "cached_teacher_output": reference,
                "delta": delta,
                "other_seeds": other,
            }
        )

    if both_other_seeds_pass == len(failures) and failures:
        measured_owner = "seed_dependent_acquisition_or_optimization_instability"
    elif both_other_seeds_pass:
        measured_owner = "no_seed104729_failures"
    else:
        measured_owner = "mixed_coverage_and_seed_instability"
    gates = {
        "source_result_hash": sha256_file(root / protocol["source_result"])
        == protocol["bindings"][protocol["source_result"]],
        "source_evidence_hash": source_evidence == source["evidence_sha256"],
        "three_registered_seeds": sorted(by_seed) == [104729, 130363, 155921],
        "one_hundred_format_rows_per_seed": all(len(rows) == 100 for rows in by_seed.values()),
        "identical_probe_sets": len({tuple(sorted(rows)) for rows in by_seed.values()}) == 1,
        "all_functional_flags_recomputed": all(flag_recomputations),
        "seed104729_failure_count_ten": len(failures) == 10,
        "all_rows_noncollapsed": all(
            not bool(row["repetition_collapse_v2"])
            for rows in by_seed.values()
            for row in rows.values()
        ),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-v21-b50-format-attribution-result/1",
        "status": "PASS_READ_ONLY_B50_FORMAT_ATTRIBUTION"
        if all(gates.values())
        else "FAIL_B50_FORMAT_ATTRIBUTION",
        "protocol_sha256": protocol_sha,
        "source_result_sha256": sha256_file(root / protocol["source_result"]),
        "source_evidence_sha256": source["evidence_sha256"],
        "gates": gates,
        "seed104729_failures": len(failures),
        "both_other_seeds_pass": both_other_seeds_pass,
        "any_other_seed_exact_seed104729_output": any_other_seed_exact_candidate,
        "primary_categories": dict(sorted(categories.items())),
        "measured_owner": measured_owner,
        "records": records,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "architecture_selected": False,
        "phase4_certified": False,
        "claim_boundary": (
            "Read-only attribution of ten B50 seed104729 format-control failures. "
            "No stabilization design, minimum, matched baseline, final test, Phase 4, "
            "or ABI-superiority claim."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(
        output,
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
