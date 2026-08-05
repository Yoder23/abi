"""Build the blinded, counterbalanced Phase 2 human-rating packet."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    Phase2Error,
    canonical_json_bytes,
    sha256_file,
)


PACKET_SEED = 1729
REFERENCE = "T0"
CANDIDATES = ("L0", "L1", "D0", "D1", "D2")
ANCHOR_SEED = 104729
OUTPUT_PATHS = {
    "T0": "results/abi_capability_compiler_phase2/teacher/T0/development_outputs.jsonl",
    "L0": "results/abi_capability_compiler_phase2/headline/L0/r16-lr1e-4-exp1-seed104729/development_outputs.jsonl",
    "L1": "results/abi_capability_compiler_phase2/headline/L1/r32-lr1e-4-exp1-seed104729/development_outputs.jsonl",
    "D0": "results/abi_capability_compiler_phase2/headline/D0/lr3e-5-exp4-seed104729/development_outputs.jsonl",
    "D1": "results/abi_capability_compiler_phase2/headline/D1/lr3e-5-exp4-seed104729/development_outputs.jsonl",
    "D2": "results/abi_capability_compiler_phase2/headline/D2/lr3e-5-exp4-seed104729/development_outputs.jsonl",
}
CATALOG_PATH = "catalogs/capability_compiler_phase1_frozen_v1.json"
FORM_COUNT = 3
PROMPTS_PER_CAPABILITY = 100


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def _output_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    expected = len(CAPABILITIES) * PROMPTS_PER_CAPABILITY
    if len(rows) != expected:
        raise Phase2Error(f"human packet input depth changed: {path}")
    mapped = {str(row["probe_id"]): row for row in rows}
    if len(mapped) != expected:
        raise Phase2Error(f"human packet input has duplicate probes: {path}")
    counts = Counter(str(row["capability"]) for row in rows)
    if set(counts) != set(CAPABILITIES) or set(counts.values()) != {PROMPTS_PER_CAPABILITY}:
        raise Phase2Error(f"human packet capability depth changed: {path}")
    return mapped


def _prompts(root: Path) -> dict[str, dict[str, str]]:
    catalog = json.loads((root / CATALOG_PATH).read_text(encoding="utf-8"))
    rows = [
        row
        for row in catalog["probes"]
        if row.get("split") == "validation"
        and row.get("canonical_capability") in CAPABILITIES
    ]
    expected = len(CAPABILITIES) * PROMPTS_PER_CAPABILITY
    mapped = {
        str(row["probe_id"]): {
            "capability": str(row["canonical_capability"]),
            "prompt": str(row["prompt"]),
        }
        for row in rows
    }
    if len(rows) != expected or len(mapped) != expected:
        raise Phase2Error("human packet prompt catalog depth changed")
    return mapped


def candidate_first(pair_index: int, form_index: int) -> bool:
    """Return an exactly balanced, three-form A/B assignment.

    Forms zero and one are exact reversals for every pair. Form two is also
    globally balanced, while guaranteeing each pair appears in both orders.
    """

    if pair_index < 0 or not 0 <= form_index < FORM_COUNT:
        raise Phase2Error("invalid counterbalance index")
    base = pair_index % 2 == 0
    if form_index == 0:
        return base
    if form_index == 1:
        return not base
    return (pair_index // 2) % 2 == 0


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise Phase2Error(f"immutable human-rating artifact exists: {path}")
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")


def build_packet(*, root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise Phase2Error("immutable human-rating packet already exists")

    prompts = _prompts(root)
    outputs = {
        system: _output_map(root / relative)
        for system, relative in OUTPUT_PATHS.items()
    }
    identities = set(prompts)
    if any(set(rows) != identities for rows in outputs.values()):
        raise Phase2Error("human packet paired prompt identities changed")

    probe_ids = sorted(
        identities,
        key=lambda probe_id: (prompts[probe_id]["capability"], probe_id),
    )
    forms: list[list[dict[str, Any]]] = [[] for _ in range(FORM_COUNT)]
    answer_key: list[dict[str, Any]] = []
    position_counts = {
        f"rater_form_{form_index + 1}": Counter()
        for form_index in range(FORM_COUNT)
    }

    pair_index = 0
    for candidate in CANDIDATES:
        for probe_id in probe_ids:
            prompt = prompts[probe_id]
            if (
                outputs[REFERENCE][probe_id]["capability"] != prompt["capability"]
                or outputs[candidate][probe_id]["capability"] != prompt["capability"]
            ):
                raise Phase2Error("human packet capability identity changed")
            pair_id = f"{candidate}-vs-{REFERENCE}-{probe_id}"
            for form_index in range(FORM_COUNT):
                form_id = f"rater_form_{form_index + 1}"
                first_is_candidate = candidate_first(pair_index, form_index)
                system_a = candidate if first_is_candidate else REFERENCE
                system_b = REFERENCE if first_is_candidate else candidate
                rating_id = f"{form_id}-{pair_id}"
                forms[form_index].append(
                    {
                        "rating_id": rating_id,
                        "pair_id": pair_id,
                        "rater_form": form_id,
                        "capability": prompt["capability"],
                        "prompt": prompt["prompt"],
                        "output_A": str(outputs[system_a][probe_id]["output"]),
                        "output_B": str(outputs[system_b][probe_id]["output"]),
                        "preference": None,
                        "fluency_A_1_to_5": None,
                        "fluency_B_1_to_5": None,
                        "grounding_and_adherence_A_1_to_5": None,
                        "grounding_and_adherence_B_1_to_5": None,
                        "repetition_or_collapse_A": None,
                        "repetition_or_collapse_B": None,
                        "rater_comment": None,
                    }
                )
                answer_key.append(
                    {
                        "rating_id": rating_id,
                        "pair_id": pair_id,
                        "rater_form": form_id,
                        "probe_id": probe_id,
                        "candidate_system": candidate,
                        "reference_system": REFERENCE,
                        "system_A": system_a,
                        "system_B": system_b,
                    }
                )
                position_counts[form_id][f"{candidate}_A" if first_is_candidate else f"{candidate}_B"] += 1
            pair_index += 1

    expected_pairs = len(CANDIDATES) * len(probe_ids)
    for form_index, rows in enumerate(forms):
        if len(rows) != expected_pairs:
            raise Phase2Error("human packet form depth changed")
        rng = random.Random(PACKET_SEED + form_index)
        rng.shuffle(rows)
    if len(answer_key) != expected_pairs * FORM_COUNT:
        raise Phase2Error("human packet answer-key depth changed")
    answer_key.sort(key=lambda row: str(row["rating_id"]))

    file_bindings: dict[str, dict[str, Any]] = {}
    for form_index, rows in enumerate(forms):
        path = output_dir / f"rater_form_{form_index + 1}.jsonl"
        _write_jsonl(path, rows)
        file_bindings[path.name] = {
            "rows": len(rows),
            "sha256": sha256_file(path),
        }
    key_path = output_dir / "blinding_key.jsonl"
    _write_jsonl(key_path, answer_key)
    file_bindings[key_path.name] = {
        "rows": len(answer_key),
        "sha256": sha256_file(key_path),
        "access": "RESTRICT_FROM_RATERS_UNTIL_ALL_FORMS_ARE_LOCKED",
    }

    manifest = {
        "format": "abi-capability-compiler-phase2-human-rating-packet/1",
        "status": "AWAITING_THREE_INDEPENDENT_HUMAN_RATERS",
        "packet_seed": PACKET_SEED,
        "anchor_seed": ANCHOR_SEED,
        "reference_system": REFERENCE,
        "candidate_systems": list(CANDIDATES),
        "distinct_prompts": len(probe_ids),
        "pairs_per_form": expected_pairs,
        "rater_forms": FORM_COUNT,
        "ratings_required": expected_pairs * FORM_COUNT,
        "independence_rule": "Assign exactly one complete form to each of three independent human raters. Raters must not see the blinding key or one another's ratings until all three forms are locked.",
        "preference_values": ["A", "B", "TIE", "BOTH_UNACCEPTABLE"],
        "rubric": [
            "Correctly addresses the supplied prompt without inventing unsupported requirements.",
            "Uses fluent grammatical English and coherent organization.",
            "Follows requested tone, format, clarification, or abstention behavior.",
            "Avoids repetition loops, collapse, fragments, and irrelevant domain facts.",
        ],
        "scoring_boundary": "Automated functional flags and system identities are absent from rater forms. Preference statistics may be unblinded only after all three completed forms are immutable.",
        "counterbalance": {
            "forms_1_and_2_are_exact_pairwise_reversals": True,
            "each_candidate_has_equal_A_and_B_positions_per_form": True,
            "position_counts": {
                form: dict(sorted(counts.items()))
                for form, counts in position_counts.items()
            },
        },
        "input_bindings": {
            system: {
                "path": relative,
                "sha256": sha256_file(root / relative),
            }
            for system, relative in OUTPUT_PATHS.items()
        },
        "file_bindings": file_bindings,
        "final_prompts_accessed": False,
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    manifest = build_packet(root=root, output_dir=Path(args.output_dir).resolve())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "distinct_prompts": manifest["distinct_prompts"],
                "pairs_per_form": manifest["pairs_per_form"],
                "ratings_required": manifest["ratings_required"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
