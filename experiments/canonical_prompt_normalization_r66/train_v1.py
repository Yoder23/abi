"""Train the corrected R66 canonical-prompt successor."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from abi.capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from abi import layercake_full_core_acquisition as acquisition
from experiments.foreign_capability_r14.core import write_json_once


BROAD_SHA256 = "82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150"
ANCHOR_SHA256 = "f6a27cae529d990ba67f1a26bb9cd79f97ba5ca0965c9debd0fc98dab8dba820"
PARENT_SHA256 = "65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b"
CANONICAL_ABI_SHA256 = "d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761"
SEED = 66_001
POLICY_SUFFIXES = (
    (
        "broad_english_form",
        "\n\nRespond in English. This is English-form acquisition: follow the "
        "request, but do not introduce outside factual claims, named entities, "
        "numerical facts, or specialist knowledge. If the request depends on "
        "missing or specialist information, state that it is not supplied or ask "
        "one concise clarification question.",
        19_337,
    ),
    (
        "supplied_text_linguistic",
        "\n\nRespond in English. Use only the supplied text and the linguistic "
        "instruction above. Do not introduce outside facts. If the request cannot "
        "be completed from the supplied text, ask one concise clarification "
        "question or say that the information is not supplied.",
        4_769,
    ),
)
EXPECTED_UNCHANGED = 315


class TrainingError(RuntimeError):
    pass


def run(
    broad: Path,
    anchor: Path,
    layercake_root: Path,
    parent: Path,
    canonical_abi: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise TrainingError(f"immutable R66 output exists: {output}")
    frozen = (
        (broad, BROAD_SHA256),
        (anchor, ANCHOR_SHA256),
        (parent / "model.safetensors", PARENT_SHA256),
        (canonical_abi, CANONICAL_ABI_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or sha256_file(path) != digest:
            raise TrainingError(f"R66 frozen input changed: {path}")

    original_loader = acquisition.load_english_training_rows
    broad_resolved = broad.resolve()
    audit_rows: list[dict[str, Any]] = []
    loader_calls = Counter()

    def normalized_loader(bundle_path, *, budget_index):
        rows, budget, bundle = original_loader(bundle_path, budget_index=budget_index)
        resolved = Path(bundle_path).resolve()
        loader_calls[str(resolved)] += 1
        if resolved != broad_resolved:
            return rows, budget, bundle
        transformed = []
        for row in rows:
            original_prompt = str(row["prompt"])
            normalized = original_prompt
            matched_name = None
            for name, suffix, _ in POLICY_SUFFIXES:
                if original_prompt.endswith(suffix):
                    if matched_name is not None:
                        raise TrainingError("prompt matched multiple policy suffixes")
                    normalized = original_prompt[: -len(suffix)].rstrip()
                    matched_name = name
            if not normalized:
                raise TrainingError("canonical normalization produced an empty prompt")
            updated = dict(row)
            updated["prompt"] = normalized
            transformed.append(updated)
            audit_rows.append(
                {
                    "record_id": str(row["record_id"]),
                    "capability": str(row["capability"]),
                    "suffix_variant": matched_name or "none",
                    "original_prompt_sha256": hashlib.sha256(
                        original_prompt.encode("utf-8")
                    ).hexdigest(),
                    "normalized_prompt_sha256": hashlib.sha256(
                        normalized.encode("utf-8")
                    ).hexdigest(),
                }
            )
        counts = Counter(row["suffix_variant"] for row in audit_rows)
        expected = {name: count for name, _, count in POLICY_SUFFIXES}
        expected["none"] = EXPECTED_UNCHANGED
        if dict(counts) != expected:
            raise TrainingError(
                f"normalization inventory changed before training: {dict(counts)}"
            )
        return transformed, budget, bundle

    acquisition.load_english_training_rows = normalized_loader
    try:
        exit_code = acquisition.main(
            [
                "--bundle", str(broad), "--layercake-root", str(layercake_root),
                "--parent", str(parent), "--canonical-abi", str(canonical_abi),
                "--output", str(output), "--budget-index", "-1",
                "--seed", str(SEED), "--steps", "6000", "--batch-size", "8",
                "--gradient-accumulation-steps", "1",
                "--trainable-scope", "deep_capability_adapter_cakes",
                "--shared-learning-rate", "0.00002", "--cake-learning-rate", "0.0001",
                "--classifier-loss-weight", "0.25", "--prompt-overlap-loss-weight", "1.0",
                "--max-tokens", "256", "--recovery-start-step", "400",
                "--recovery-interval", "8", "--recovery-horizons", "8,16,32",
                "--sampling-strategy", "balanced_capabilities",
                "--anchor-bundle", str(anchor), "--anchor-budget-index", "-1",
                "--anchor-batch-size", "4", "--anchor-loss-weight", "1.0",
                "--anchor-sampling-strategy", "balanced_capabilities",
                "--exclude-overlength-prompts", "--parent-logit-preservation-weight", "0.5",
                "--device", "cuda",
            ]
        )
    finally:
        acquisition.load_english_training_rows = original_loader
    if exit_code != 0:
        raise TrainingError(f"R66 trainer exited {exit_code}")
    if len(audit_rows) != len({row["record_id"] for row in audit_rows}):
        raise TrainingError("R66 normalization audit IDs are not unique")
    counts = Counter(row["suffix_variant"] for row in audit_rows)
    expected = {name: count for name, _, count in POLICY_SUFFIXES}
    expected["none"] = EXPECTED_UNCHANGED
    if dict(counts) != expected:
        raise TrainingError(f"normalization inventory changed: {dict(counts)}")
    audit_payload = b"".join(canonical_json_bytes(row) for row in audit_rows)
    receipt = {
        "format": "abi-r66-complete-canonical-host-prompt-normalization/1",
        "status": "COMPLETE_UNSCREENED", "seed": SEED,
        "broad_archive_sha256": BROAD_SHA256, "anchor_archive_sha256": ANCHOR_SHA256,
        "parent_checkpoint_sha256": PARENT_SHA256,
        "canonical_abi_sha256": CANONICAL_ABI_SHA256,
        "policy_suffixes": [
            {"name": name, "rows": count, "sha256": hashlib.sha256(suffix.lstrip("\n").encode("utf-8")).hexdigest()}
            for name, suffix, count in POLICY_SUFFIXES
        ],
        "audited_records": len(audit_rows), "normalization_counts": dict(counts),
        "normalization_by_capability": {
            variant: dict(sorted(Counter(
                row["capability"] for row in audit_rows if row["suffix_variant"] == variant
            ).items())) for variant in sorted(counts)
        },
        "audit_rows_sha256": hashlib.sha256(audit_payload).hexdigest(),
        "original_prompt_hashes_sha256": hashlib.sha256(
            "\n".join(row["original_prompt_sha256"] for row in audit_rows).encode("utf-8")
        ).hexdigest(),
        "normalized_prompt_hashes_sha256": hashlib.sha256(
            "\n".join(row["normalized_prompt_sha256"] for row in audit_rows).encode("utf-8")
        ).hexdigest(),
        "loader_calls": dict(sorted(loader_calls.items())), "new_teacher_outputs": 0,
        "r60_outputs_used_for_training": 0, "teacher_calls": 0,
        "checkpoint_sha256": sha256_file(output / "model.safetensors"),
        "metadata_sha256": sha256_file(output / "metadata.json"),
        "promotion_eligible": False, "full_abi_moonshot": "OPEN",
    }
    write_json_once(output / "normalization_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--canonical-abi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.broad.resolve(), args.anchor.resolve(), args.layercake_root.resolve(),
        args.parent.resolve(), args.canonical_abi.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
