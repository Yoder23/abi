"""Train the single preregistered R63 prompt-invariant sparse successor."""

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
SEED = 63_001
PREFIXES = (
    "",
    "Please handle the request below. ",
    "For this exchange, respond to the following. ",
    "The next item is a user request. ",
    "Carefully read and complete this task. ",
    "A user has provided this instruction. ",
    "Reference {nonce}. ",
    "Item {number}. ",
)


class TrainingError(RuntimeError):
    pass


def _view(record_id: str) -> tuple[int, str]:
    digest = hashlib.sha256(("r63:" + record_id).encode("utf-8")).digest()
    index = digest[0] % len(PREFIXES)
    prefix = PREFIXES[index].format(
        nonce=hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:10],
        number=int.from_bytes(digest[1:5], "big") % 100_000,
    )
    return index, prefix


def run(
    broad: Path,
    anchor: Path,
    layercake_root: Path,
    parent: Path,
    canonical_abi: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise TrainingError(f"immutable R63 output exists: {output}")
    frozen = (
        (broad, BROAD_SHA256),
        (anchor, ANCHOR_SHA256),
        (parent / "model.safetensors", PARENT_SHA256),
        (canonical_abi, CANONICAL_ABI_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or sha256_file(path) != digest:
            raise TrainingError(f"R63 frozen input changed: {path}")

    original_loader = acquisition.load_english_training_rows
    broad_resolved = broad.resolve()
    audit_rows: list[dict[str, Any]] = []
    loader_calls = Counter()

    def augmented_loader(bundle_path, *, budget_index):
        rows, budget, bundle = original_loader(bundle_path, budget_index=budget_index)
        resolved = Path(bundle_path).resolve()
        loader_calls[str(resolved)] += 1
        if resolved != broad_resolved:
            return rows, budget, bundle
        transformed = []
        for row in rows:
            original_prompt = str(row["prompt"])
            view_index, prefix = _view(str(row["record_id"]))
            updated = dict(row)
            updated["prompt"] = prefix + original_prompt
            transformed.append(updated)
            audit_rows.append(
                {
                    "record_id": str(row["record_id"]),
                    "view_index": view_index,
                    "prefix_utf8_bytes": len(prefix.encode("utf-8")),
                    "original_prompt_sha256": hashlib.sha256(
                        original_prompt.encode("utf-8")
                    ).hexdigest(),
                    "augmented_prompt_sha256": hashlib.sha256(
                        updated["prompt"].encode("utf-8")
                    ).hexdigest(),
                }
            )
        return transformed, budget, bundle

    acquisition.load_english_training_rows = augmented_loader
    try:
        exit_code = acquisition.main(
            [
                "--bundle", str(broad),
                "--layercake-root", str(layercake_root),
                "--parent", str(parent),
                "--canonical-abi", str(canonical_abi),
                "--output", str(output),
                "--budget-index", "-1",
                "--seed", str(SEED),
                "--steps", "6000",
                "--batch-size", "8",
                "--gradient-accumulation-steps", "1",
                "--trainable-scope", "deep_capability_adapter_cakes",
                "--shared-learning-rate", "0.00002",
                "--cake-learning-rate", "0.0001",
                "--classifier-loss-weight", "0.25",
                "--prompt-overlap-loss-weight", "1.0",
                "--max-tokens", "256",
                "--recovery-start-step", "400",
                "--recovery-interval", "8",
                "--recovery-horizons", "8,16,32",
                "--sampling-strategy", "balanced_capabilities",
                "--anchor-bundle", str(anchor),
                "--anchor-budget-index", "-1",
                "--anchor-batch-size", "4",
                "--anchor-loss-weight", "1.0",
                "--anchor-sampling-strategy", "balanced_capabilities",
                "--exclude-overlength-prompts",
                "--parent-logit-preservation-weight", "0.5",
                "--device", "cuda",
            ]
        )
    finally:
        acquisition.load_english_training_rows = original_loader
    if exit_code != 0:
        raise TrainingError(f"R63 trainer exited {exit_code}")
    if len(audit_rows) != len({row["record_id"] for row in audit_rows}):
        raise TrainingError("R63 augmentation record audit is not unique")
    audit_payload = b"".join(canonical_json_bytes(row) for row in audit_rows)
    receipt = {
        "format": "abi-r63-capability-blind-prompt-augmentation/1",
        "status": "COMPLETE_UNSCREENED",
        "seed": SEED,
        "broad_archive_sha256": BROAD_SHA256,
        "anchor_archive_sha256": ANCHOR_SHA256,
        "parent_checkpoint_sha256": PARENT_SHA256,
        "canonical_abi_sha256": CANONICAL_ABI_SHA256,
        "augmented_records": len(audit_rows),
        "identity_view_records": sum(row["view_index"] == 0 for row in audit_rows),
        "view_counts": dict(
            sorted(Counter(str(row["view_index"]) for row in audit_rows).items())
        ),
        "audit_rows_sha256": hashlib.sha256(audit_payload).hexdigest(),
        "original_prompt_hashes_sha256": hashlib.sha256(
            "\n".join(row["original_prompt_sha256"] for row in audit_rows).encode("utf-8")
        ).hexdigest(),
        "augmented_prompt_hashes_sha256": hashlib.sha256(
            "\n".join(row["augmented_prompt_sha256"] for row in audit_rows).encode("utf-8")
        ).hexdigest(),
        "loader_calls": dict(sorted(loader_calls.items())),
        "new_teacher_outputs": 0,
        "r60_outputs_used_for_training": 0,
        "teacher_calls": 0,
        "checkpoint_sha256": sha256_file(output / "model.safetensors"),
        "metadata_sha256": sha256_file(output / "metadata.json"),
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
    }
    write_json_once(output / "augmentation_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--canonical-abi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.broad.resolve(),
        args.anchor.resolve(),
        args.layercake_root.resolve(),
        args.parent.resolve(),
        args.canonical_abi.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()

