"""Execute the preregistered R15B pre-existing representation campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.foreign_neural_state_r15.recipient_worker import summarize
from experiments.foreign_teacher_r12.custody import verify_r11_freeze
from experiments.native_isa_r11.core import transition_accuracy, write_package_once
from experiments.native_isa_r11.run import _host_matrix

from .isolation import run_wsl_isolated_extraction
from .protocol import evaluation_rows, heldout_capabilities
from .public_extraction import (
    LABEL_NAMESPACE,
    _control_accuracy,
    _source_identity,
    extract_with_source,
    load_source,
)
from .public_qualification import OPERATIONS, QualificationError
from .representation import decode_labels, decode_transition, labels_to_operations


def _write_label(
    path: Path,
    *,
    package_sha256: str,
    capability_id: str,
    labels: list[str],
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    value = {
        "format": "abi-r15b-semantic-label-manifest/1",
        "package_sha256": package_sha256,
        "capability_id": capability_id,
        "namespace": LABEL_NAMESPACE,
        "operation_labels": labels,
        "source_model_id": model_id,
        "source_revision": revision,
    }
    value["evidence_sha256"] = evidence_hash(value)
    write_json_once(path, value)
    return value


def run(config_path: Path, reveal_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R15B output exists: {output}")
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    if config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL":
        raise R14Error("R15B preregistration status changed")
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R14Error("R15B reveal is malformed") from exc
    if (
        len(secret) != 32
        or hashlib.sha256(secret).hexdigest() != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        raise R14Error("R15B reveal commitment changed")
    root = config_path.parents[3]
    for relative, expected in config["code_bindings"].items():
        if sha256_file(root / relative) != expected:
            raise R14Error(f"R15B code binding changed: {relative}")
    public = config["public_preflight"]
    if any(sha256_file(root / path) != digest for path, digest in public["files"].items()):
        raise R14Error("R15B public prerequisite binding changed")
    public_receipt = json_object(root / public["receipt"])
    if (
        public_receipt.get("status") != "PUBLIC_PREFLIGHT_NOT_HELDOUT_CERTIFICATION"
        or public_receipt.get("package_evaluation", {}).get("accuracy") != 1.0
    ):
        raise R14Error("R15B public prerequisite did not pass")
    r11 = verify_r11_freeze(root, config)
    heldout = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    evaluations = evaluation_rows(config, heldout)
    recipient_rows = [
        rows[: int(config["data"]["recipient_rows_per_capability"])] for rows in evaluations
    ]

    output.mkdir(parents=True)
    shutil.copyfile(reveal_path, output / "heldout_reveal.json")
    source_dir = output / "source"
    source_dir.mkdir()
    extraction_dir = output / "extractions"
    packages_dir = output / "packages"
    labels_dir = output / "labels"
    labels_dir.mkdir()
    source_config = config["source"]
    tokenizer, model, digit_ids, output_rows = load_source(
        model_id=str(source_config["model_id"]),
        model_revision=str(source_config["revision"]),
    )
    source_observations = []
    bundles = []
    transitions = []
    sources = []
    isolated = []
    for index, item in enumerate(heldout):
        residuals, observations, metrics = extract_with_source(
            tokenizer,
            model,
            digit_ids,
            output_rows,
            max_new_tokens=int(source_config["max_new_tokens"]),
            slot_order=item.slot_order,
        )
        if len(observations) != 6 or any(
            int(row["prediction"]) != int(row["answer"]) for row in observations
        ):
            raise QualificationError("R15B held-out source anchor failed")
        bundle_path = source_dir / f"anonymous-{index:02d}.safetensors"
        save_file(
            {"residuals": residuals, "output_rows": output_rows},
            str(bundle_path),
            metadata={"format": "abi-r15b-anonymous-pre-answer-representation/1"},
        )
        extraction = run_wsl_isolated_extraction(
            root,
            representation=bundle_path,
            destination=extraction_dir / f"anonymous-{index:02d}",
            distribution=str(config["physical_extraction"]["distribution"]),
        )
        labels = [int(value) for value in extraction["result"]["labels"]]
        expected_operations = [
            (OPERATIONS[semantic][1], OPERATIONS[semantic][2]) for semantic in item.slot_order
        ]
        if (
            labels_to_operations(labels) != expected_operations
            or min(float(value) for value in extraction["result"]["margins"]) <= 0
        ):
            raise R14Error("R15B isolated extraction recovered wrong capability")
        transition = decode_transition(residuals, output_rows)
        if decode_labels(residuals, output_rows) != labels:
            raise R14Error("R15B isolated and local decoders disagree")
        if transition_accuracy(transition, evaluations[index]) != 1.0:
            raise R14Error("R15B extracted transition failed held-out evaluation")
        bundles.append((residuals, bundle_path))
        transitions.append(transition)
        isolated.append(
            {
                "capability_id": item.capability.capability_id,
                "path": str((extraction_dir / f"anonymous-{index:02d}").relative_to(output)),
                "result_sha256": sha256_file(
                    extraction_dir / f"anonymous-{index:02d}" / "result.json"
                ),
                "launcher_sha256": sha256_file(
                    extraction_dir / f"anonymous-{index:02d}" / "launcher.json"
                ),
            }
        )
        for row in observations:
            source_observations.append({"capability_id": item.capability.capability_id, **row})
        sources.append(
            {
                "capability_id": item.capability.capability_id,
                "anchor_rows": 6,
                "anchor_accuracy": 1.0,
                "bundle": {
                    "path": str(bundle_path.relative_to(output)),
                    "bytes": bundle_path.stat().st_size,
                    "sha256": sha256_file(bundle_path),
                    "residual_shape": list(residuals.shape),
                    "output_rows_shape": list(output_rows.shape),
                },
                "metrics": metrics,
            }
        )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    before = torch.full((3, 8, 8), 1.0 / 8.0)
    source_identity = _source_identity(
        str(source_config["model_id"]), str(source_config["revision"])
    )
    before_package = write_package_once(
        packages_dir,
        before,
        {"teacher_before_sha256": source_identity, "teacher_after_sha256": source_identity},
    )
    after_packages = []
    label_records = []
    for index, (item, transition) in enumerate(zip(heldout, transitions)):
        package = write_package_once(
            packages_dir,
            transition,
            {"teacher_before_sha256": source_identity, "teacher_after_sha256": source_identity},
        )
        package["capability_id"] = item.capability.capability_id
        after_packages.append(package)
        label_path = labels_dir / f"label-{index:02d}.json"
        label = _write_label(
            label_path,
            package_sha256=package["sha256"],
            capability_id=item.capability.capability_id,
            labels=item.operation_labels,
            model_id=str(source_config["model_id"]),
            revision=str(source_config["revision"]),
        )
        label_records.append(
            {
                "path": str(label_path.relative_to(output)),
                "sha256": sha256_file(label_path),
                **label,
            }
        )
    manifest = {"before": before_package, "after": after_packages}
    write_json_once(output / "package_manifest.json", manifest)

    controls = []
    for index, (item, (residuals, _bundle)) in enumerate(zip(heldout, bundles)):
        values = {
            name: _control_accuracy(
                residuals,
                output_rows,
                evaluations[index],
                control=name,
            )
            for name in ("zero", "shuffle", "random", "head_shuffle")
        }
        wrong = decode_transition(bundles[(index + 1) % len(bundles)][0], output_rows)
        values["wrong"] = transition_accuracy(wrong, evaluations[index])
        if max(values.values()) > float(config["gates"]["negative_control_accuracy_maximum"]):
            raise R14Error("R15B representation negative-control gate failed")
        controls.append({"capability_id": item.capability.capability_id, **values})

    raw_source_path = output / "source_observations.jsonl"
    write_jsonl_once(raw_source_path, source_observations)
    recipient_workers = []
    codec_receipt = json_object(root / config["codec_freeze"]["receipt"])
    codec_tensors = load_file(str(root / config["codec_freeze"]["tensors"]), device="cpu")
    capabilities = [item.capability for item in heldout]
    for host_key in config["recipient_hosts"]:
        observations, host_receipt = _host_matrix(
            host_key,
            config,
            codec_tensors,
            codec_receipt,
            capabilities,
            recipient_rows,
            before,
            transitions,
            manifest,
        )
        host_dir = output / "recipients" / str(host_key)
        host_dir.mkdir(parents=True)
        rows_path = host_dir / "observations.jsonl"
        write_jsonl_once(rows_path, observations)
        summary = summarize(observations, recipient_rows)
        worker = {
            "format": "abi-r15b-recipient-worker/1",
            "host": host_key,
            "observations": {
                "path": str(rows_path.relative_to(output)),
                "rows": len(observations),
                "sha256": sha256_file(rows_path),
            },
            "host_receipt": host_receipt,
            "summary": summary,
        }
        worker["evidence_sha256"] = evidence_hash(worker)
        write_json_once(host_dir / "receipt.json", worker)
        recipient_workers.append(
            {
                "host": host_key,
                "path": str((host_dir / "receipt.json").relative_to(output)),
                "sha256": sha256_file(host_dir / "receipt.json"),
            }
        )

    receipt = {
        "format": "abi-r15b-preexisting-representation-run/1",
        "verification_status": "UNVERIFIED_RUN_OUTPUT",
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "r11_freeze": r11,
        "capabilities": [item.capability.capability_id for item in heldout],
        "source": {
            "model_id": source_config["model_id"],
            "revision": source_config["revision"],
            "training_steps": 0,
            "observations": {
                "path": raw_source_path.name,
                "rows": len(source_observations),
                "sha256": sha256_file(raw_source_path),
            },
            "capability_receipts": sources,
        },
        "isolated_extractions": isolated,
        "packages": manifest,
        "package_manifest_sha256": sha256_file(output / "package_manifest.json"),
        "semantic_labels": label_records,
        "controls": controls,
        "recipient_workers": recipient_workers,
        "information_accounting": {
            "raw_source_prompts": 6 * len(heldout),
            "source_generated_tokens": sum(item["metrics"]["generated_tokens"] for item in sources),
            "representation_bundle_bytes": sum(item[1].stat().st_size for item in bundles),
            "package_bytes": sum(item["bytes"] for item in after_packages),
            "source_training_steps": 0,
            "recipient_training_steps": 0,
        },
        "teacher_present_at_recipient_execution": False,
        "claim_ceiling": "BOUNDED_PREEXISTING_REPRESENTATION_CAPABILITY_RECOVERY_ONLY",
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.reveal, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
