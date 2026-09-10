"""Fresh live replay of the frozen R23 semantic candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)
from experiments.generative_transfer_r21.live_verify_v6 import _package_for, _prompt
from experiments.generative_transfer_r21.live_verify_v7 import (
    targeted_tensor_corruption,
)
from experiments.generative_transfer_r21.protocol import LABELS, SEEDS, WordLabeler

from .acquire import _jsonl, validate_source
from .binding import load_config, load_seed_reveal
from .live_binding import load_live_config
from .protocol import hidden_rows, semantic_score
from .verify import verify as verify_stored


def _inventory(engine: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "path": package["path"],
                "bytes": package["bytes"],
                "sha256": package["sha256"],
            }
            for seed in engine["systems"].values()
            for method in seed.values()
            for package in method["packages"]
        ),
        key=lambda row: row["path"],
    )


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R23 live result exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    live = load_live_config(root, config_path)
    r23_path = (root / str(live["r23_config"]["path"])).resolve()
    reveal_path = (root / str(live["seed_reveal"]["path"])).resolve()
    source_run = (root / str(live["source_receipt"]["path"])).parent.resolve()
    candidate = (root / str(live["candidate_wrapper"]["path"])).parent.resolve()
    stored = verify_stored(r23_path, reveal_path, source_run, candidate)
    if stored["status"] != "PASS_VERIFIED_SEMANTIC_REPLICATION":
        raise R14Error("R23 stored prerequisite did not pass")
    if not torch.cuda.is_available():
        raise R14Error("R23 live replay requires CUDA")
    if "transformers" in sys.modules:
        raise R14Error("R23 source software was loaded before package execution")
    config = load_config(root, r23_path)
    seed = load_seed_reveal(config, reveal_path)
    expected = hidden_rows(seed)
    source_rows = validate_source(root, r23_path, reveal_path, source_run)
    source_by_id = {row["record_id"]: row for row in source_rows}
    engine = json_object(root / str(live["candidate_engine"]["path"]))
    observations = _jsonl(root / str(live["candidate_observations"]["path"]))
    stored_by_key = {
        (row["method"], int(row["seed"]), row["record_id"]): row
        for row in observations
    }
    expected_keys = {
        (method, seed_value, row["record_id"])
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
        for seed_value in SEEDS
        for row in expected
    }
    if set(stored_by_key) != expected_keys or _inventory(engine) != config["packages"]:
        raise R14Error("R23 candidate matrix or package inventory changed")
    labeler = WordLabeler(
        json_object(root / str(config["public_candidate_labeler"]["path"]))
    )
    base = json_object(root / str(config["base_config"]["path"]))
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(base["layercake"]["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    api = base_run._layercake(root)
    from layercake.cake.installer import InstallationError

    signer = api["key_id"](public_pem)
    package_count = 0
    source_parameters = 0
    receiver_steps = 0
    for seed_systems in engine["systems"].values():
        for system in seed_systems.values():
            for package in system["packages"]:
                loaded = api["load_package"](
                    root / package["path"], trust_store={signer: public_pem}
                )
                if (
                    not loaded.signed
                    or loaded.manifest.input_contract.get("mode")
                    != "direct_selected_portable_decoder"
                ):
                    raise R14Error("R23 package signature or input contract failed")
                source_parameters += int(
                    loaded.manifest.training_data_provenance["source_parameters_copied"]
                )
                receiver_steps += int(
                    loaded.manifest.training_data_provenance["receiver_training_steps"]
                )
                package_count += 1
    if package_count != 24:
        raise R14Error("R23 package inventory is incomplete")

    output.mkdir(parents=True)
    started = time.perf_counter()
    live_rows = []
    cpu_rows = []
    lifecycle = []
    for seed_value in SEEDS:
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized"):
            print(json.dumps({"live_system": method, "seed": seed_value}), flush=True)
            system = engine["systems"][str(seed_value)][method]
            with tempfile.TemporaryDirectory(
                prefix=f"r23-live-{method}-{seed_value}-"
            ) as raw:
                raw_path = Path(raw)
                host = api["DirectCakeHost"](
                    raw_path / "gpu",
                    abi_version=base["layercake"]["abi_version"],
                    abi_hash=base["layercake"]["abi_sha256"],
                    trust_store={signer: public_pem},
                    device="cuda",
                )
                cpu_host = api["DirectCakeHost"](
                    raw_path / "cpu",
                    abi_version=base["layercake"]["abi_version"],
                    abi_hash=base["layercake"]["abi_sha256"],
                    trust_store={signer: public_pem},
                    device="cpu",
                )
                for package in system["packages"]:
                    installed = host.install(root / package["path"])
                    cpu_host.install(root / package["path"])
                    if installed["archive_hash"] != package["sha256"]:
                        raise R14Error("R23 installed archive changed")
                for position, row in enumerate(expected):
                    predicted, prompt = _prompt(method, row, labeler)
                    package = _package_for(system, method, seed_value, predicted)
                    text = host.generate(
                        package["cake_id"], prompt, maximum_actions=256
                    ).output.decode("utf-8")
                    saved = stored_by_key[(method, seed_value, row["record_id"])]
                    if text != saved["output"]:
                        raise R14Error("R23 fresh GPU output changed")
                    live_rows.append(
                        {
                            "method": method,
                            "seed": seed_value,
                            "position": position,
                            "record_id": row["record_id"],
                            "task": row["task"],
                            "predicted_label": predicted,
                            "cake_id": package["cake_id"],
                            "output": text,
                            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            **semantic_score(
                                row,
                                text,
                                source_by_id[row["record_id"]]["teacher_output"],
                            ),
                        }
                    )
                selected = [
                    next(row for row in expected if row["task"] == task)
                    for task in LABELS
                ]
                for row in selected:
                    predicted, prompt = _prompt(method, row, labeler)
                    package = _package_for(system, method, seed_value, predicted)
                    text = cpu_host.generate(
                        package["cake_id"], prompt, maximum_actions=256
                    ).output.decode("utf-8")
                    saved = stored_by_key[(method, seed_value, row["record_id"])]
                    if text != saved["output"]:
                        raise R14Error("R23 fresh CPU output changed")
                    cpu_rows.append(
                        {
                            "method": method,
                            "seed": seed_value,
                            "record_id": row["record_id"],
                            "task": row["task"],
                            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    )
                for package_index, package in enumerate(system["packages"]):
                    row = (
                        next(
                            item
                            for item in expected
                            if package["cake_id"].endswith(
                                f"-{item['task']}-seed{seed_value}"
                            )
                        )
                        if method == "abi_factorized"
                        else selected[0]
                    )
                    predicted, prompt = _prompt(method, row, labeler)
                    saved = stored_by_key[(method, seed_value, row["record_id"])][
                        "output"
                    ]
                    removed = host.remove(package["cake_id"])
                    absent_rejected = False
                    try:
                        host.generate(package["cake_id"], prompt, maximum_actions=256)
                    except (KeyError, FileNotFoundError):
                        absent_rejected = True
                    restored = host.install(root / package["path"])
                    restored_text = host.generate(
                        package["cake_id"], prompt, maximum_actions=256
                    ).output.decode("utf-8")
                    corrupt_path = raw_path / f"corrupt-{package_index}.cake"
                    corrupt_path.write_bytes(
                        targeted_tensor_corruption((root / package["path"]).read_bytes())
                    )
                    corrupt_host = api["DirectCakeHost"](
                        raw_path / f"corrupt-{package_index}",
                        abi_version=base["layercake"]["abi_version"],
                        abi_hash=base["layercake"]["abi_sha256"],
                        trust_store={signer: public_pem},
                        device="cuda",
                    )
                    corrupt_rejected = False
                    try:
                        corrupt_host.install(corrupt_path)
                    except InstallationError:
                        corrupt_rejected = True
                    lifecycle.append(
                        {
                            "method": method,
                            "seed": seed_value,
                            "cake_id": package["cake_id"],
                            "removed": removed.get("cake_id") == package["cake_id"],
                            "absent_rejected": absent_rejected,
                            "restored_archive_exact": restored["archive_hash"]
                            == package["sha256"],
                            "restored_output_exact": restored_text == saved,
                            "targeted_corruption_rejected": corrupt_rejected,
                        }
                    )
            del host, cpu_host
            gc.collect()
            torch.cuda.empty_cache()

    aggregates = base_run._aggregate(live_rows)
    teacher_scores = [
        semantic_score(row, source_by_id[row["record_id"]]["teacher_output"], "")
        for row in expected
    ]
    teacher_functional = sum(row["functional_pass"] for row in teacher_scores)
    label_exact = sum(labeler.predict(row["instruction"]) == row["task"] for row in expected)
    gates = {
        "label_exact": label_exact >= int(config["gates"]["minimum_label_exact"]),
        "each_seed_each_task": all(
            min(aggregates[f"abi_factorized:{seed_value}"]["functional_by_task"].values())
            >= int(config["gates"]["minimum_task_functional_per_seed"])
            for seed_value in SEEDS
        ),
        "each_seed_non_hallucinating": all(
            aggregates[f"abi_factorized:{seed_value}"]["non_hallucinating"]
            >= int(config["gates"]["minimum_non_hallucinating_per_seed"])
            for seed_value in SEEDS
        ),
        "each_seed_non_collapsed": all(
            aggregates[f"abi_factorized:{seed_value}"]["non_collapsed"]
            >= int(config["gates"]["minimum_non_collapsed_per_seed"])
            for seed_value in SEEDS
        ),
        "each_seed_no_worse_than_teacher": all(
            aggregates[f"abi_factorized:{seed_value}"]["functional"]
            >= teacher_functional
            for seed_value in SEEDS
        ),
        "each_seed_no_worse_than_controls": all(
            aggregates[f"abi_factorized:{seed_value}"]["functional"]
            >= aggregates[f"{method}:{seed_value}"]["functional"]
            for seed_value in SEEDS
            for method in ("raw_sequence", "labeled_monolith")
        ),
        "all_packages_frozen": _inventory(engine) == config["packages"],
        "gpu_rows_complete": len(live_rows) == 1080,
        "cpu_gpu_rows_complete": len(cpu_rows) == 54,
        "teacher_absent_at_execution": "transformers" not in sys.modules,
        "student_retraining_zero": receiver_steps == 0,
    }
    controls_pass = all(
        all(
            row[key]
            for key in (
                "removed",
                "absent_rejected",
                "restored_archive_exact",
                "restored_output_exact",
                "targeted_corruption_rejected",
            )
        )
        for row in lifecycle
    )
    if (
        aggregates != engine["aggregates"]
        or teacher_functional != engine["teacher"]["functional"]
        or gates != engine["gates"]
        or not all(gates.values())
        or len(cpu_rows) != 54
        or len(lifecycle) != 24
        or not controls_pass
        or source_parameters != 0
    ):
        raise R14Error("R23 fresh live recomputation failed")
    live_path = output / "live_observations.jsonl"
    cpu_path = output / "cpu_observations.jsonl"
    lifecycle_path = output / "lifecycle_observations.jsonl"
    write_jsonl_once(live_path, live_rows)
    write_jsonl_once(cpu_path, cpu_rows)
    write_jsonl_once(lifecycle_path, lifecycle)
    receipt = {
        "format": "abi-r23-fresh-live-verification/1",
        "status": "PASS_FRESH_LIVE_SEMANTIC_REPLICATION",
        "config_sha256": sha256_file(config_path),
        "candidate_engine_sha256": sha256_file(
            root / str(live["candidate_engine"]["path"])
        ),
        "candidate_observations_sha256": sha256_file(
            root / str(live["candidate_observations"]["path"])
        ),
        "live_observations": {
            "path": live_path.name,
            "rows": len(live_rows),
            "sha256": sha256_file(live_path),
        },
        "cpu_observations": {
            "path": cpu_path.name,
            "rows": len(cpu_rows),
            "sha256": sha256_file(cpu_path),
        },
        "lifecycle_observations": {
            "path": lifecycle_path.name,
            "rows": len(lifecycle),
            "sha256": sha256_file(lifecycle_path),
        },
        "packages_signature_verified": package_count,
        "package_removals": sum(row["removed"] for row in lifecycle),
        "package_restorations": sum(row["restored_output_exact"] for row in lifecycle),
        "targeted_corruptions_rejected": sum(
            row["targeted_corruption_rejected"] for row in lifecycle
        ),
        "source_parameters_in_packages": source_parameters,
        "receiver_training_steps": receiver_steps,
        "source_software_loaded": "transformers" in sys.modules,
        "execution_pid": os.getpid(),
        "recomputed_gates": gates,
        "recomputed_aggregates": aggregates,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_claim": "R23_BOUNDED_FRESH_SEMANTIC_SUPPLIED_CONTENT_TRANSFER",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "full_abi_moonshot": "OPEN",
    }
    receipt["evidence_sha256"] = selfless_evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.config, args.output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
