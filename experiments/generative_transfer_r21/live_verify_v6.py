"""Fresh live recomputation and real-host replay for the R21 public candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

from . import run as base_run
from .hash_assurance_binding import selfless_evidence_hash
from .live_binding import load_live_config
from .protocol import (
    LABELS,
    SEEDS,
    WordLabeler,
    bootstrap_difference,
    evaluation_rows,
    instruction_from_prompt,
    normalized_prompt,
    score_output,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error(f"R21 live JSONL unreadable: {path}") from exc


def _configs(root: Path, live: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json_object(root / str(live["manifest_repair_config"]["path"]))
    assurance = json_object(root / str(manifest["hash_assurance_config"]["path"]))
    control = json_object(root / str(assurance["label_control_config"]["path"]))
    base_path = root / str(control["base_config"]["path"])
    return json_object(base_path), {"base_path": base_path}


def _prompt(method: str, row: dict[str, Any], labeler: WordLabeler) -> tuple[str, str]:
    label = labeler.predict(instruction_from_prompt(str(row["prompt"])))
    prompt = (
        str(row["prompt"])
        if method == "raw_sequence"
        else normalized_prompt(label, dict(row["slots"]))
    )
    return label, prompt


def _package_for(system: dict[str, Any], method: str, seed: int, label: str) -> dict[str, Any]:
    if method != "abi_factorized":
        return system["packages"][0]
    suffix = f"-{label}-seed{seed}"
    selected = [row for row in system["packages"] if row["cake_id"].endswith(suffix)]
    if len(selected) != 1:
        raise R14Error("R21 live factor selection is ambiguous")
    return selected[0]


def _stored_checks(
    root: Path,
    live: dict[str, Any],
    base: dict[str, Any],
    engine: dict[str, Any],
    observations: list[dict[str, Any]],
    labeler: WordLabeler,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    if (
        engine.get("format") != "abi-r21-public-generative-bakeoff-result/1"
        or engine.get("verdict") != "PASS_PUBLIC_PREREQUISITE"
        or engine.get("evidence_sha256") != selfless_evidence_hash(engine)
        or len(observations) != 1080
    ):
        raise R14Error("R21 stored candidate result failed")
    expected = evaluation_rows()
    expected_by_id = {row["record_id"]: row for row in expected}
    teacher_rows = base_run._evaluation_teacher(root, base)
    teacher_by_id = {row["record_id"]: str(row["output"]) for row in teacher_rows}
    expected_keys = {
        (method, seed, row["record_id"])
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
        for seed in SEEDS
        for row in expected
    }
    actual_keys = {
        (row.get("method"), row.get("seed"), row.get("record_id")) for row in observations
    }
    if actual_keys != expected_keys:
        raise R14Error("R21 stored observation matrix changed")
    for row in observations:
        reference = expected_by_id[str(row["record_id"])]
        teacher = teacher_by_id[str(row["record_id"])]
        score = score_output(reference, str(row["output"]), teacher)
        if any(row.get(key) != value for key, value in score.items()):
            raise R14Error("R21 stored score changed")
        if (
            row.get("output_sha256") != hashlib.sha256(str(row["output"]).encode()).hexdigest()
            or row.get("teacher_output_sha256") != hashlib.sha256(teacher.encode()).hexdigest()
        ):
            raise R14Error("R21 stored output identity changed")
    aggregates = base_run._aggregate(observations)
    if aggregates != engine.get("aggregates"):
        raise R14Error("R21 stored aggregate recomputation changed")
    return expected, teacher_by_id, aggregates


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 live verification exists: {output}")
    root = Path(__file__).resolve().parents[2]
    live = load_live_config(root, config_path)
    if not torch.cuda.is_available():
        raise R14Error("R21 live verification requires registered CUDA execution")
    base, _ = _configs(root, live)
    wrapper = json_object(root / str(live["candidate_wrapper"]["path"]))
    engine = json_object(root / str(live["candidate_engine"]["path"]))
    if (
        wrapper.get("verdict") != "PASS_PUBLIC_PREREQUISITE"
        or wrapper.get("evidence_sha256") != selfless_evidence_hash(wrapper)
        or wrapper.get("engine_result", {}).get("sha256") != live["candidate_engine"]["sha256"]
    ):
        raise R14Error("R21 candidate wrapper failed")
    observations = _jsonl(root / str(live["candidate_observations"]["path"]))
    labeler = WordLabeler(json_object(root / str(live["candidate_labeler"]["path"])))
    expected, teacher_by_id, recomputed_aggregates = _stored_checks(
        root, live, base, engine, observations, labeler
    )
    stored = {(row["method"], int(row["seed"]), row["record_id"]): row for row in observations}
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
    source_parameter_count = 0
    receiver_training_steps = 0
    for seed in engine["systems"].values():
        for system in seed.values():
            for package in system["packages"]:
                loaded = api["load_package"](
                    root / package["path"], trust_store={signer: public_pem}
                )
                if (
                    not loaded.signed
                    or loaded.manifest.input_contract.get("mode")
                    != "direct_selected_portable_decoder"
                ):
                    raise R14Error("R21 live package signature or contract failed")
                source_parameter_count += int(
                    loaded.manifest.training_data_provenance["source_parameters_copied"]
                )
                receiver_training_steps += int(
                    loaded.manifest.training_data_provenance["receiver_training_steps"]
                )
                package_count += 1
    if package_count != 24:
        raise R14Error("R21 live signed package inventory is incomplete")

    output.mkdir(parents=True)
    started = time.perf_counter()
    live_rows = []
    cpu_rows = []
    lifecycle = []
    for seed in SEEDS:
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized"):
            print(json.dumps({"live_system": method, "seed": seed}), flush=True)
            system = engine["systems"][str(seed)][method]
            with tempfile.TemporaryDirectory(prefix=f"r21-live-{method}-{seed}-") as raw:
                raw_path = Path(raw)
                host = api["DirectCakeHost"](
                    raw_path / "gpu",
                    abi_version=base["layercake"]["abi_version"],
                    abi_hash=base["layercake"]["abi_sha256"],
                    trust_store={signer: public_pem},
                    device="cuda",
                )
                for package in system["packages"]:
                    installed = host.install(root / package["path"])
                    if installed["archive_hash"] != package["sha256"]:
                        raise R14Error("R21 live installed archive changed")
                for position, row in enumerate(expected):
                    predicted, prompt = _prompt(method, row, labeler)
                    package = _package_for(system, method, seed, predicted)
                    generated = host.generate(package["cake_id"], prompt, maximum_actions=256)
                    output_text = generated.output.decode("utf-8")
                    stored_row = stored[(method, seed, row["record_id"])]
                    if output_text != stored_row["output"]:
                        raise R14Error("R21 fresh GPU output changed")
                    score = score_output(row, output_text, teacher_by_id[row["record_id"]])
                    live_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "position": position,
                            "record_id": row["record_id"],
                            "task": row["task"],
                            "predicted_label": predicted,
                            "cake_id": package["cake_id"],
                            "output": output_text,
                            "output_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
                            **score,
                        }
                    )
                cpu_host = api["DirectCakeHost"](
                    raw_path / "cpu",
                    abi_version=base["layercake"]["abi_version"],
                    abi_hash=base["layercake"]["abi_sha256"],
                    trust_store={signer: public_pem},
                    device="cpu",
                )
                for package in system["packages"]:
                    cpu_host.install(root / package["path"])
                selected_rows = [
                    next(row for row in expected if row["task"] == task) for task in LABELS
                ]
                for row in selected_rows:
                    predicted, prompt = _prompt(method, row, labeler)
                    package = _package_for(system, method, seed, predicted)
                    cpu_output = cpu_host.generate(
                        package["cake_id"], prompt, maximum_actions=256
                    ).output.decode("utf-8")
                    gpu_output = stored[(method, seed, row["record_id"])]["output"]
                    if cpu_output != gpu_output:
                        raise R14Error("R21 fresh CPU/GPU output changed")
                    cpu_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "record_id": row["record_id"],
                            "task": row["task"],
                            "output_sha256": hashlib.sha256(cpu_output.encode()).hexdigest(),
                        }
                    )
                for package_index, package in enumerate(system["packages"]):
                    if method == "abi_factorized":
                        row = next(
                            row
                            for row in expected
                            if package["cake_id"].endswith(f"-{row['task']}-seed{seed}")
                        )
                    else:
                        row = selected_rows[package_index % len(selected_rows)]
                    predicted, prompt = _prompt(method, row, labeler)
                    expected_output = stored[(method, seed, row["record_id"])]["output"]
                    removed = host.remove(package["cake_id"])
                    absent_rejected = False
                    try:
                        host.generate(package["cake_id"], prompt, maximum_actions=256)
                    except (KeyError, FileNotFoundError):
                        absent_rejected = True
                    installed = host.install(root / package["path"])
                    restored = host.generate(
                        package["cake_id"], prompt, maximum_actions=256
                    ).output.decode("utf-8")
                    original = (root / package["path"]).read_bytes()
                    corrupted = bytearray(original)
                    corrupted[-1] ^= 1
                    corrupt_path = raw_path / f"corrupt-{package_index}.cake"
                    corrupt_path.write_bytes(corrupted)
                    corrupt_host = api["DirectCakeHost"](
                        raw_path / f"corrupt-registry-{package_index}",
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
                            "seed": seed,
                            "cake_id": package["cake_id"],
                            "removed": removed.get("cake_id") == package["cake_id"],
                            "absent_rejected": absent_rejected,
                            "restored_archive_exact": installed["archive_hash"]
                            == package["sha256"],
                            "restored_output_exact": restored == expected_output,
                            "corrupt_rejected": corrupt_rejected,
                        }
                    )
            del host, cpu_host
            gc.collect()
            torch.cuda.empty_cache()

    if base_run._aggregate(live_rows) != recomputed_aggregates:
        raise R14Error("R21 fresh live aggregates changed")
    if len(cpu_rows) != 54 or len(lifecycle) != 24:
        raise R14Error("R21 fresh control matrix is incomplete")
    if not all(
        all(
            row[key]
            for key in (
                "removed",
                "absent_rejected",
                "restored_archive_exact",
                "restored_output_exact",
                "corrupt_rejected",
            )
        )
        for row in lifecycle
    ):
        raise R14Error("R21 fresh lifecycle control failed")

    evaluation = evaluation_rows()
    teacher_scores = [
        score_output(row, teacher_by_id[row["record_id"]], teacher_by_id[row["record_id"]])
        for row in evaluation
    ]
    headline_seed = int(engine["headline_seed"])
    headline = recomputed_aggregates[f"abi_factorized:{headline_seed}"]
    label_exact = sum(
        labeler.predict(instruction_from_prompt(row["prompt"])) == row["task"] for row in evaluation
    )
    ratios = [
        engine["systems"][str(seed)]["abi_factorized"]["deployed_parameters"]
        / engine["systems"][str(seed)]["labeled_monolith"]["deployed_parameters"]
        for seed in SEEDS
    ]
    gates = {
        "label_exact": label_exact >= int(base["gates"]["minimum_label_exact"]),
        "headline_each_task": min(headline["functional_by_task"].values())
        >= int(base["gates"]["minimum_task_functional"]),
        "headline_non_hallucinating": headline["non_hallucinating"]
        >= int(base["gates"]["minimum_non_hallucinating"]),
        "headline_non_collapsed": headline["non_collapsed"]
        >= int(base["gates"]["minimum_non_collapsed"]),
        "no_worse_than_teacher": headline["functional"]
        >= sum(row["functional_pass"] for row in teacher_scores),
        "no_worse_than_raw_sequence": headline["functional"]
        >= recomputed_aggregates[f"raw_sequence:{headline_seed}"]["functional"],
        "no_worse_than_labeled_monolith": headline["functional"]
        >= recomputed_aggregates[f"labeled_monolith:{headline_seed}"]["functional"],
        "factor_parameter_budget": max(ratios)
        <= float(base["gates"]["maximum_factor_to_labeled_parameters"]),
        "row_exposure_exact": all(
            engine["systems"][str(seed)][method]["training"]["row_exposure"]
            == int(base["training"]["row_exposures_per_method"])
            for seed in SEEDS
            for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
        ),
        "all_packages_signed": package_count == 24,
        "cpu_gpu_execution": len(cpu_rows) == 54,
        "causal_lifecycle": len(lifecycle) == 24,
        "teacher_absent_at_execution": "transformers" not in sys.modules,
        "source_parameters_in_packages_zero": source_parameter_count == 0,
        "receiver_training_steps_zero": receiver_training_steps == 0,
    }
    if gates != engine.get("gates") or not all(gates.values()):
        raise R14Error("R21 fresh gate recomputation changed")

    factor_rows = [
        row
        for row in live_rows
        if row["method"] == "abi_factorized" and row["seed"] == headline_seed
    ]
    comparisons = {}
    for other in ("raw_sequence", "labeled_monolith"):
        other_rows = [
            row for row in live_rows if row["method"] == other and row["seed"] == headline_seed
        ]
        comparisons[f"abi_minus_{other}"] = bootstrap_difference(
            [float(row["functional_pass"]) for row in factor_rows],
            [float(row["functional_pass"]) for row in other_rows],
            seed=21_000 + headline_seed + len(other),
            samples=int(base["gates"]["bootstrap_samples"]),
        )
    comparisons["abi_minus_teacher"] = bootstrap_difference(
        [float(row["functional_pass"]) for row in factor_rows],
        [float(row["functional_pass"]) for row in teacher_scores],
        seed=42_000 + headline_seed,
        samples=int(base["gates"]["bootstrap_samples"]),
    )
    if comparisons != engine.get("comparisons"):
        raise R14Error("R21 fresh paired comparisons changed")

    live_path = output / "live_observations.jsonl"
    cpu_path = output / "cpu_observations.jsonl"
    lifecycle_path = output / "lifecycle_observations.jsonl"
    write_jsonl_once(live_path, live_rows)
    write_jsonl_once(cpu_path, cpu_rows)
    write_jsonl_once(lifecycle_path, lifecycle)
    receipt = {
        "format": "abi-r21-fresh-live-verification/1",
        "status": "PASS_FRESH_LIVE_PUBLIC_PREREQUISITE",
        "config_sha256": sha256_file(config_path),
        "candidate_engine_sha256": live["candidate_engine"]["sha256"],
        "candidate_observations_sha256": live["candidate_observations"]["sha256"],
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
        "corrupt_packages_rejected": sum(row["corrupt_rejected"] for row in lifecycle),
        "source_software_loaded": "transformers" in sys.modules,
        "recomputed_gates": gates,
        "recomputed_aggregates": recomputed_aggregates,
        "recomputed_comparisons": comparisons,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_claim": "R21_BOUNDED_PUBLIC_LABEL_SEPARATED_GENERATIVE_TRANSFER",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "full_abi_moonshot": "OPEN",
        "next_action": "preregister fresh hidden replication",
    }
    receipt["evidence_sha256"] = selfless_evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
