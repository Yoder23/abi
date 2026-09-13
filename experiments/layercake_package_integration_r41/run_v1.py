"""Build and execute frozen R36/R39 components as real LayerCake cakes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from safetensors.torch import load_file

from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31.cascade_v3 import (
    _infer_task,
    _runtime_score,
)
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)

EXPECTED_LAYERCAKE_COMMIT = "f8a4569"
EXPECTED_R36 = "1830a0a1be9b520376ce228fd50514e79f9b3a8224d56f7a1b36e0338d03d5bd"
EXPECTED_R39 = "eee143b1f35038ed518cbd4a9a8aa034da03e0799fea39a9498e76b7bed242f0"
EXPECTED_R40 = "50c6a5aabb8e416ec7e959919591246eca1193b36c0c6b707ad0c217f7f324d6"
ABI_VERSION = "lc-field-addressed-neural-plan/1"
ABI_HASH = hashlib.sha256(ABI_VERSION.encode()).hexdigest()
RESEARCH_KEY_SEED = hashlib.sha256(
    b"ABI R41 deterministic non-production research package key v1"
).digest()


def _layercake(root: Path) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if not head.startswith(EXPECTED_LAYERCAKE_COMMIT):
        raise RuntimeError(f"R41 LayerCake commit changed: {head}")
    sys.path.insert(0, str(root))
    from layercake.cake.manifest import CakeManifest
    from layercake.cake.package import (
        build_package,
        load_package,
        tensor_specs,
    )
    from layercake.cake.signing import key_id
    from layercake.models.direct_cake_host import DirectCakeHost
    from layercake.models.portable_decoder import (
        field_addressed_token_plan_manifest_architecture,
    )
    from layercake.portable_token_plan import FieldAddressedPointerTokenizer

    return {
        "CakeManifest": CakeManifest,
        "build_package": build_package,
        "load_package": load_package,
        "tensor_specs": tensor_specs,
        "key_id": key_id,
        "DirectCakeHost": DirectCakeHost,
        "field_addressed_token_plan_manifest_architecture": (
            field_addressed_token_plan_manifest_architecture
        ),
        "FieldAddressedPointerTokenizer": FieldAddressedPointerTokenizer,
    }


def _keypair(api: dict[str, Any]) -> tuple[bytes, bytes, str]:
    private = Ed25519PrivateKey.from_private_bytes(RESEARCH_KEY_SEED)
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem, api["key_id"](public_pem)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _components(
    r36: Path,
    r39: Path,
) -> tuple[dict[str, dict[str, Path]], dict[str, dict[str, str]], dict[str, Any]]:
    document = json.loads(r39.read_text(encoding="utf-8"))
    components: dict[str, dict[str, Path]] = {}
    expected: dict[str, dict[str, str]] = {}
    for row in document["inherited_r36_components"]:
        kind = "tokenizer" if row["name"].endswith("tokenizer.json") else "state"
        components.setdefault(row["task"], {})[kind] = r36.parent / "models" / row["name"]
        expected.setdefault(row["task"], {})[kind] = row["sha256"]
    candidate = document["candidate"]
    components["abstention"] = {
        "state": r39.parent / candidate["state"]["path"],
        "tokenizer": r39.parent / candidate["tokenizer"]["path"],
    }
    expected["abstention"] = {
        "state": candidate["state"]["sha256"],
        "tokenizer": candidate["tokenizer"]["sha256"],
    }
    if set(components) != set(TASKS) or any(
        set(value) != {"state", "tokenizer"} for value in components.values()
    ):
        raise RuntimeError("R41 component matrix is incomplete")
    for task in TASKS:
        for kind, path in components[task].items():
            if not path.is_file() or sha256_file(path) != expected[task][kind]:
                raise RuntimeError(f"R41 frozen {task} {kind} changed")
    return components, expected, document


def _model_config(tokenizer: Any) -> dict[str, Any]:
    return {
        "fixed_vocab_size": tokenizer.vocab_size,
        "model_width": 64,
        "attention_heads": 4,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "feedforward_width": 192,
        "pointer_width": 32,
        "dropout": 0.0,
        "maximum_source_lexemes": 128,
        "maximum_target_actions": 96,
    }


def _make_packages(
    api: dict[str, Any],
    components: dict[str, dict[str, Path]],
    expected: dict[str, dict[str, str]],
    r40_hash: str,
    output: Path,
    private: bytes,
    public: bytes,
    signer: str,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    package_dir = output / "packages"
    package_dir.mkdir()
    public_path = output / "research_public_key.pem"
    public_path.write_bytes(public)
    packages: dict[str, Path] = {}
    inventory: list[dict[str, Any]] = []
    for task in TASKS:
        tokenizer_document = json.loads(components[task]["tokenizer"].read_text(encoding="utf-8"))
        if tokenizer_document.get("format") != "abi-r36-field-addressed-neural-plan/1":
            raise RuntimeError(f"R41 {task} source tokenizer format changed")
        tokenizer_document = {
            **tokenizer_document,
            "format": "layercake-field-addressed-token-plan/1",
        }
        tokenizer = api["FieldAddressedPointerTokenizer"].from_document(tokenizer_document)
        state = load_file(components[task]["state"])
        before = {name: tensor.detach().cpu().clone() for name, tensor in state.items()}
        architecture = api["field_addressed_token_plan_manifest_architecture"](
            model=_model_config(tokenizer),
            tokenizer=tokenizer_document,
            tokenizer_sha256=tokenizer.hash(),
        )
        manifest = api["CakeManifest"](
            schema_version="1",
            cake_id=f"abi-r41-{task}",
            name=f"R41 {task} neural plan",
            description="Frozen teacher-derived field-addressed neural plan",
            version="1.0.0",
            publisher={"id": "abi-r41", "name": "ABI R41", "key_id": signer},
            abi_version=ABI_VERSION,
            abi_hash=ABI_HASH,
            cake_type="portable_decoder",
            input_contract={
                "external": "UTF-8 bytes",
                "mode": "direct_selected_portable_decoder",
                "selection": "explicit",
            },
            output_contract={
                "external": "UTF-8 bytes",
                "composition": "direct_selected_capability",
            },
            architecture=architecture,
            supported_precisions=("fp32",),
            supported_backends=("pytorch", "cuda"),
            minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "incremental"]},
            tensor_payload_hash="",
            tensor_shapes=api["tensor_specs"](state),
            package_hash="",
            training_data_provenance={
                "source_state_sha256": expected[task]["state"],
                "source_tokenizer_sha256": expected[task]["tokenizer"],
                "host_tokenizer_semantic_sha256": tokenizer.hash(),
                "teacher_absent_at_inference": True,
                "receiver_training_steps": 0,
            },
            evaluation_evidence={
                "r40_result_sha256": r40_hash,
                "status": "EXTERNAL_R41_GATE_REQUIRED",
            },
            license="Apache-2.0",
            dependencies=(),
            parent_version=None,
            signature={"algorithm": "ed25519", "key_id": signer},
            domains=(f"english-{task}",),
            permissions=("local-inference",),
        )
        path = api["build_package"](
            package_dir / f"{task}.cake", manifest, state, private_key=private
        )
        loaded = api["load_package"](path, trust_store={signer: public})
        equal = set(before) == set(loaded.tensors) and all(
            torch.equal(before[name], loaded.tensors[name]) for name in before
        )
        unchanged = sha256_file(components[task]["state"]) == expected[task]["state"]
        if not equal or not unchanged:
            raise RuntimeError(f"R41 {task} tensor identity failed")
        packages[task] = path
        inventory.append(
            {
                "task": task,
                "cake_id": manifest.cake_id,
                "package": path.relative_to(output).as_posix(),
                "package_sha256": sha256_file(path),
                "package_bytes": path.stat().st_size,
                "package_hash": loaded.manifest.package_hash,
                "payload_hash": loaded.manifest.tensor_payload_hash,
                "signature_key_id": loaded.signature_key_id,
                "source_state_sha256": expected[task]["state"],
                "source_tokenizer_sha256": expected[task]["tokenizer"],
                "host_tokenizer_semantic_sha256": tokenizer.hash(),
                "tensor_names": sorted(before),
                "tensor_identity": equal,
                "source_unchanged": unchanged,
                "parameters": sum(tensor.numel() for tensor in before.values()),
            }
        )
    return packages, inventory


def _host(api: dict[str, Any], root: Path, public: bytes, signer: str, device: str):
    return api["DirectCakeHost"](
        root,
        abi_version=ABI_VERSION,
        abi_hash=ABI_HASH,
        trust_store={signer: public},
        device=device,
    )


def _evaluate(
    api: dict[str, Any],
    packages: dict[str, Path],
    fixture: list[dict[str, Any]],
    frozen: dict[str, dict[str, Any]],
    public: bytes,
    signer: str,
    output: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    host = _host(api, output / f"registry_{device}", public, signer, device)
    install = [host.install(packages[task]) for task in TASKS]
    verify = [host.installer.verify(f"abi-r41-{task}") for task in TASKS]
    rows = []
    started = time.perf_counter()
    for row in fixture:
        task = row["oracle_task"]
        routed = _infer_task(row["prompt"])
        error = None
        try:
            generated = host.generate(
                f"abi-r41-{routed}", row["prompt"], maximum_actions=96
            ).output.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError) as exc:
            generated, error = "", str(exc)
        inferred, score = _runtime_score(row["prompt"], generated)
        frozen_row = frozen[row["record_id"]]
        rows.append(
            {
                "device": device,
                "record_id": row["record_id"],
                "oracle_task": task,
                "routed_task": routed,
                "route_exact": routed == task,
                "inferred_task": inferred,
                "contract_exact": inferred == task,
                "output": generated,
                "frozen_r40_output_sha256": hashlib.sha256(
                    frozen_row["output"].encode()
                ).hexdigest(),
                "output_matches_r40": generated == frozen_row["output"],
                "representation_error": error,
                "score": score,
            }
        )
    return (
        rows,
        {
            "device": device,
            "seconds": time.perf_counter() - started,
            "install_records": install,
            "verify_records": verify,
            "telemetry": host.telemetry(),
        },
        host,
    )


def run(
    layercake_root: Path,
    r36: Path,
    r39: Path,
    r40: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"immutable R41 output exists: {output}")
    for path, digest in ((r36, EXPECTED_R36), (r39, EXPECTED_R39), (r40, EXPECTED_R40)):
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"R41 prerequisite missing or changed: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("R41 requires CUDA for cross-device identity")
    api = _layercake(layercake_root)
    torch.cuda.reset_peak_memory_stats()
    components, expected, _ = _components(r36, r39)
    r40_document = json.loads(r40.read_text(encoding="utf-8"))
    fixture_path = r40.parent / r40_document["artifacts"]["fixture"]["path"]
    evaluation_path = r40.parent / r40_document["artifacts"]["evaluation"]["path"]
    if (
        sha256_file(fixture_path) != r40_document["artifacts"]["fixture"]["sha256"]
        or sha256_file(evaluation_path) != r40_document["artifacts"]["evaluation"]["sha256"]
    ):
        raise RuntimeError("R41 R40 raw evidence changed")
    fixture = _jsonl(fixture_path)
    frozen_rows = {row["record_id"]: row for row in _jsonl(evaluation_path)}
    if len(fixture) != 144 or len(frozen_rows) != 144:
        raise RuntimeError("R41 requires the full R40 matrix")
    output.mkdir(parents=True)
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    private, public, signer = _keypair(api)
    packages, inventory = _make_packages(
        api, components, expected, EXPECTED_R40, output, private, public, signer
    )

    cpu_rows, cpu_lifecycle, cpu_host = _evaluate(
        api, packages, fixture, frozen_rows, public, signer, output, "cpu"
    )
    peak_rss = max(peak_rss, process.memory_info().rss)
    cuda_rows, cuda_lifecycle, _ = _evaluate(
        api, packages, fixture, frozen_rows, public, signer, output, "cuda"
    )
    peak_rss = max(peak_rss, process.memory_info().rss)

    lifecycle_controls = []
    for task in TASKS:
        cake_id = f"abi-r41-{task}"
        first = next(row for row in fixture if row["oracle_task"] == task)
        expected_output = frozen_rows[first["record_id"]]["output"].encode()
        removed = cpu_host.remove(cake_id)
        rejected = False
        try:
            cpu_host.generate(cake_id, first["prompt"])
        except KeyError:
            rejected = True
        reinstalled = cpu_host.install(packages[task])
        restored = cpu_host.generate(cake_id, first["prompt"]).output
        lifecycle_controls.append(
            {
                "task": task,
                "removed_status": removed["status"],
                "removed_generation_rejected": rejected,
                "reinstalled_status": reinstalled["status"],
                "restored_output_sha256": hashlib.sha256(restored).hexdigest(),
                "restored_exact": restored == expected_output,
            }
        )

    corrupt_dir = output / "corrupted_packages"
    corrupt_dir.mkdir()
    corruption_controls = []
    corrupt_host = _host(api, output / "registry_corrupt", public, signer, "cpu")
    for task in TASKS:
        raw = bytearray(packages[task].read_bytes())
        raw[len(raw) // 2] ^= 1
        corrupt = corrupt_dir / f"{task}.cake"
        corrupt.write_bytes(raw)
        rejected = False
        error = None
        try:
            corrupt_host.install(corrupt)
        except Exception as exc:  # exact exception type is host-owned
            rejected, error = True, type(exc).__name__ + ": " + str(exc)
        corruption_controls.append(
            {
                "task": task,
                "corrupt_package": corrupt.relative_to(output).as_posix(),
                "corrupt_sha256": sha256_file(corrupt),
                "strict_install_rejected": rejected,
                "error": error,
            }
        )

    sparse = _host(api, output / "registry_sparse", public, signer, "cpu")
    for task in TASKS:
        sparse.install(packages[task])
    sparse_task = TASKS[0]
    sparse_row = next(row for row in fixture if row["oracle_task"] == sparse_task)
    sparse_result = sparse.generate(f"abi-r41-{sparse_task}", sparse_row["prompt"])
    sparse_telemetry = sparse.telemetry()
    sparse_control = {
        "selected_task": sparse_task,
        "installed_count": len(sparse.installed_ids()),
        "loaded_module_count": sum(
            values["module_load_calls"] for values in sparse_telemetry.values()
        ),
        "selected_module_load_calls": sparse_telemetry[f"abi-r41-{sparse_task}"][
            "module_load_calls"
        ],
        "unselected_module_load_calls": sum(
            values["module_load_calls"]
            for key, values in sparse_telemetry.items()
            if key != f"abi-r41-{sparse_task}"
        ),
        "output_matches_r40": sparse_result.output.decode()
        == frozen_rows[sparse_row["record_id"]]["output"],
        "telemetry": sparse_telemetry,
    }

    all_rows = cpu_rows + cuda_rows
    rows_path = output / "evaluation.jsonl"
    inventory_path = output / "package_inventory.json"
    lifecycle_path = output / "lifecycle_controls.jsonl"
    corruption_path = output / "corruption_controls.jsonl"
    write_jsonl_once(rows_path, all_rows)
    write_json_once(inventory_path, {"packages": inventory})
    write_jsonl_once(lifecycle_path, lifecycle_controls)
    write_jsonl_once(corruption_path, corruption_controls)
    cpu_by_id = {row["record_id"]: row for row in cpu_rows}
    cuda_by_id = {row["record_id"]: row for row in cuda_rows}
    metrics = {
        "rows": len(all_rows),
        "cpu_functional": sum(row["score"]["functional"] for row in cpu_rows),
        "cuda_functional": sum(row["score"]["functional"] for row in cuda_rows),
        "route_exact": sum(row["route_exact"] for row in all_rows),
        "contract_exact": sum(row["contract_exact"] for row in all_rows),
        "representation_errors": sum(row["representation_error"] is not None for row in all_rows),
        "matches_r40": sum(row["output_matches_r40"] for row in all_rows),
        "cpu_cuda_exact": sum(
            cpu_by_id[key]["output"] == cuda_by_id[key]["output"] for key in cpu_by_id
        ),
        "tensor_identity": sum(row["tensor_identity"] for row in inventory),
        "signed_packages": sum(row["signature_key_id"] == signer for row in inventory),
        "remove_rejections": sum(row["removed_generation_rejected"] for row in lifecycle_controls),
        "restored_exact": sum(row["restored_exact"] for row in lifecycle_controls),
        "corrupt_rejections": sum(row["strict_install_rejected"] for row in corruption_controls),
    }
    passed = (
        metrics
        == {
            "rows": 288,
            "cpu_functional": 144,
            "cuda_functional": 144,
            "route_exact": 288,
            "contract_exact": 288,
            "representation_errors": 0,
            "matches_r40": 288,
            "cpu_cuda_exact": 144,
            "tensor_identity": 12,
            "signed_packages": 12,
            "remove_rejections": 12,
            "restored_exact": 12,
            "corrupt_rejections": 12,
        }
        and sparse_control["installed_count"] == 12
        and sparse_control["loaded_module_count"] == 1
        and sparse_control["selected_module_load_calls"] == 1
        and sparse_control["unselected_module_load_calls"] == 0
        and sparse_control["output_matches_r40"]
    )
    result = {
        "format": "abi-r41-layercake-package-integration/1",
        "verdict": "PASS_R41_LAYERCAKE_INTEGRATION" if passed else "FAIL_R41_LAYERCAKE_INTEGRATION",
        "inputs": {
            "layercake_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=layercake_root, text=True
            ).strip(),
            "r36_result_sha256": EXPECTED_R36,
            "r39_result_sha256": EXPECTED_R39,
            "r40_result_sha256": EXPECTED_R40,
            "component_hashes": expected,
        },
        "metrics": metrics,
        "sparse_control": sparse_control,
        "runtime": {
            "cpu_seconds": cpu_lifecycle["seconds"],
            "cuda_seconds": cuda_lifecycle["seconds"],
            "total_seconds": time.perf_counter() - started,
            "peak_cpu_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "information_accounting": {
            "teacher_loaded": False,
            "teacher_weights_loaded": 0,
            "teacher_logits_loaded": 0,
            "teacher_activations_loaded": 0,
            "source_corpus_loaded": False,
            "training_rows_loaded": 0,
            "receiver_training_steps": 0,
            "abi_local_generation_calls": 0,
            "layercake_host_generation_calls": 301,
        },
        "artifacts": {
            "evaluation": {"path": rows_path.name, "sha256": sha256_file(rows_path)},
            "inventory": {"path": inventory_path.name, "sha256": sha256_file(inventory_path)},
            "lifecycle": {"path": lifecycle_path.name, "sha256": sha256_file(lifecycle_path)},
            "corruption": {"path": corruption_path.name, "sha256": sha256_file(corruption_path)},
            "public_key": {
                "path": "research_public_key.pem",
                "sha256": sha256_file(output / "research_public_key.pem"),
            },
        },
        "claim_ceiling": "TWELVE_BOUNDED_TEACHER_DERIVED_NEURAL_COMPONENTS_CROSS_REAL_LAYERCAKE_PACKAGE_BOUNDARY_NOT_UNRESTRICTED_ENGLISH",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--r36-result", type=Path, required=True)
    parser.add_argument("--r39-result", type=Path, required=True)
    parser.add_argument("--r40-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.layercake_root.resolve(),
        args.r36_result.resolve(),
        args.r39_result.resolve(),
        args.r40_result.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
