"""Fail-closed independent recomputation and live replay for R41."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from experiments.english_substrate_r30.protocol import TASKS
from experiments.english_sufficiency_r31.cascade_v3 import _infer_task, _runtime_score
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    sha256_file,
    write_json_once,
)
from experiments.layercake_package_integration_r41 import run_v1 as campaign


class VerificationError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required JSON missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"required JSON unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"required JSON is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VerificationError(f"required JSONL missing: {path}")
    try:
        values = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"required JSONL unreadable: {path}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise VerificationError(f"required JSONL contains non-object: {path}")
    return values


def _selfless_hash(result: dict[str, Any]) -> str:
    copy = dict(result)
    stored = copy.pop("evidence_sha256", None)
    if not isinstance(stored, str) or evidence_hash(copy) != stored:
        raise VerificationError("R41 result evidence hash changed")
    return stored


def _live_evaluate(
    api: dict[str, Any],
    packages: dict[str, Path],
    fixture: list[dict[str, Any]],
    public: bytes,
    signer: str,
    registry: Path,
    device: str,
) -> list[dict[str, Any]]:
    host = campaign._host(api, registry, public, signer, device)
    for task in TASKS:
        host.install(packages[task])
        host.installer.verify(f"abi-r41-{task}")
    rows = []
    for row in fixture:
        task = row["oracle_task"]
        routed = _infer_task(row["prompt"])
        generated = host.generate(
            f"abi-r41-{routed}", row["prompt"], maximum_actions=96
        ).output.decode("utf-8", errors="strict")
        inferred, score = _runtime_score(row["prompt"], generated)
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
                "representation_error": None,
                "score": score,
            }
        )
    return rows


def verify(
    layercake_root: Path,
    r36: Path,
    r39: Path,
    r40: Path,
    run_dir: Path,
    scratch: Path,
) -> dict[str, Any]:
    for path, digest in (
        (r36, campaign.EXPECTED_R36),
        (r39, campaign.EXPECTED_R39),
        (r40, campaign.EXPECTED_R40),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise VerificationError(f"R41 prerequisite missing or changed: {path}")
    if not torch.cuda.is_available():
        raise VerificationError("R41 strict replay requires CUDA")
    result = _json(run_dir / "result.json")
    _selfless_hash(result)
    if (
        result.get("format") != "abi-r41-layercake-package-integration/1"
        or result.get("full_abi_moonshot") != "OPEN"
        or result.get("claim_ceiling")
        != "TWELVE_BOUNDED_TEACHER_DERIVED_NEURAL_COMPONENTS_CROSS_REAL_LAYERCAKE_PACKAGE_BOUNDARY_NOT_UNRESTRICTED_ENGLISH"
    ):
        raise VerificationError("R41 result scope changed")
    expected_artifacts = {
        "evaluation": "evaluation.jsonl",
        "inventory": "package_inventory.json",
        "lifecycle": "lifecycle_controls.jsonl",
        "corruption": "corruption_controls.jsonl",
        "public_key": "research_public_key.pem",
    }
    if set(result.get("artifacts", {})) != set(expected_artifacts):
        raise VerificationError("R41 artifact inventory changed")
    for key, filename in expected_artifacts.items():
        reference = result["artifacts"][key]
        path = run_dir / filename
        if (
            reference.get("path") != filename
            or not path.is_file()
            or sha256_file(path) != reference.get("sha256")
        ):
            raise VerificationError(f"R41 {key} artifact changed")

    api = campaign._layercake(layercake_root)
    components, expected_components, _ = campaign._components(r36, r39)
    if result.get("inputs", {}).get("component_hashes") != expected_components:
        raise VerificationError("R41 component inventory changed")
    _, public, signer = campaign._keypair(api)
    if (run_dir / "research_public_key.pem").read_bytes() != public:
        raise VerificationError("R41 public key changed")

    inventory_document = _json(run_dir / "package_inventory.json")
    inventory = inventory_document.get("packages")
    if not isinstance(inventory, list) or len(inventory) != 12:
        raise VerificationError("R41 package inventory cardinality changed")
    packages: dict[str, Path] = {}
    for item in inventory:
        task = item.get("task")
        if task not in TASKS or task in packages:
            raise VerificationError("R41 package task inventory changed")
        path = run_dir / str(item.get("package"))
        if not path.is_file() or sha256_file(path) != item.get("package_sha256"):
            raise VerificationError(f"R41 {task} package bytes changed")
        package = api["load_package"](path, trust_store={signer: public})
        source = load_file(components[task]["state"])
        if (
            package.manifest.cake_id != f"abi-r41-{task}"
            or package.manifest.abi_version != campaign.ABI_VERSION
            or package.manifest.abi_hash != campaign.ABI_HASH
            or package.manifest.training_data_provenance.get("receiver_training_steps") != 0
            or package.manifest.training_data_provenance.get("teacher_absent_at_inference")
            is not True
            or item.get("signature_key_id") != signer
            or item.get("package_hash") != package.manifest.package_hash
            or item.get("payload_hash") != package.manifest.tensor_payload_hash
            or item.get("source_state_sha256") != expected_components[task]["state"]
            or item.get("source_tokenizer_sha256") != expected_components[task]["tokenizer"]
            or set(source) != set(package.tensors)
            or not all(torch.equal(source[name], package.tensors[name]) for name in source)
            or item.get("parameters") != sum(value.numel() for value in source.values())
        ):
            raise VerificationError(f"R41 {task} package/tensor identity failed")
        packages[task] = path

    r40_document = _json(r40)
    fixture_path = r40.parent / r40_document["artifacts"]["fixture"]["path"]
    frozen_path = r40.parent / r40_document["artifacts"]["evaluation"]["path"]
    if (
        sha256_file(fixture_path) != r40_document["artifacts"]["fixture"]["sha256"]
        or sha256_file(frozen_path) != r40_document["artifacts"]["evaluation"]["sha256"]
    ):
        raise VerificationError("R41 frozen R40 rows changed")
    fixture = _jsonl(fixture_path)
    frozen = {row["record_id"]: row for row in _jsonl(frozen_path)}
    stored = _jsonl(run_dir / "evaluation.jsonl")
    if len(fixture) != 144 or len(frozen) != 144 or len(stored) != 288:
        raise VerificationError("R41 evaluation cardinality changed")
    stored_by_key = {(row.get("device"), row.get("record_id")): row for row in stored}
    if len(stored_by_key) != 288:
        raise VerificationError("R41 evaluation keys are duplicated")
    for device in ("cpu", "cuda"):
        for row in fixture:
            observed = stored_by_key.get((device, row["record_id"]))
            output = observed.get("output") if observed else None
            if not isinstance(output, str):
                raise VerificationError("R41 output is absent")
            inferred, score = _runtime_score(row["prompt"], output)
            if (
                observed.get("oracle_task") != row["oracle_task"]
                or observed.get("routed_task") != _infer_task(row["prompt"])
                or observed.get("route_exact") is not True
                or observed.get("inferred_task") != inferred
                or observed.get("contract_exact") is not True
                or observed.get("score") != score
                or observed.get("representation_error") is not None
                or observed.get("output_matches_r40") is not True
                or output != frozen[row["record_id"]]["output"]
                or observed.get("frozen_r40_output_sha256")
                != hashlib.sha256(frozen[row["record_id"]]["output"].encode()).hexdigest()
            ):
                raise VerificationError(f"R41 stored row changed: {device}/{row['record_id']}")

    lifecycle = _jsonl(run_dir / "lifecycle_controls.jsonl")
    corruption = _jsonl(run_dir / "corruption_controls.jsonl")
    if len(lifecycle) != 12 or len(corruption) != 12:
        raise VerificationError("R41 control cardinality changed")
    if any(
        row.get("task") not in TASKS
        or row.get("removed_generation_rejected") is not True
        or row.get("restored_exact") is not True
        for row in lifecycle
    ):
        raise VerificationError("R41 stored lifecycle control failed")
    for row in corruption:
        corrupt = run_dir / str(row.get("corrupt_package"))
        if (
            row.get("task") not in TASKS
            or row.get("strict_install_rejected") is not True
            or not corrupt.is_file()
            or sha256_file(corrupt) != row.get("corrupt_sha256")
        ):
            raise VerificationError("R41 stored corruption control failed")
        try:
            api["load_package"](corrupt, trust_store={signer: public})
        except Exception:
            pass
        else:
            raise VerificationError("R41 corrupt package passed live verification")

    if scratch.exists():
        raise VerificationError(f"R41 verifier scratch already exists: {scratch}")
    scratch.mkdir(parents=True)
    cpu_live = _live_evaluate(api, packages, fixture, public, signer, scratch / "cpu", "cpu")
    cuda_live = _live_evaluate(api, packages, fixture, public, signer, scratch / "cuda", "cuda")
    for live in (cpu_live, cuda_live):
        for row in live:
            stored_row = stored_by_key[(row["device"], row["record_id"])]
            for key in (
                "oracle_task",
                "routed_task",
                "route_exact",
                "inferred_task",
                "contract_exact",
                "output",
                "representation_error",
                "score",
            ):
                if row[key] != stored_row[key]:
                    raise VerificationError(
                        f"R41 live replay changed: {row['device']}/{row['record_id']}"
                    )

    lifecycle_host = campaign._host(api, scratch / "lifecycle", public, signer, "cpu")
    for task in TASKS:
        lifecycle_host.install(packages[task])
    for task in TASKS:
        cake_id = f"abi-r41-{task}"
        row = next(value for value in fixture if value["oracle_task"] == task)
        lifecycle_host.remove(cake_id)
        try:
            lifecycle_host.generate(cake_id, row["prompt"])
        except KeyError:
            pass
        else:
            raise VerificationError(f"R41 {task} removal did not fail closed")
        lifecycle_host.install(packages[task])
        if (
            lifecycle_host.generate(cake_id, row["prompt"]).output.decode()
            != frozen[row["record_id"]]["output"]
        ):
            raise VerificationError(f"R41 {task} reinstall did not restore output")

    sparse = campaign._host(api, scratch / "sparse", public, signer, "cpu")
    for task in TASKS:
        sparse.install(packages[task])
    row = fixture[0]
    sparse.generate(f"abi-r41-{row['oracle_task']}", row["prompt"])
    telemetry = sparse.telemetry()
    if (
        len(sparse.installed_ids()) != 12
        or sum(value["module_load_calls"] for value in telemetry.values()) != 1
        or telemetry[f"abi-r41-{row['oracle_task']}"]["module_load_calls"] != 1
    ):
        raise VerificationError("R41 live sparse execution failed")

    metrics = {
        "rows": len(stored),
        "cpu_functional": sum(
            row["score"]["functional"] for row in stored if row["device"] == "cpu"
        ),
        "cuda_functional": sum(
            row["score"]["functional"] for row in stored if row["device"] == "cuda"
        ),
        "route_exact": sum(row["route_exact"] for row in stored),
        "contract_exact": sum(row["contract_exact"] for row in stored),
        "representation_errors": sum(row["representation_error"] is not None for row in stored),
        "matches_r40": sum(row["output_matches_r40"] for row in stored),
        "cpu_cuda_exact": sum(
            stored_by_key[("cpu", row["record_id"])]["output"]
            == stored_by_key[("cuda", row["record_id"])]["output"]
            for row in fixture
        ),
        "tensor_identity": len(inventory),
        "signed_packages": len(inventory),
        "remove_rejections": sum(row["removed_generation_rejected"] for row in lifecycle),
        "restored_exact": sum(row["restored_exact"] for row in lifecycle),
        "corrupt_rejections": len(corruption),
    }
    expected_metrics = {
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
    if metrics != expected_metrics or result.get("metrics") != metrics:
        raise VerificationError("R41 recomputed aggregate gate failed")
    accounting = result.get("information_accounting", {})
    if (
        accounting
        != {
            "teacher_loaded": False,
            "teacher_weights_loaded": 0,
            "teacher_logits_loaded": 0,
            "teacher_activations_loaded": 0,
            "source_corpus_loaded": False,
            "training_rows_loaded": 0,
            "receiver_training_steps": 0,
            "abi_local_generation_calls": 0,
            "layercake_host_generation_calls": 301,
        }
        or "transformers" in sys.modules
    ):
        raise VerificationError("R41 information-boundary gate failed")

    return {
        "format": "abi-r41-strict-verification/1",
        "verdict": "PASS_R41_STRICT_VERIFICATION",
        "result_sha256": sha256_file(run_dir / "result.json"),
        "evidence_sha256": result["evidence_sha256"],
        "packages_verified": 12,
        "stored_rows_recomputed": 288,
        "live_rows_replayed": 288,
        "live_lifecycle_controls": 12,
        "live_corrupt_rejections": 12,
        "live_sparse_module_loads": 1,
        "full_abi_moonshot": "OPEN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--r36-result", type=Path, required=True)
    parser.add_argument("--r39-result", type=Path, required=True)
    parser.add_argument("--r40-result", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(
        args.layercake_root.resolve(),
        args.r36_result.resolve(),
        args.r39_result.resolve(),
        args.r40_result.resolve(),
        args.run_dir.resolve(),
        args.scratch.resolve(),
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
