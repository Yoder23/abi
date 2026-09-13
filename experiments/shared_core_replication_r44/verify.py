"""Fail-closed raw recomputation and fresh candidate replay for R44."""

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
from experiments.layercake_package_integration_r41 import run_v1 as r41
from experiments.role_tagged_shared_core_r43.run_v1 import _role_prompt
from experiments.shared_core_replication_r44 import run_v1 as campaign


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
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"required JSONL unreadable: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise VerificationError("R44 JSONL contains a non-object")
    return rows


def _live_candidate(
    api: dict[str, Any],
    package: Path,
    fixture: list[dict[str, Any]],
    public: bytes,
    signer: str,
    registry: Path,
    device: str,
) -> list[dict[str, Any]]:
    host = r41._host(api, registry, public, signer, device)
    host.install(package)
    host.installer.verify("abi-r42-english-core")
    rows = []
    for row in fixture:
        routed = _infer_task(row["prompt"])
        canonical = _role_prompt(routed, row["prompt"])
        output = host.generate(
            "abi-r42-english-core", canonical, maximum_actions=campaign.MAXIMUM_ACTIONS
        ).output.decode("utf-8", errors="strict")
        inferred, score = _runtime_score(row["prompt"], output)
        rows.append(
            {
                "device": device,
                "record_id": row["record_id"],
                "oracle_task": row["oracle_task"],
                "routed_task": routed,
                "route_exact": routed == row["oracle_task"],
                "inferred_task": inferred,
                "contract_exact": inferred == row["oracle_task"],
                "canonical_prompt_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "output": output,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "representation_error": None,
                "score": score,
            }
        )
    return rows


def verify(
    layercake_root: Path,
    run_dir: Path,
    r43_result: Path,
    candidate: Path,
    zero: Path,
    state: Path,
    public_key: Path,
    scratch: Path,
) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    result = _json(result_path)
    copy = dict(result)
    stored_evidence = copy.pop("evidence_sha256", None)
    if not isinstance(stored_evidence, str) or evidence_hash(copy) != stored_evidence:
        raise VerificationError("R44 evidence digest changed")
    if (
        result.get("format") != "abi-r44-frozen-shared-core-replication/1"
        or result.get("claim_ceiling")
        != "PROSPECTIVE_REPLICATED_COMPACT_SUPPLIED_CONTENT_ENGLISH_CORE_NOT_UNRESTRICTED_ENGLISH"
        or result.get("full_abi_moonshot") != "OPEN"
    ):
        raise VerificationError("R44 result scope changed")
    for path, digest in (
        (r43_result, campaign.EXPECTED_R43),
        (candidate, campaign.EXPECTED_CANDIDATE),
        (zero, campaign.EXPECTED_ZERO),
        (state, campaign.EXPECTED_STATE),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise VerificationError(f"R44 frozen input changed: {path}")
    expected_artifacts = {
        "fixture": "fixture.jsonl",
        "evaluation": "evaluation.jsonl",
        "corrupted_package": "corrupted_candidate.cake",
    }
    if set(result.get("artifacts", {})) != set(expected_artifacts):
        raise VerificationError("R44 artifact inventory changed")
    for name, filename in expected_artifacts.items():
        reference = result["artifacts"][name]
        path = run_dir / filename
        if (
            reference.get("path") != filename
            or not path.is_file()
            or sha256_file(path) != reference.get("sha256")
        ):
            raise VerificationError(f"R44 {name} artifact changed")

    fixture = _jsonl(run_dir / "fixture.jsonl")
    if fixture != campaign.fixture():
        raise VerificationError("R44 prospective fixture changed")
    observations = _jsonl(run_dir / "evaluation.jsonl")
    if len(observations) != 312:
        raise VerificationError("R44 observation cardinality changed")
    candidate_rows = [row for row in observations if row.get("condition") == "candidate"]
    zero_rows = [row for row in observations if row.get("condition") == "zero_state"]
    if len(candidate_rows) != 288 or len(zero_rows) != 24:
        raise VerificationError("R44 condition cardinality changed")
    by_key = {(row.get("device"), row.get("record_id")): row for row in candidate_rows}
    if len(by_key) != 288:
        raise VerificationError("R44 candidate keys are duplicated")
    for device in ("cpu", "cuda"):
        for source in fixture:
            row = by_key.get((device, source["record_id"]))
            if row is None:
                raise VerificationError("R44 candidate row missing")
            output = row.get("output")
            if not isinstance(output, str):
                raise VerificationError("R44 output missing")
            routed = _infer_task(source["prompt"])
            canonical = _role_prompt(routed, source["prompt"])
            inferred, score = _runtime_score(source["prompt"], output)
            expected = {
                "device": device,
                "condition": "candidate",
                "record_id": source["record_id"],
                "oracle_task": source["oracle_task"],
                "routed_task": routed,
                "route_exact": routed == source["oracle_task"],
                "inferred_task": inferred,
                "contract_exact": inferred == source["oracle_task"],
                "canonical_prompt_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "output": output,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "representation_error": None,
                "score": score,
            }
            if row != expected:
                raise VerificationError(
                    f"R44 candidate row changed: {device}/{source['record_id']}"
                )
    zero_keys = {(row.get("device"), row.get("oracle_task")): row for row in zero_rows}
    if len(zero_keys) != 24:
        raise VerificationError("R44 zero-state keys are duplicated")
    for device in ("cpu", "cuda"):
        for task in TASKS:
            row = zero_keys.get((device, task))
            source = next(value for value in fixture if value["oracle_task"] == task)
            if row is None or not isinstance(row.get("output"), str):
                raise VerificationError("R44 zero-state row missing")
            _, score = _runtime_score(source["prompt"], row["output"])
            if (
                row.get("record_id") != source["record_id"]
                or row.get("score") != score
                or row.get("output_sha256") != hashlib.sha256(row["output"].encode()).hexdigest()
                or row.get("candidate_exact")
                != (row["output"] == by_key[(device, source["record_id"])]["output"])
            ):
                raise VerificationError("R44 zero-state row changed")

    api = r41._layercake(layercake_root)
    public = public_key.read_bytes()
    _, expected_public, signer = r41._keypair(api)
    if public != expected_public:
        raise VerificationError("R44 public key changed")
    loaded = api["load_package"](candidate, trust_store={signer: public})
    loaded_zero = api["load_package"](zero, trust_store={signer: public})
    source_state = load_file(state)
    if (
        set(source_state) != set(loaded.tensors)
        or not all(torch.equal(source_state[name], loaded.tensors[name]) for name in source_state)
        or not all(torch.count_nonzero(value).item() == 0 for value in loaded_zero.tensors.values())
        or sum(value.numel() for value in source_state.values()) != 296_554
    ):
        raise VerificationError("R44 package tensor proof failed")
    corrupt = run_dir / "corrupted_candidate.cake"
    try:
        api["load_package"](corrupt, trust_store={signer: public})
    except Exception:
        pass
    else:
        raise VerificationError("R44 corrupt package passed live verification")

    by_task = {
        device: {
            task: sum(
                row["score"]["functional"]
                for row in candidate_rows
                if row["device"] == device and row["oracle_task"] == task
            )
            for task in TASKS
        }
        for device in ("cpu", "cuda")
    }
    metrics = {
        "fixture_rows": len(fixture),
        "candidate_cpu_functional": sum(
            row["score"]["functional"] for row in candidate_rows if row["device"] == "cpu"
        ),
        "candidate_cuda_functional": sum(
            row["score"]["functional"] for row in candidate_rows if row["device"] == "cuda"
        ),
        "candidate_by_device_task": by_task,
        "route_exact": sum(row["route_exact"] for row in candidate_rows),
        "contract_exact": sum(row["contract_exact"] for row in candidate_rows),
        "representation_errors": sum(
            row["representation_error"] is not None for row in candidate_rows
        ),
        "noncollapsed": sum(
            bool(row["output"]) and row["score"]["noncollapsed"] for row in candidate_rows
        ),
        "cpu_cuda_exact": sum(
            by_key[("cpu", source["record_id"])]["output"]
            == by_key[("cuda", source["record_id"])]["output"]
            for source in fixture
        ),
        "zero_functional": sum(row["score"]["functional"] for row in zero_rows),
        "zero_candidate_exact": sum(row["candidate_exact"] for row in zero_rows),
        "active_parameters": sum(value.numel() for value in source_state.values()),
    }
    if result.get("metrics") != metrics or (
        metrics["candidate_cpu_functional"] < 132
        or metrics["candidate_cuda_functional"] < 132
        or any(value < 10 for device in by_task.values() for value in device.values())
        or by_task["cpu"]["abstention"] != 12
        or by_task["cuda"]["abstention"] != 12
        or metrics["route_exact"] != 288
        or metrics["contract_exact"] != 288
        or metrics["representation_errors"] != 0
        or metrics["noncollapsed"] != 288
        or metrics["cpu_cuda_exact"] != 144
        or metrics["zero_functional"] != 0
        or metrics["zero_candidate_exact"] != 0
    ):
        raise VerificationError("R44 recomputed gate failed")

    if scratch.exists():
        raise VerificationError(f"R44 verifier scratch already exists: {scratch}")
    scratch.mkdir(parents=True)
    live = []
    for device in ("cpu", "cuda"):
        live.extend(
            _live_candidate(api, candidate, fixture, public, signer, scratch / device, device)
        )
    for row in live:
        stored = by_key[(row["device"], row["record_id"])]
        for key in row:
            if row[key] != stored[key]:
                raise VerificationError(
                    f"R44 live replay changed: {row['device']}/{row['record_id']}"
                )

    lifecycle_host = r41._host(api, scratch / "lifecycle", public, signer, "cpu")
    lifecycle_host.install(candidate)
    first = fixture[0]
    first_prompt = _role_prompt(first["oracle_task"], first["prompt"])
    lifecycle_host.remove("abi-r42-english-core")
    try:
        lifecycle_host.generate("abi-r42-english-core", first_prompt)
    except KeyError:
        pass
    else:
        raise VerificationError("R44 live removal did not fail closed")
    lifecycle_host.install(candidate)
    if (
        lifecycle_host.generate("abi-r42-english-core", first_prompt).output.decode()
        != by_key[("cpu", first["record_id"])]["output"]
    ):
        raise VerificationError("R44 live reinstall did not restore output")
    if "transformers" in sys.modules:
        raise VerificationError("R44 verifier loaded a source-model runtime")
    return {
        "format": "abi-r44-strict-verification/1",
        "verdict": "PASS_R44_STRICT_VERIFICATION",
        "result_sha256": sha256_file(result_path),
        "evidence_sha256": stored_evidence,
        "raw_rows_recomputed": 312,
        "candidate_rows_live_replayed": 288,
        "packages_verified": 2,
        "corrupt_packages_rejected": 1,
        "live_lifecycle_passed": True,
        "full_abi_moonshot": "OPEN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--r43-result", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(
        args.layercake_root.resolve(),
        args.run_dir.resolve(),
        args.r43_result.resolve(),
        args.candidate.resolve(),
        args.zero.resolve(),
        args.state.resolve(),
        args.public_key.resolve(),
        args.scratch.resolve(),
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
