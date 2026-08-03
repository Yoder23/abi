"""Verify the real sealed LayerCake host as an immutable external control.

This harness imports the native runtime from the separate ``layercake_release``
checkout.  It never copies or edits LayerCake code and never treats this host
control as evidence for an ABI-derived candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .failure_attribution import AttributionError, verify_contract


EVIDENCE_FORMAT = "abi-layercake-external-host-control/1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttributionError(f"invalid external-host JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AttributionError(f"external-host JSON must be an object: {path}")
    return value


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AttributionError(f"external-host path escapes LayerCake: {relative}") from exc
    return path


def _artifact_hashes(artifact: Path) -> dict[str, str]:
    names = ("metadata.json", "model-int8.onnx", "tokenizer.json")
    result = {}
    for name in names:
        path = artifact / name
        if not path.is_file():
            raise AttributionError(f"LayerCake runtime artifact is missing {name}")
        result[name] = _sha256_file(path)
    return result


def verify_external_host_evidence(
    *,
    evidence_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Verify a completed host-control record without rerunning inference."""

    evidence = _read(Path(evidence_path).resolve())
    contract = _read(Path(contract_path).resolve())
    claimed = evidence.pop("evidence_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha(evidence):
        raise AttributionError("external host evidence claim hash mismatch")
    control = contract["sealed_layercake_control"]
    if (
        evidence.get("format") != EVIDENCE_FORMAT
        or evidence.get("status") != "PASS"
        or evidence.get("claim_scope")
        != "EXACT_LAYERCAKE_NATIVE_HOST_CONTROL_ONLY_NOT_ABI_TRANSFER"
        or evidence.get("promotion_eligible") is not False
        or evidence.get("abi_candidate_metrics_inherited") is not False
        or evidence.get("moonshot_complete") is not False
    ):
        raise AttributionError("external host evidence overclaims its scope")
    checks = evidence.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise AttributionError("external host evidence contains a failed check")
    native = evidence.get("native_control", {})
    if (
        native.get("checkpoint_sha256") != control["primary_checkpoint_sha256"]
        or native.get("runtime_graph_sha256")
        != control["native_runtime_graph_sha256"]
        or native.get("architecture_id") != control["architecture_id"]
        or native.get("architecture_hash") != control["architecture_hash"]
    ):
        raise AttributionError("external host evidence has the wrong control lineage")
    smoke = evidence.get("autonomous_neural_smoke")
    if not isinstance(smoke, list) or len(smoke) < 2:
        raise AttributionError("external host evidence lacks autonomous smoke rows")
    if any(
        not row.get("cache_lengths")
        or len(set(row["cache_lengths"])) != 1
        or row.get("generated_tokens", 0) < 1
        for row in smoke
    ):
        raise AttributionError("external host smoke has invalid incremental state")
    return {
        "status": "PASS",
        "evidence_sha256": claimed,
        "checkpoint_sha256": native["checkpoint_sha256"],
        "runtime_graph_sha256": native["runtime_graph_sha256"],
    }


def _import_external_runtime(layercake_root: Path):
    existing = sys.modules.get("layercake")
    if existing is not None:
        origin = Path(str(getattr(existing, "__file__", ""))).resolve()
        try:
            origin.relative_to(layercake_root)
        except ValueError as exc:
            raise AttributionError(
                f"a non-control LayerCake package is already imported: {origin}"
            ) from exc
    sys.path.insert(0, str(layercake_root))
    module = importlib.import_module(
        "layercake.runtime.native.shallow_sparse_onnx"
    )
    origin = Path(module.__file__).resolve()
    try:
        origin.relative_to(layercake_root)
    except ValueError as exc:
        raise AttributionError("native runtime was not imported from the control repo") from exc
    return module


def run_external_host_control(
    *,
    contract_path: str | Path,
    layercake_root: str | Path,
    output_path: str | Path,
    threads: int = 1,
) -> dict[str, Any]:
    """Run identity, ABI, sparse-execution, state, and smoke controls."""

    contract_path = Path(contract_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise AttributionError(f"external host evidence is immutable: {output_path}")
    if threads < 1:
        raise AttributionError("external host threads must be positive")

    verification = verify_contract(
        contract_path,
        layercake_root=layercake_root,
    )
    contract = _read(contract_path)
    control = contract["sealed_layercake_control"]
    expected_commit = control["repository_commit"]
    commit_before = _git(layercake_root, "rev-parse", "HEAD").strip()
    porcelain_before = _git(layercake_root, "status", "--porcelain")
    if commit_before != expected_commit or porcelain_before:
        raise AttributionError("LayerCake external control is not the clean sealed checkout")

    artifact = _within(
        layercake_root,
        control["native_runtime_artifact"]["path"],
    )
    hashes_before = _artifact_hashes(artifact)
    expected_hashes = {
        "metadata.json": control["native_runtime_artifact"]["metadata"]["sha256"],
        "model-int8.onnx": control["native_runtime_artifact"]["graph"]["sha256"],
        "tokenizer.json": control["native_runtime_artifact"]["tokenizer"]["sha256"],
    }
    if hashes_before != expected_hashes:
        raise AttributionError("LayerCake native artifact differs from its control hashes")

    source_paths = (
        "layercake/__init__.py",
        "layercake/runtime/native/shallow_sparse_onnx.py",
    )
    source_hashes = {
        relative: _sha256_file(_within(layercake_root, relative))
        for relative in source_paths
    }
    runtime_module = _import_external_runtime(layercake_root)
    runtime = runtime_module.NativeRuntime(artifact, threads=threads)
    session_inputs = {
        value.name: [str(dimension) for dimension in value.shape]
        for value in runtime.session.get_inputs()
    }
    session_outputs = {
        value.name: [str(dimension) for dimension in value.shape]
        for value in runtime.session.get_outputs()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="layercake-external-control-",
        dir=output_path.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        physical = runtime_module.verify_physical_graph(
            artifact,
            temporary_root / "physical.json",
        )
        canonical = runtime_module.verify_canonical_abi(
            artifact,
            temporary_root / "canonical-abi.json",
        )

    prompts = (
        "Explain why a careful summary should preserve the supplied meaning.",
        "Rewrite this politely: send the revised note by noon.",
    )
    smoke = []
    for prompt in prompts:
        generated = runtime_module._generate(
            runtime,
            prompt,
            output_bytes=32,
        )
        payload = generated.pop("payload")
        generated_ids = generated.pop("generated_ids")
        smoke.append(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "output_hex": payload.hex(),
                "authoritative_generated_token_ids_sha256": _canonical_sha(
                    generated_ids
                ),
                **generated,
            }
        )

    hashes_after = _artifact_hashes(artifact)
    commit_after = _git(layercake_root, "rev-parse", "HEAD").strip()
    porcelain_after = _git(layercake_root, "status", "--porcelain")
    checks = {
        "attribution_contract_verified": verification["status"] == "PASS",
        "separate_repository_commit_exact": commit_before == commit_after == expected_commit,
        "separate_repository_clean_before_and_after": not porcelain_before and not porcelain_after,
        "runtime_artifact_immutable": hashes_before == hashes_after == expected_hashes,
        "runtime_module_from_sealed_repository": Path(runtime_module.__file__).resolve().is_relative_to(
            layercake_root
        ),
        "native_runtime_graph_exact": hashes_after["model-int8.onnx"]
        == control["native_runtime_graph_sha256"],
        "source_checkpoint_exact": runtime.metadata.get("source_checkpoint_sha256")
        == control["primary_checkpoint_sha256"],
        "physical_sparse_execution": physical.get("status") == "PASS",
        "canonical_abi": canonical.get("status") == "PASS",
        "persistent_incremental_state": all(
            len(set(row["cache_lengths"])) == 1
            and row["cache_lengths"][0] >= row["prompt_tokens"]
            for row in smoke
        ),
        "no_pytorch_or_transformers_runtime_loaded": "torch" not in sys.modules
        and "transformers" not in sys.modules,
    }
    result = {
        "format": EVIDENCE_FORMAT,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim_scope": "EXACT_LAYERCAKE_NATIVE_HOST_CONTROL_ONLY_NOT_ABI_TRANSFER",
        "promotion_eligible": False,
        "layercake_repository": {
            "path": str(layercake_root),
            "commit": commit_after,
            "clean": not bool(porcelain_after),
        },
        "native_control": {
            "architecture_id": control["architecture_id"],
            "architecture_hash": control["architecture_hash"],
            "checkpoint_sha256": control["primary_checkpoint_sha256"],
            "runtime_graph_sha256": control["native_runtime_graph_sha256"],
            "artifact_path": str(artifact),
            "artifact_hashes": hashes_after,
        },
        "source_hashes": source_hashes,
        "runtime_interface": {
            "inputs": session_inputs,
            "outputs": session_outputs,
            "provider": runtime.session.get_providers(),
            "threads": threads,
        },
        "physical_sparse_proof": physical,
        "canonical_abi_proof": canonical,
        "autonomous_neural_smoke": smoke,
        "checks": checks,
        "abi_candidate_metrics_inherited": False,
        "moonshot_complete": False,
    }
    result["evidence_sha256"] = _canonical_sha(result)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_external_host_control(
        contract_path=args.contract,
        layercake_root=args.layercake_root,
        output_path=args.output,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
