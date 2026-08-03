"""Run a preregistered GPU survey as resumable, verified catalog shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence
import zipfile

from .hf_extraction import load_probe_catalog
from .moonshot import verify_extraction_bundle


FORMAT = "abi-sharded-source-survey-state/1"


class ShardedSurveyError(RuntimeError):
    """Raised when a sharded survey cannot preserve an exact evidence union."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_survey_command(
    *,
    catalog: Path,
    output: Path,
    model: str,
    revision: str,
    license_id: str,
    batch_size: int,
    splits: Sequence[str] = ("search",),
    development: bool = True,
) -> list[str]:
    normalized_splits = tuple(sorted(set(splits)))
    if not normalized_splits or any(
        split not in {"search", "validation"} for split in normalized_splits
    ):
        raise ShardedSurveyError(
            "sharded surveys may use search and/or validation, never final_test"
        )
    command = [
        sys.executable,
        "-m",
        "abi.moonshot",
        "survey",
        "--model",
        model,
        "--revision",
        revision,
        "--license",
        license_id,
        "--device",
        "cuda",
        "--catalog",
        str(catalog),
        "--output",
        str(output),
        "--minimum-distinct-probes",
        "1",
        "--minimum-pass-rate",
        "0.90",
        "--minimum-wilson-lower-bound",
        "0.75",
        "--batch-size",
        str(batch_size),
        "--splits",
        ",".join(normalized_splits),
    ]
    if development:
        command.append("--development")
    return command


def _load_partition_manifest(path: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_EXACT_DISJOINT_PARTITION":
        raise ShardedSurveyError("partition manifest is not passing")
    if any(
        int(manifest.get(field, -1)) != 0
        for field in (
            "intersection_probe_count",
            "missing_probe_count",
            "extra_probe_count",
            "probe_payload_changes",
        )
    ):
        raise ShardedSurveyError("partition manifest does not bind an exact union")
    directory = path.parent
    catalogs = []
    for expected_index, receipt in enumerate(manifest.get("shards", [])):
        if int(receipt.get("index", -1)) != expected_index:
            raise ShardedSurveyError("partition shard ordering is invalid")
        catalog = directory / str(receipt["path"])
        if not catalog.is_file():
            raise ShardedSurveyError(f"partition shard is missing: {catalog}")
        if _sha256_file(catalog) != receipt["sha256"]:
            raise ShardedSurveyError(f"partition shard hash drift: {catalog}")
        loaded = load_probe_catalog(catalog)
        if len(loaded["probes"]) != int(receipt["probes"]):
            raise ShardedSurveyError(f"partition shard count drift: {catalog}")
        catalogs.append(catalog)
    if len(catalogs) != int(manifest["shard_count"]):
        raise ShardedSurveyError("partition shard count mismatch")
    return manifest, catalogs


def _archive_observations(path: Path) -> dict[str, Any]:
    verification = verify_extraction_bundle(path)
    with zipfile.ZipFile(path) as archive:
        results = json.loads(archive.read("probe_results.json"))
        ledger = json.loads(archive.read("ledger.json"))
    return {
        "archive_sha256": verification["archive_sha256"],
        "manifest_sha256": verification["manifest_sha256"],
        "bytes": path.stat().st_size,
        "probe_count": len(results),
        "passing_probe_count": sum(result["passed"] is True for result in results),
        "failing_probe_count": sum(result["passed"] is not True for result in results),
        "probe_ids": sorted(str(result["probe_id"]) for result in results),
        "teacher_tokens": int(ledger["teacher_tokens"]),
        "teacher_generated_output_bytes": int(
            ledger["teacher_generated_output_bytes"]
        ),
        "source_model_inference_seconds": float(
            ledger["source_model_inference_seconds"]
        ),
        "source_extraction_devices": ledger["source_extraction_devices"],
    }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_shards(
    *,
    partition_manifest: Path,
    output_directory: Path,
    log_directory: Path,
    state_path: Path,
    model: str,
    revision: str,
    license_id: str,
    batch_size: int,
    splits: Sequence[str] = ("search",),
    development: bool = True,
    output_prefix: str = "phi3-broad-natural-conversation-search-v3",
) -> dict[str, Any]:
    manifest, catalogs = _load_partition_manifest(partition_manifest)
    output_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "format": FORMAT,
        "status": "RUNNING",
        "partition_manifest": {
            "path": str(partition_manifest),
            "sha256": _sha256_file(partition_manifest),
            "evidence_sha256": manifest["evidence_sha256"],
        },
        "source": {
            "model": model,
            "revision": revision,
            "license": license_id,
            "device": "cuda",
            "batch_size": batch_size,
            "splits": sorted(set(splits)),
            "development": development,
            "cpu_fallback_allowed": False,
        },
        "shards": [],
        "completed_shards": 0,
        "required_shards": len(catalogs),
        "training_authorized": False,
    }
    expected_union: set[str] = set()
    observed_union: set[str] = set()
    for index, catalog in enumerate(catalogs, start=1):
        expected_ids = {
            str(probe["probe_id"])
            for probe in load_probe_catalog(catalog)["probes"]
        }
        if expected_union & expected_ids:
            raise ShardedSurveyError("preflight catalog shards overlap")
        expected_union.update(expected_ids)
        output = output_directory / (
            f"{output_prefix}-shard-{index:02d}-of-{len(catalogs):02d}.abix"
        )
        command = build_survey_command(
            catalog=catalog,
            output=output,
            model=model,
            revision=revision,
            license_id=license_id,
            batch_size=batch_size,
            splits=splits,
            development=development,
        )
        if not output.exists():
            stdout_path = log_directory / f"shard-{index:02d}.stdout.log"
            stderr_path = log_directory / f"shard-{index:02d}.stderr.log"
            if stdout_path.exists() or stderr_path.exists():
                raise ShardedSurveyError(
                    f"logs exist without archive for shard {index}; preserve "
                    "them and use a successor log directory"
                )
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if completed.returncode != 0:
                state["status"] = "FAIL_CLOSED_SHARD_SUBPROCESS"
                state["failed_shard"] = index
                state["returncode"] = completed.returncode
                state["training_authorized"] = False
                _write_state(state_path, state)
                raise ShardedSurveyError(
                    f"source survey failed for shard {index} with "
                    f"return code {completed.returncode}"
                )
        observations = _archive_observations(output)
        ids = set(observations.pop("probe_ids"))
        if ids != expected_ids:
            raise ShardedSurveyError(f"archive probe union drift in shard {index}")
        if observed_union & ids:
            raise ShardedSurveyError("archive probe IDs overlap")
        observed_union.update(ids)
        state["shards"].append(
            {
                "index": index,
                "catalog": str(catalog),
                "catalog_sha256": _sha256_file(catalog),
                "output": str(output),
                "command": command,
                **observations,
            }
        )
        state["completed_shards"] = index
        _write_state(state_path, state)
    if observed_union != expected_union:
        raise ShardedSurveyError("completed archive union differs from catalogs")
    if len(observed_union) != int(manifest["union_probe_count"]):
        raise ShardedSurveyError("completed archive union count differs from parent")
    state.update(
        {
            "status": "PASS_ALL_SHARDS_VERIFIED_EXACT_UNION",
            "union_probe_count": len(observed_union),
            "intersection_probe_count": 0,
            "missing_probe_count": 0,
            "extra_probe_count": 0,
            "total_passing_probes": sum(
                shard["passing_probe_count"] for shard in state["shards"]
            ),
            "total_failing_probes": sum(
                shard["failing_probe_count"] for shard in state["shards"]
            ),
            "total_teacher_tokens": sum(
                shard["teacher_tokens"] for shard in state["shards"]
            ),
            "total_teacher_generated_output_bytes": sum(
                shard["teacher_generated_output_bytes"]
                for shard in state["shards"]
            ),
            "total_source_model_inference_seconds": sum(
                shard["source_model_inference_seconds"]
                for shard in state["shards"]
            ),
            "training_authorized": False,
            "claim_boundary": (
                "All source shards are verified, but LayerCake training remains "
                "unauthorized until a separate composed training artifact is "
                "verified and its single GPU candidate is preregistered."
            ),
        }
    )
    payload = json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    state["evidence_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    _write_state(state_path, state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--log-directory", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--splits",
        default="search",
        help="comma-separated search and/or validation splits; final_test is forbidden",
    )
    parser.add_argument("--development", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default="phi3-broad-natural-conversation-search-v3",
    )
    args = parser.parse_args(argv)
    result = run_shards(
        partition_manifest=Path(args.partition_manifest).resolve(),
        output_directory=Path(args.output_directory).resolve(),
        log_directory=Path(args.log_directory).resolve(),
        state_path=Path(args.state).resolve(),
        model=args.model,
        revision=args.revision,
        license_id=args.license,
        batch_size=args.batch_size,
        splits=tuple(
            value.strip() for value in args.splits.split(",") if value.strip()
        ),
        development=args.development,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
