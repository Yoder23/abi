"""Run the post-certificate novel-prompt English generalization audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .hf_extraction import (
    HuggingFaceCausalSource,
    evaluate_output,
    load_probe_catalog,
    run_probe_catalog,
)
from .moonshot_release import ROOT, _read, verify_certificate
from .layercake_product_host import LayerCakeProductHost
from .layercake_host_runtime import (
    NativeHostRuntime,
    _runtime_candidate_manifest_sha,
    generate_native_host,
)


EVIDENCE_FORMAT = "abi-postcert-novel-english-audit-evidence/1"
CATALOG = ROOT / "catalogs/postcert_novel_english_audit_v1.json"


def _canonical_sha(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"audit evidence is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _aggregate(
    *,
    engine: str,
    observations: list[dict[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    capability_metrics: dict[str, Any] = {}
    for capability in sorted({row["capability"] for row in observations}):
        selected = [
            row for row in observations if row["capability"] == capability
        ]
        capability_metrics[capability] = {
            "observations": len(selected),
            "passes": sum(bool(row["passed"]) for row in selected),
        }
    evidence: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "status": (
            "PASS"
            if all(row["passed"] for row in observations)
            else "FAIL"
        ),
        "engine": engine,
        "catalog": str(CATALOG.relative_to(ROOT)).replace("\\", "/"),
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "observation_count": len(observations),
        "passes": sum(bool(row["passed"]) for row in observations),
        "capability_metrics": capability_metrics,
        "identity": dict(identity),
        "observations": observations,
        "claim_boundary": (
            "This audit tests 28 disclosed novel prompt forms after the "
            "bounded v2 certificate. A failure blocks a broad English-fluency "
            "claim but does not alter the historical locked-suite result."
        ),
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    return evidence


def run_layercake(
    *,
    certificate_path: Path,
    output_path: Path,
    threads: int,
) -> dict[str, Any]:
    verified = verify_certificate(certificate_path)
    certificate = _read(certificate_path)
    catalog = load_probe_catalog(CATALOG)
    observations: list[dict[str, Any]] = []
    artifact = (ROOT / certificate["candidate"]["artifact"]).resolve()
    layercake_root = (
        ROOT / certificate["sealed_layercake"]["relative_root"]
    ).resolve()
    import tempfile

    with tempfile.TemporaryDirectory(prefix="abi-postcert-audit-") as registry:
        with LayerCakeProductHost(
            english_artifact=artifact,
            layercake_root=layercake_root,
            registry_root=registry,
            threads=threads,
        ) as host:
            for probe in catalog["probes"]:
                started = time.perf_counter()
                result = host.generate(
                    probe["prompt"],
                    max_new_tokens=int(probe["max_new_tokens"]),
                )
                passed, score = evaluate_output(
                    result.output, probe["evaluator"]
                )
                observations.append(
                    {
                        "probe_id": probe["probe_id"],
                        "capability": probe["capability"],
                        "prompt": probe["prompt"],
                        "evaluator": probe["evaluator"],
                        "output": result.output,
                        "output_sha256": result.output_sha256,
                        "passed": passed,
                        "score": score,
                        "latency_seconds": time.perf_counter() - started,
                        "engine": result.engine,
                    }
                )
    evidence = _aggregate(
        engine="layercake",
        observations=observations,
        identity={
            "certificate_evidence_sha256": verified[
                "certificate_evidence_sha256"
            ],
            "runtime_graph_sha256": certificate["candidate"][
                "runtime_graph_sha256"
            ],
            "teacher_present_at_inference": False,
        },
    )
    _write_immutable(output_path, evidence)
    return evidence


def run_source(
    *,
    output_path: Path,
    model: str,
    revision: str,
    license_id: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    catalog = load_probe_catalog(CATALOG)
    source = HuggingFaceCausalSource(
        model,
        revision=revision,
        license_id=license_id,
        device=device,
        local_files_only=True,
        trust_remote_code=False,
        use_chat_template=True,
    )
    records, results = run_probe_catalog(
        source, catalog, batch_size=batch_size
    )
    observations = [
        {
            "probe_id": result["probe_id"],
            "capability": result["capability"],
            "prompt": record["prompt"],
            "evaluator": result["evaluator"],
            "output": record["output"],
            "output_sha256": record["output_sha256"],
            "passed": result["passed"],
            "score": result["score"],
            "teacher_tokens": record["teacher_tokens"],
            "teacher_token_counter": record["teacher_token_counter"],
        }
        for record, result in zip(records, results, strict=True)
    ]
    evidence = _aggregate(
        engine="source",
        observations=observations,
        identity=source.source_manifest,
    )
    _write_immutable(output_path, evidence)
    return evidence


def run_native(
    *,
    artifact_path: Path,
    output_path: Path,
    threads: int,
) -> dict[str, Any]:
    """Run the disclosed audit on one exact teacher-free native artifact."""

    catalog = load_probe_catalog(CATALOG)
    runtime = NativeHostRuntime(artifact_path, threads=threads)
    observations: list[dict[str, Any]] = []
    for probe in catalog["probes"]:
        started = time.perf_counter()
        result = generate_native_host(
            runtime,
            probe["prompt"],
            max_new_tokens=int(probe["max_new_tokens"]),
        )
        passed, score = evaluate_output(
            result["output"], probe["evaluator"]
        )
        observations.append(
            {
                "probe_id": probe["probe_id"],
                "capability": probe["capability"],
                "prompt": probe["prompt"],
                "evaluator": probe["evaluator"],
                "output": result["output"],
                "output_sha256": result["output_sha256"],
                "passed": passed,
                "score": score,
                "latency_seconds": time.perf_counter() - started,
                "engine": "native_layercake",
                "route": result["route"],
                "symbolic_handler_used": result["symbolic_handler_used"],
                "collapse": {
                    "authoritative_generated_tokens": result[
                        "authoritative_generated_tokens"
                    ],
                    "generated_utf8_bytes": result["generated_utf8_bytes"],
                },
            }
        )
    evidence = _aggregate(
        engine="native_layercake",
        observations=observations,
        identity={
            "artifact": str(artifact_path.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "candidate_manifest_sha256": (
                _runtime_candidate_manifest_sha(runtime.metadata)
            ),
            "candidate_kind": runtime.metadata["host"].get("kind"),
            "runtime_graph_sha256": runtime.metadata["runtime"][
                "graph_sha256"
            ],
            "runtime_metadata_evidence_sha256": runtime.metadata[
                "evidence_sha256"
            ],
            "decoding": runtime.decoding,
            "teacher_present_at_inference": False,
            "source_transformer_blocks_retained": 0,
        },
    )
    _write_immutable(output_path, evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="engine", required=True)
    layercake = subparsers.add_parser("layercake")
    layercake.add_argument(
        "--certificate", default="ABI_MOONSHOT_CERTIFICATE_V2.json"
    )
    layercake.add_argument("--output", required=True)
    layercake.add_argument("--threads", type=int, default=16)
    source = subparsers.add_parser("source")
    source.add_argument("--output", required=True)
    source.add_argument("--model", required=True)
    source.add_argument("--revision", required=True)
    source.add_argument("--license", required=True)
    source.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    source.add_argument("--batch-size", type=int, default=4)
    native = subparsers.add_parser("native")
    native.add_argument("--artifact", required=True)
    native.add_argument("--output", required=True)
    native.add_argument("--threads", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.engine == "layercake":
        evidence = run_layercake(
            certificate_path=(ROOT / args.certificate).resolve(),
            output_path=(ROOT / args.output).resolve(),
            threads=args.threads,
        )
    elif args.engine == "source":
        evidence = run_source(
            output_path=(ROOT / args.output).resolve(),
            model=args.model,
            revision=args.revision,
            license_id=args.license,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        evidence = run_native(
            artifact_path=(ROOT / args.artifact).resolve(),
            output_path=(ROOT / args.output).resolve(),
            threads=args.threads,
        )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "passes": evidence["passes"],
                "observations": evidence["observation_count"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
