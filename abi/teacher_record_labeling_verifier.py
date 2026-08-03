"""Independently verify labeling evidence and exercise fail-closed tamper checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .layercake_host import _sha256_file
from .teacher_record_labeling import (
    BENCHMARK_FORMAT,
    EVIDENCE_FORMAT,
    _canonical_sha,
    _metrics,
    _parse_semantic_label,
    _write_immutable,
    deterministic_risk_screen,
    finalize_label,
)


class LabelingVerificationError(RuntimeError):
    """Raised when labeling evidence is stale, inconsistent, or tampered."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LabelingVerificationError(message)


def _without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop(key, None)
    return result


def verify_documents(
    *,
    protocol: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    evidence: Mapping[str, Any],
    lock: Mapping[str, Any],
    protocol_path: Path,
    benchmark_path: Path,
    implementation_path: Path,
) -> dict[str, Any]:
    _require(benchmark.get("format") == BENCHMARK_FORMAT, "benchmark format")
    _require(evidence.get("format") == EVIDENCE_FORMAT, "evidence format")
    _require(
        benchmark.get("benchmark_sha256")
        == _canonical_sha(_without(benchmark, "benchmark_sha256")),
        "benchmark self hash",
    )
    _require(
        evidence.get("evidence_sha256")
        == _canonical_sha(_without(evidence, "evidence_sha256")),
        "evidence self hash",
    )
    _require(
        lock["protocol"]["sha256"] == _sha256_file(protocol_path),
        "protocol lock hash",
    )
    _require(
        lock["benchmark"]["file_sha256"] == _sha256_file(benchmark_path),
        "benchmark lock file hash",
    )
    _require(
        lock["benchmark"]["self_sha256"] == benchmark["benchmark_sha256"],
        "benchmark lock self hash",
    )
    implementation_sha = _sha256_file(implementation_path)
    _require(
        lock["implementation"]["labeler_sha256"] == implementation_sha,
        "implementation lock hash",
    )
    _require(
        evidence["implementation_sha256"] == implementation_sha,
        "evidence implementation hash",
    )
    _require(
        evidence["protocol_sha256"] == _sha256_file(protocol_path),
        "evidence protocol hash",
    )
    _require(
        evidence["benchmark_file_sha256"] == _sha256_file(benchmark_path),
        "evidence benchmark file hash",
    )
    _require(
        evidence["benchmark_sha256"] == benchmark["benchmark_sha256"],
        "evidence benchmark self hash",
    )
    for source in protocol["source_material"]:
        _require(
            _sha256_file(Path(source["path"])) == source["sha256"],
            "source archive hash",
        )
    _require(
        _sha256_file(Path(protocol["ontology"]["path"]))
        == protocol["ontology"]["sha256"],
        "ontology hash",
    )

    if "exclusion" in benchmark:
        exclusion = benchmark["exclusion"]
        exclusion_path = Path(exclusion["benchmark_path"])
        _require(
            _sha256_file(exclusion_path) == exclusion["benchmark_file_sha256"],
            "exclusion benchmark file hash",
        )
        prior = json.loads(exclusion_path.read_text(encoding="utf-8"))
        _require(
            prior["benchmark_sha256"] == exclusion["benchmark_sha256"],
            "exclusion benchmark identity",
        )
        prior_ids = {
            str(record_id)
            for partition in prior["partitions"].values()
            for row in partition
            for record_id in row["source_record_ids"]
        }
        current_ids = {
            str(record_id)
            for row in benchmark["partitions"]["validation"]
            for record_id in row["source_record_ids"]
        }
        _require(not prior_ids & current_ids, "source record overlap")

    rows = benchmark["partitions"][evidence["mode"]]
    observations = evidence["observations"]
    _require(len(rows) == len(observations), "observation depth")
    _require(
        len(observations) == int(benchmark["counts"][evidence["mode"]]),
        "partition count",
    )
    row_by_id = {row["benchmark_id"]: row for row in rows}
    _require(
        set(row_by_id) == {row["benchmark_id"] for row in observations},
        "observation identity set",
    )
    for observation in observations:
        _require(
            observation["observation_sha256"]
            == _canonical_sha(_without(observation, "observation_sha256")),
            "observation self hash",
        )
        row = row_by_id[observation["benchmark_id"]]
        _require(observation["blind_sha256"] == row["blind_sha256"], "blind hash")
        _require(observation["gold"] == row["gold"], "gold binding")
        _require(observation["family"] == row["family"], "family binding")
        semantic_output = str(observation["semantic_output"])
        _require(
            observation["semantic_output_sha256"]
            == hashlib.sha256(semantic_output.encode("utf-8")).hexdigest(),
            "semantic output hash",
        )
        semantic = _parse_semantic_label(semantic_output)
        _require(semantic == observation["semantic_label"], "semantic parse replay")
        deterministic = deterministic_risk_screen(
            str(row["blind"]["prompt"]), str(row["blind"]["response"])
        )
        _require(
            deterministic == observation["deterministic"],
            "deterministic replay",
        )
        final_label = finalize_label(
            semantic=semantic,
            deterministic=deterministic,
        )
        _require(final_label == observation["final_label"], "final label replay")
        emitted = observation["emitted_label"]
        _require(
            {key: emitted[key] for key in final_label} == final_label,
            "emitted label binding",
        )
        evidence_basis = {
            "blind_sha256": row["blind_sha256"],
            "ontology_sha256": protocol["ontology"]["sha256"],
            "semantic_output_sha256": observation["semantic_output_sha256"],
            "deterministic": deterministic,
            "implementation_sha256": implementation_sha,
            "final_label": final_label,
        }
        _require(
            emitted["label_evidence_sha256"] == _canonical_sha(evidence_basis),
            "label evidence hash",
        )
        _require(
            len(observation["authoritative_generated_token_ids"])
            == observation["generated_tokens"],
            "authoritative token count",
        )

    recomputed_metrics = _metrics(observations)
    _require(recomputed_metrics == evidence["metrics"], "metric recomputation")
    runtime = evidence["classifier"]["runtime"]
    recomputed_checks = {
        "expected_partition_depth": len(observations)
        == int(benchmark["counts"][evidence["mode"]]),
        "all_semantic_outputs_parse": all(
            row["semantic_parsed"] for row in observations
        ),
        "all_outputs_eos_terminated": all(
            row["finish_reason"] == "eos_token" for row in observations
        ),
        "authoritative_token_counts_match": all(
            len(row["authoritative_generated_token_ids"])
            == row["generated_tokens"]
            for row in observations
        ),
        "classifier_is_cuda_int8_without_cpu_offload": bool(
            runtime["device"] == "cuda"
            and runtime["weight_execution_precision"] == "bitsandbytes_int8"
            and runtime["cpu_offload_enabled"] is False
        ),
        "source_archives_unchanged": all(
            _sha256_file(Path(source["path"])) == source["sha256"]
            for source in protocol["source_material"]
        ),
        "final_test_absent": benchmark["counts"]["final_test"] == 0,
        "layercake_not_invoked": evidence["layercake_invoked"] is False,
        "training_artifact_not_composed": evidence["training_artifact_composed"]
        is False,
        "overall_accuracy_at_least_095": recomputed_metrics[
            "overall_exact_destination_domain_accuracy"
        ]
        >= 0.95,
        "all_class_recalls_at_least_090": all(
            row["recall"] >= 0.90
            for row in recomputed_metrics["per_class"].values()
        ),
        "english_precision_at_least_098": recomputed_metrics["per_class"][
            "english_core"
        ]["precision"]
        >= 0.98,
        "no_non_english_leak_to_core": recomputed_metrics[
            "non_english_mislabeled_as_english"
        ]
        == 0,
        "known_domain_macro_f1_at_least_095": recomputed_metrics[
            "known_domain_macro_f1"
        ]
        >= 0.95,
        "quarantine_recall_at_least_095": recomputed_metrics["per_class"][
            "quarantine"
        ]["recall"]
        >= 0.95,
        "known_capability_accuracy_at_least_085": recomputed_metrics[
            "known_record_capability_accuracy"
        ]
        >= 0.85,
        "all_quarantine_family_recalls_at_least_090": all(
            value >= 0.90
            for value in recomputed_metrics["quarantine_family_recall"].values()
        ),
    }
    _require(recomputed_checks == evidence["checks"], "gate recomputation")
    _require(all(recomputed_checks.values()), "recomputed gate failure")
    _require(evidence["status"] == "PASS", "evidence status")
    _require(evidence["layercake_invoked"] is False, "LayerCake invocation")
    _require(
        evidence["training_artifact_composed"] is False,
        "training artifact composition",
    )
    return {
        "observation_count": len(observations),
        "metrics": recomputed_metrics,
        "implementation_sha256": implementation_sha,
        "benchmark_sha256": benchmark["benchmark_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
    }


def adversarial_suite(
    *,
    protocol: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    evidence: Mapping[str, Any],
    lock: Mapping[str, Any],
    protocol_path: Path,
    benchmark_path: Path,
    implementation_path: Path,
) -> dict[str, str]:
    cases: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = {}

    altered_label = copy.deepcopy(evidence)
    altered_label["observations"][0]["final_label"]["domain"] = "python"
    altered_label["observations"][0]["observation_sha256"] = _canonical_sha(
        _without(altered_label["observations"][0], "observation_sha256")
    )
    altered_label["evidence_sha256"] = _canonical_sha(
        _without(altered_label, "evidence_sha256")
    )
    cases["mutated_final_label"] = (
        copy.deepcopy(protocol), copy.deepcopy(benchmark), altered_label, copy.deepcopy(lock)
    )
    altered_tokens = copy.deepcopy(evidence)
    altered_tokens["observations"][0]["generated_tokens"] += 1
    altered_tokens["observations"][0]["observation_sha256"] = _canonical_sha(
        _without(altered_tokens["observations"][0], "observation_sha256")
    )
    altered_tokens["evidence_sha256"] = _canonical_sha(
        _without(altered_tokens, "evidence_sha256")
    )
    cases["mutated_token_count"] = (
        copy.deepcopy(protocol), copy.deepcopy(benchmark), altered_tokens, copy.deepcopy(lock)
    )
    altered_gold = copy.deepcopy(benchmark)
    altered_gold["partitions"]["validation"][0]["gold"]["domain"] = "python"
    altered_gold["benchmark_sha256"] = _canonical_sha(
        _without(altered_gold, "benchmark_sha256")
    )
    cases["rehashed_gold_mutation"] = (
        copy.deepcopy(protocol), altered_gold, copy.deepcopy(evidence), copy.deepcopy(lock)
    )
    altered_protocol = copy.deepcopy(protocol)
    altered_protocol["source_material"][0]["sha256"] = "0" * 64
    cases["mutated_source_identity"] = (
        altered_protocol, copy.deepcopy(benchmark), copy.deepcopy(evidence), copy.deepcopy(lock)
    )
    altered_lock = copy.deepcopy(lock)
    altered_lock["implementation"]["labeler_sha256"] = "0" * 64
    cases["mutated_implementation_lock"] = (
        copy.deepcopy(protocol), copy.deepcopy(benchmark), copy.deepcopy(evidence), altered_lock
    )

    results = {}
    for name, (case_protocol, case_benchmark, case_evidence, case_lock) in cases.items():
        try:
            verify_documents(
                protocol=case_protocol,
                benchmark=case_benchmark,
                evidence=case_evidence,
                lock=case_lock,
                protocol_path=protocol_path,
                benchmark_path=benchmark_path,
                implementation_path=implementation_path,
            )
        except LabelingVerificationError as exc:
            results[name] = f"REJECTED: {exc}"
        else:
            raise LabelingVerificationError(f"adversarial case accepted: {name}")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    paths = {
        key: Path(getattr(args, key)).resolve()
        for key in ("protocol", "benchmark", "evidence", "lock", "implementation")
    }
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    benchmark = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
    evidence = json.loads(paths["evidence"].read_text(encoding="utf-8"))
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    verified = verify_documents(
        protocol=protocol,
        benchmark=benchmark,
        evidence=evidence,
        lock=lock,
        protocol_path=paths["protocol"],
        benchmark_path=paths["benchmark"],
        implementation_path=paths["implementation"],
    )
    attacks = adversarial_suite(
        protocol=protocol,
        benchmark=benchmark,
        evidence=evidence,
        lock=lock,
        protocol_path=paths["protocol"],
        benchmark_path=paths["benchmark"],
        implementation_path=paths["implementation"],
    )
    report = {
        "format": "abi-teacher-record-labeling-verifier-report/1",
        "status": "PASS",
        "verified": verified,
        "adversarial_cases": attacks,
        "all_adversarial_cases_rejected": all(
            result.startswith("REJECTED:") for result in attacks.values()
        ),
        "files": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in paths.items()
        },
    }
    report["report_sha256"] = _canonical_sha(report)
    _write_immutable(Path(args.output).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
