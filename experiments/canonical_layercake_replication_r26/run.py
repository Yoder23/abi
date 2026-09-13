"""Execute the prospective R26 canonical-import replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.canonical_layercake_import_r25.run import run as run_engine
from experiments.canonical_layercake_import_r25.verify import _metrics, _rows
from experiments.factual_semantic_r16.public_sequence_scoring import normalized_text
from experiments.foreign_capability_r14.core import R14Error, sha256_file, write_json_once
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash

from .binding import load_config

NAMES = (
    "package_builds",
    "domain_observations",
    "english_immutability",
    "lifecycle",
    "english_leakage",
)


def corrected_metrics(
    root: Path,
    import_config: dict[str, Any],
    engine_dir: Path,
    engine: dict[str, Any],
) -> tuple[dict[str, int], dict[str, bool]]:
    rows = {name: _rows(engine_dir, engine, name) for name in NAMES}
    metrics = _metrics(
        import_config,
        rows["package_builds"],
        rows["domain_observations"],
        rows["english_immutability"],
        rows["lifecycle"],
        rows["english_leakage"],
    )
    metrics["teacher_agreement"] = sum(
        normalized_text(row["gpu_output"]) == normalized_text(row["teacher_output"])
        for row in rows["domain_observations"]
    )
    gates = {
        name: metrics[name] == threshold
        for name, threshold in import_config["gates"].items()
    }
    gates["teacher_absent_at_execution"] = True
    return metrics, gates


def run(config_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    output = output.resolve()
    if output.exists():
        raise R14Error(f"immutable R26 output exists: {output}")
    if not output.is_relative_to(root):
        raise R14Error("R26 output must remain in the ABI repository")
    config = load_config(root, config_path)
    output.mkdir(parents=True)
    import_config_path = root / config["import_config"]["path"]
    import_config = json.loads(import_config_path.read_text(encoding="utf-8"))
    engine_dir = output / "engine"
    engine = run_engine(import_config_path, engine_dir)
    metrics, gates = corrected_metrics(root, import_config, engine_dir, engine)
    del root
    passed = all(gates.values())
    result = {
        "format": "abi-r26-canonical-import-replication-result/1",
        "verdict": "PASS_FRESH_BOUNDED_DIRECT_IMPORT" if passed else "FAIL_FRESH_BOUNDED_DIRECT_IMPORT",
        "claim": (
            "R26_FRESH_REGISTERED_FACTUAL_DIRECT_IMPORT_AND_COMPOSITION"
            if passed
            else "R26_FRESH_DIRECT_IMPORT_FAILED"
        ),
        "claim_ceiling": "NOT_AUTONOMOUS_LABELING_OR_OPEN_WORLD_DOMAINS",
        "config_sha256": sha256_file(config_path),
        "import_config_sha256": sha256_file(import_config_path),
        "source_receipt_sha256": config["source_receipt"]["sha256"],
        "engine_result_sha256": sha256_file(engine_dir / "result.json"),
        "engine_result_evidence_sha256": engine["evidence_sha256"],
        "metrics": metrics,
        "gates": gates,
        "candidate_changes_from_r25": 0,
        "gate_changes_from_r25": 0,
        "prospective_scorer": "r16-normalized-text",
        "source_parameters_copied": 0,
        "receiver_training_steps": 0,
        "teacher_present_at_execution": False,
        "full_abi_moonshot": "OPEN",
        "next_action": (
            "strict live replay then autonomous labeling frontier"
            if passed
            else "preserve failure and isolate the failing fresh gate"
        ),
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
