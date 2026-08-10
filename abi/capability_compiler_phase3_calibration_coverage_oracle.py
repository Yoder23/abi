"""Run the native-attention oracle at one fixed existing-data coverage point."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path

from . import capability_compiler_phase3_native_attention_interface_oracle as oracle
from .capability_compiler_phase2_common import CAPABILITIES
from .capability_compiler_phase3 import Phase3Error


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    coverage = protocol.get("coverage_control", {})
    if coverage.get("train_records_per_capability") != 300 or coverage.get("validation_rank_indices") != [30, 31]:
        raise Phase3Error("calibration coverage boundary changed")
    original = oracle.dual._calibration_examples

    def expanded(examples, *, seed, train_per_capability, validation_per_capability, maximum_tokens):
        if int(train_per_capability) != 30 or int(validation_per_capability) != 2 or int(maximum_tokens) != 128:
            raise Phase3Error("historical calibration boundary changed")
        grouped = defaultdict(list)
        for row in examples:
            grouped[str(row["capability"])].append(row)
        train = []; validation = []; token_count = 0
        for capability in CAPABILITIES:
            ranked = sorted(
                grouped[capability],
                key=lambda row: hashlib.sha256(f"{seed}:{row['record_id']}".encode()).digest(),
            )
            selected_train = ranked[:30] + ranked[32:302]
            selected_validation = ranked[30:32]
            if len(selected_train) != 300 or len(selected_validation) != 2:
                raise Phase3Error("insufficient fixed coverage records")
            for row in selected_train:
                packed = (list(row["source_ids"]) + list(row["target_actions"])[:-1])[:maximum_tokens]
                train.append({"record_id": row["record_id"], "capability": capability, "input_ids": packed})
                token_count += len(packed)
            for row in selected_validation:
                packed = (list(row["source_ids"]) + list(row["target_actions"])[:-1])[:maximum_tokens]
                validation.append({"record_id": row["record_id"], "capability": capability, "input_ids": packed})
                token_count += len(packed)
        train.sort(key=lambda row: hashlib.sha256(f"train:{seed}:{row['record_id']}".encode()).digest())
        validation.sort(key=lambda row: hashlib.sha256(f"validation:{seed}:{row['record_id']}".encode()).digest())
        return train, validation, token_count

    oracle.dual._calibration_examples = expanded
    try:
        return oracle.execute(root, protocol_path, output)
    finally:
        oracle.dual._calibration_examples = original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CALIBRATION_COVERAGE_ORACLE_PROTOCOL_V376.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/calibration_coverage_oracle_v377")
    args = parser.parse_args(); root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
