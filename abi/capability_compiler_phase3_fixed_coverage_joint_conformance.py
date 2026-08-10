"""Run the existing full-operator layer-zero fit at the fixed proven coverage."""

from __future__ import annotations
from collections import defaultdict
import argparse, hashlib, json
from pathlib import Path
from . import capability_compiler_phase3_native_trajectory_full_operator_layer0 as joint
from .capability_compiler_phase2_common import CAPABILITIES
from .capability_compiler_phase3 import Phase3Error


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol=json.loads(protocol_path.read_text(encoding="utf-8"));coverage=protocol.get("coverage_control",{})
    if coverage.get("train_records_per_capability")!=300 or coverage.get("validation_rank_indices")!=[30,31]:raise Phase3Error("joint conformance coverage changed")
    original=joint.dual._calibration_examples
    def expanded(examples,*,seed,train_per_capability,validation_per_capability,maximum_tokens):
        if (train_per_capability,validation_per_capability,maximum_tokens)!=(30,2,128):raise Phase3Error("historical split changed")
        grouped=defaultdict(list)
        for row in examples:grouped[str(row["capability"])].append(row)
        train=[];validation=[];tokens=0
        for capability in CAPABILITIES:
            ranked=sorted(grouped[capability],key=lambda row:hashlib.sha256(f"{seed}:{row['record_id']}".encode()).digest())
            for source,destination in ((ranked[:30]+ranked[32:302],train),(ranked[30:32],validation)):
                for row in source:
                    packed=(list(row["source_ids"])+list(row["target_actions"])[:-1])[:maximum_tokens];destination.append({"record_id":row["record_id"],"capability":capability,"input_ids":packed});tokens+=len(packed)
        if len(train)!=4200 or len(validation)!=28:raise Phase3Error("fixed coverage population changed")
        train.sort(key=lambda row:hashlib.sha256(f"train:{seed}:{row['record_id']}".encode()).digest());validation.sort(key=lambda row:hashlib.sha256(f"validation:{seed}:{row['record_id']}".encode()).digest());return train,validation,tokens
    joint.dual._calibration_examples=expanded
    try:return joint.execute(root,protocol_path,output)
    finally:joint.dual._calibration_examples=original


def main():
    p=argparse.ArgumentParser();p.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_FIXED_COVERAGE_JOINT_CONFORMANCE_PROTOCOL_V384.json");p.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_native_trajectory/fixed_coverage_joint_v385");a=p.parse_args();root=Path.cwd().resolve();print(json.dumps(execute(root,(root/a.protocol).resolve(),(root/a.output_dir).resolve()),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
