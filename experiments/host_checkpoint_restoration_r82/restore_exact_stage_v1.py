"""Restore one cleaned historical weight file only after exact hash reproduction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from abi.capability_compiler_phase2_common import sha256_file
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once


STAGES = {
    "v6": {
        "checkpoint_sha256": "5bae16dec3a55759e92a8481ec5ee519a5ee8687ad1a6f019edc1df3225d092e",
        "checkpoint_bytes": 246_627_944,
        "historical_metadata_sha256": "6c9a5b6be0773b607eef45b427802354a09fb2da33db5915582feea690c3c1a1",
    },
    "v21": {
        "checkpoint_sha256": "b6bda948f24eb93e4abbfebf1e451b8911b4614a1829527c257007d8052753e1",
        "checkpoint_bytes": 246_627_944,
        "historical_metadata_sha256": "b38cab807bb10378f25a4c676d5b357583402b561dcc9ea24c9b55ec6f604b7a",
    },
    "v29": {
        "checkpoint_sha256": "537b274c752d167f4a636bf7608120f35f8fa158d604bebd018484cc8dfe79b9",
        "checkpoint_bytes": 246_627_944,
        "historical_metadata_sha256": "85a947315f9bb807d2cd0fc58a8836ae7ad963e372ce439cbc1450d79d82bad2",
    },
    "v41": {
        "checkpoint_sha256": "22df8256a0479596516c1d09c29200fc19c1cb810ac8204979b2192ce39372e0",
        "checkpoint_bytes": 246_627_944,
        "historical_metadata_sha256": "df278dffbc4ad888ce31346eedf12301bfa812cb032d040db3fa49bb5018c36e",
    },
    "v51": {
        "checkpoint_sha256": "012c5443dca21fb73874f1329bcfb6a10526284092e15d7408fff15614dd563f",
        "checkpoint_bytes": 246_720_200,
        "historical_metadata_sha256": "1419252f8e2c060154bc2abc7fabb5a3badb3dfb1368c0e11500bf8362bffcf2",
    },
}


class RestorationError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RestorationError(f"immutable restoration evidence exists: {args.output}")
    spec = STAGES[args.stage]
    candidate_model = args.candidate / "model.safetensors"
    candidate_metadata = args.candidate / "metadata.json"
    historical_model = args.historical / "model.safetensors"
    historical_metadata = args.historical / "metadata.json"
    if not candidate_model.is_file() or not candidate_metadata.is_file():
        raise RestorationError("reproduction candidate is incomplete")
    if not historical_metadata.is_file() or sha256_file(historical_metadata) != spec["historical_metadata_sha256"]:
        raise RestorationError("historical metadata is absent or changed")
    metadata = json.loads(candidate_metadata.read_text(encoding="utf-8"))
    if metadata.get("checkpoint", {}).get("sha256") != spec["checkpoint_sha256"]:
        raise RestorationError("candidate metadata does not claim the historical checkpoint")
    if candidate_model.stat().st_size != spec["checkpoint_bytes"]:
        raise RestorationError("candidate checkpoint byte count differs")
    if sha256_file(candidate_model) != spec["checkpoint_sha256"]:
        raise RestorationError("candidate checkpoint is not an exact reproduction")

    already_present = historical_model.is_file()
    if already_present:
        if historical_model.stat().st_size != spec["checkpoint_bytes"] or sha256_file(historical_model) != spec["checkpoint_sha256"]:
            raise RestorationError("a nonmatching historical weight file already exists")
    else:
        temporary = args.historical / f".model.safetensors.r82-{os.getpid()}.tmp"
        if temporary.exists():
            raise RestorationError("restoration temporary path already exists")
        try:
            shutil.copyfile(candidate_model, temporary)
            if temporary.stat().st_size != spec["checkpoint_bytes"] or sha256_file(temporary) != spec["checkpoint_sha256"]:
                raise RestorationError("copied checkpoint failed exact verification")
            os.replace(temporary, historical_model)
        finally:
            if temporary.exists():
                temporary.unlink()
    if sha256_file(historical_model) != spec["checkpoint_sha256"]:
        raise RestorationError("historical checkpoint failed final verification")
    result = {
        "format": "abi-r82-exact-host-checkpoint-restoration/1",
        "verdict": f"PASS_EXACT_{args.stage.upper()}_RESTORATION",
        "stage": args.stage,
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_bytes": spec["checkpoint_bytes"],
        "historical_metadata_sha256": spec["historical_metadata_sha256"],
        "candidate_metadata_sha256": sha256_file(candidate_metadata),
        "already_present": already_present,
        "model_bytes_equal": candidate_model.read_bytes() == historical_model.read_bytes(),
        "scientific_result_changed": False,
        "full_abi_moonshot": "OPEN",
    }
    if not result["model_bytes_equal"]:
        raise RestorationError("candidate and restored bytes differ")
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
