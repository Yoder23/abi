"""Repair omitted inherited R78 topology before any R84 candidate binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import _sha256_file


CHECKPOINT_SHA256 = "45a8a4c39bee2ad1e5492df2a4906e80ade09eb7ac0fac9378a9947cb344bf0c"
OLD_METADATA_SHA256 = "b470fadd0c8faf1834c255df31455b74021ca385d02b1bf76386ec68ec5a4e8c"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--screen-output", required=True, type=Path)
    args = parser.parse_args()
    metadata_path = args.candidate / "metadata.json"
    checkpoint_path = args.candidate / "model.safetensors"
    parent_metadata_path = args.parent / "metadata.json"
    if (
        not metadata_path.is_file()
        or _sha256_file(metadata_path) != OLD_METADATA_SHA256
        or not checkpoint_path.is_file()
        or _sha256_file(checkpoint_path) != CHECKPOINT_SHA256
        or _sha256_file(parent_metadata_path) != PARENT_METADATA_SHA256
        or args.binding.exists()
        or args.screen_output.exists()
    ):
        parser.error("R84 candidate changed, or candidate observations may exist")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    unsigned = dict(metadata)
    claimed = unsigned.pop("manifest_sha256", None)
    pointer = metadata.get("acquired_core", {}).get("prompt_identity_carriage") or {}
    inherited = parent.get("acquired_core", {}).get("capability_cake_expansion")
    if (
        _manifest_sha(unsigned) != claimed
        or metadata.get("acquired_core", {}).get("capability_cake_expansion") is not None
        or not isinstance(inherited, dict)
        or inherited.get("installed_deep_adapters") != 84
        or pointer.get("parent_state_preserved_exact") is not True
        or pointer.get("parent_state_sha256_before")
        != pointer.get("parent_state_sha256_after")
    ):
        parser.error("R84 pre-amendment state contract changed")
    metadata["acquired_core"]["capability_cake_expansion"] = inherited
    extension = metadata["r84_additive_extension"]
    extension["extension_sha256"] = _sha256_file(args.extension)
    extension["screen_sha256"] = _sha256_file(args.screen)
    extension["protocol_sha256"] = _sha256_file(args.protocol)
    metadata["r84_preobservation_amendments"] = [
        {
            "format": "abi-r84-preobservation-metadata-amendment/1",
            "path": args.amendment.name,
            "sha256": _sha256_file(args.amendment),
            "candidate_outputs_observed_before_amendment": 0,
            "parent_outputs_observed_before_amendment": 0,
            "candidate_checkpoint_changed": False,
            "reason": "inherit_unchanged_r78_capability_expansion_ledger",
        }
    ]
    metadata.pop("manifest_sha256", None)
    metadata["manifest_sha256"] = _manifest_sha(metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "metadata_sha256": _sha256_file(metadata_path),
                "manifest_sha256": metadata["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
