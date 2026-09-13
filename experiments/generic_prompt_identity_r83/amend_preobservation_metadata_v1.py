"""Rebind R83 metadata after the zero-observation source-hash correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import _sha256_file


OLD_METADATA_SHA256 = "8505a8167a739a4dd9d207f06d11db07c03c7cd0deb00c01882156ff67eda8c3"
CHECKPOINT_SHA256 = "be62b3129b18c7f22816a2bd4e5f27abcc24526c04c99f7750cc8bd5246c5f76"
OLD_SCREEN_SHA256 = "bb7707f6ce6496588bd6e35f9abc89d6066ad81c5af47c705fb3289de42a8499"
FAILED_BINDING_SHA256 = "198027685eff3c8d356d4d61d780fa26b2cfa38583b62554900f46546cfeaf13"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--failed-screen-output", required=True, type=Path)
    args = parser.parse_args()
    metadata_path = args.candidate / "metadata.json"
    checkpoint_path = args.candidate / "model.safetensors"
    if (
        not metadata_path.is_file()
        or _sha256_file(metadata_path) != OLD_METADATA_SHA256
        or not checkpoint_path.is_file()
        or _sha256_file(checkpoint_path) != CHECKPOINT_SHA256
        or args.failed_screen_output.exists()
    ):
        parser.error("R83 candidate changed or validation output was already materialized")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    unsigned = dict(metadata)
    claimed = unsigned.pop("manifest_sha256", None)
    extension = metadata.get("r83_additive_extension", {})
    if (
        _manifest_sha(unsigned) != claimed
        or extension.get("screen_sha256") != OLD_SCREEN_SHA256
        or metadata.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
    ):
        parser.error("R83 pre-amendment metadata contract changed")
    extension["screen_sha256"] = _sha256_file(args.screen)
    extension["protocol_sha256"] = _sha256_file(args.protocol)
    metadata["r83_preobservation_amendments"] = [
        {
            "format": "abi-r83-preobservation-instrumentation-amendment/1",
            "path": args.amendment.name,
            "sha256": _sha256_file(args.amendment),
            "candidate_outputs_observed_before_amendment": 0,
            "parent_outputs_observed_before_amendment": 0,
            "candidate_checkpoint_changed": False,
            "failed_binding_sha256": FAILED_BINDING_SHA256,
            "reason": "remove_self_digest_field_before_recomputing_r81_evidence_hash",
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
