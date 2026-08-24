import json
from pathlib import Path

import pytest

from abi.final_mile import FinalMileError, freeze_starting_point, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_starting_point_binds_existing_product(tmp_path):
    output = tmp_path / "frozen_starting_point.json"
    result = freeze_starting_point(ROOT, output=output)
    assert result["repository_commit"] == "a2f2f8f685119c658048faa50b58d01544bc6e92"
    assert len(result["release_inventory"]) == 52
    assert len(result["package_bindings"]) == 7
    assert result["product_identity"]["english_core"]["archive_sha256"] == (
        "acb787b3ffa0153c57d88cd37ba81c3f00b370d4ca4937e659cd4c775851f25d"
    )
    assert result["current_claim_ceiling"]["tier_d_full_abi_moonshot"] == "NOT_PROVEN"
    assert sha256_file(output) == sha256_file(output)


def test_frozen_starting_point_is_immutable(tmp_path):
    output = tmp_path / "frozen_starting_point.json"
    freeze_starting_point(ROOT, output=output)
    with pytest.raises(FinalMileError, match="already exists"):
        freeze_starting_point(ROOT, output=output)


def test_frozen_starting_point_evidence_hash_replays(tmp_path):
    output = tmp_path / "frozen_starting_point.json"
    freeze_starting_point(ROOT, output=output)
    value = json.loads(output.read_text(encoding="utf-8"))
    claimed = value.pop("evidence_sha256")
    import hashlib

    from abi.final_mile import canonical_json_bytes

    assert hashlib.sha256(canonical_json_bytes(value)).hexdigest() == claimed
