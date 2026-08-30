"""Reveal committed held-out R8 seeds only after the freeze receipt exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .capability_generator import canonical_json_bytes, committed_heldout_capabilities
from .native_host import sha256_file


class RevealError(RuntimeError):
    """Raised when the committed held-out order cannot be proven."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RevealError(f"expected JSON object: {path}")
    return value


def reveal(config_path: Path, campaign_root: Path, secret_hex: str) -> dict[str, Any]:
    freeze_path = campaign_root / "freeze_receipt.json"
    reveal_path = campaign_root / "heldout_reveal.json"
    private_path = campaign_root / "evaluator_private/capabilities.json"
    if reveal_path.exists() or private_path.exists() or not freeze_path.is_file():
        raise RevealError("held-out reveal requires one existing freeze and fresh outputs")
    config = _json(config_path)
    frozen = _json(freeze_path)
    if frozen.get("status") != "FROZEN_BEFORE_HELDOUT_REVEAL":
        raise RevealError("pre-reveal components were not frozen")
    commitment = str(config["splits"]["heldout_secret_commitment_sha256"])
    if frozen.get("heldout_secret_commitment_sha256") != commitment:
        raise RevealError("held-out commitment differs from freeze receipt")
    capabilities = committed_heldout_capabilities(
        secret_hex,
        expected_commitment=commitment,
        count=int(config["splits"]["heldout_capabilities"]),
    )
    private_value = {
        "format": "abi-native-transfer-r8-private-heldout-capabilities/1",
        "freeze_receipt_sha256": sha256_file(freeze_path),
        "secret_sha256": commitment,
        "capabilities": [capability.private_document() for capability in capabilities],
    }
    private_value["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(private_value)
    ).hexdigest()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(
        json.dumps(private_value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    value = {
        "format": "abi-native-transfer-r8-heldout-reveal/1",
        "status": "HELDOUT_REVEALED_AFTER_FREEZE",
        "created_unix_time_ns": time.time_ns(),
        "freeze_receipt_sha256": sha256_file(freeze_path),
        "secret_sha256": commitment,
        "capabilities": [
            {
                "capability_id": capability.capability_id,
                "seed_commitment": capability.seed_commitment,
            }
            for capability in capabilities
        ],
        "private_capabilities_sha256": sha256_file(private_path),
        "private_capability_rules_disclosed_to_worker": False,
    }
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    reveal_path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--secret-hex", required=True)
    args = parser.parse_args()
    try:
        value = reveal(
            Path(args.config).resolve(), Path(args.campaign_root).resolve(), args.secret_hex
        )
    except RevealError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
