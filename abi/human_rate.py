"""One-command, blind-safe Phase 2 human rating workflow."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .capability_compiler_phase2_common import (
    Phase2Error,
    canonical_json_bytes,
    sha256_file,
)
from .capability_compiler_phase2_rater_session import (
    export_completed_form,
    initialize_session,
    rate_interactively,
)

RATER_FORMS = {"R1": 1, "R2": 2, "R3": 3}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _identity_hash(identity: str) -> str:
    normalized = " ".join(identity.strip().casefold().split())
    if not normalized:
        raise Phase2Error("a nonempty human rater identity is required")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2Error(f"expected JSON object: {path}")
    return value


def _append_identity(registry: Path, *, rater: str, identity_sha256: str) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if registry.exists():
        for line in registry.read_bytes().splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise Phase2Error("invalid rater identity registry")
                rows.append(row)
    seen_raters = {row.get("rater") for row in rows}
    seen_identities = {row.get("identity_sha256") for row in rows}
    if rater in seen_raters:
        matching = [row for row in rows if row.get("rater") == rater]
        if len(matching) != 1 or matching[0].get("identity_sha256") != identity_sha256:
            raise Phase2Error("this sealed form is already bound to another rater identity")
        return
    if identity_sha256 in seen_identities:
        raise Phase2Error("duplicate rater identity is prohibited across forms")
    previous = rows[-1]["event_sha256"] if rows else "0" * 64
    event = {
        "format": "abi-human-rater-identity-binding/1",
        "sequence": len(rows) + 1,
        "rater": rater,
        "identity_sha256": identity_sha256,
        "bound_utc": _utc_now(),
        "previous_event_sha256": previous,
    }
    event["event_sha256"] = hashlib.sha256(canonical_json_bytes(event)).hexdigest()
    with registry.open("ab") as handle:
        handle.write(canonical_json_bytes(event))


def _sign_completion(session_dir: Path, completed_form: Path) -> dict[str, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise Phase2Error(
            "completion signing requires the 'human' extra: pip install abi-capability-compiler[human]"
        ) from exc

    key_path = session_dir / "rater_signing_key.pem"
    if key_path.exists():
        private = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        if not isinstance(private, Ed25519PrivateKey):
            raise Phase2Error("rater signing key is not Ed25519")
    else:
        private = Ed25519PrivateKey.generate()
        key_path.write_bytes(
            private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        try:
            os.chmod(key_path, stat.S_IREAD | stat.S_IWRITE)
        except OSError:
            pass
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    session = _read_json(session_dir / "session.json")
    receipt_path = session_dir / "completion_receipt.json"
    payload: dict[str, Any] = {
        "format": "abi-human-rater-signed-attestation/1",
        "rater_id": session["rater_id"],
        "rater_form": session["rater_form"],
        "human_rater": True,
        "completed_independently": True,
        "answer_key_not_accessed_before_lock": True,
        "other_ratings_not_accessed_before_lock": True,
        "completed_form_sha256": sha256_file(completed_form),
        "completion_receipt_sha256": sha256_file(receipt_path),
        "packet_manifest_sha256": session["packet_manifest_sha256"],
        "public_key_pem": public.decode("ascii"),
        "signed_utc": _utc_now(),
        "custody_boundary": (
            "The signature binds bytes and declarations. The external custodian, not software, "
            "must verify that R1, R2, and R3 are three distinct independent humans."
        ),
    }
    signature = private.sign(canonical_json_bytes(payload))
    attestation = {**payload, "signature_ed25519_hex": signature.hex()}
    target = session_dir / "signed_attestation.json"
    if target.exists():
        existing = _read_json(target)
        if existing != attestation:
            raise Phase2Error("signed completion attestation already exists and differs")
        return existing
    target.write_bytes(canonical_json_bytes(attestation))
    for path in (
        session_dir / "blinded_form.jsonl",
        session_dir / "rating_events.jsonl",
        session_dir / "session.json",
        receipt_path,
        completed_form,
        target,
    ):
        try:
            os.chmod(path, stat.S_IREAD)
        except OSError:
            pass
    return attestation


def human_rate(
    *,
    rater: str,
    packet_dir: Path,
    work_dir: Path,
    rater_identity: str | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    maximum_new_ratings: int | None = None,
) -> dict[str, Any]:
    rater = rater.upper()
    if rater not in RATER_FORMS:
        raise Phase2Error("rater must be R1, R2, or R3")
    work_dir = work_dir.resolve()
    packet_dir = packet_dir.resolve()
    session_dir = work_dir / "sessions" / rater
    completed_dir = work_dir / "completed"
    completed_form = completed_dir / f"rater_form_{RATER_FORMS[rater]}.completed.jsonl"

    if not session_dir.exists():
        identity = rater_identity or input_fn("Your full rater identity (kept hash-bound locally): ")
        identity_sha256 = _identity_hash(identity)
        _append_identity(
            work_dir / "identity_bindings.jsonl",
            rater=rater,
            identity_sha256=identity_sha256,
        )
        initialize_session(
            packet_dir=packet_dir,
            form_index=RATER_FORMS[rater],
            session_dir=session_dir,
            rater_id=identity_sha256,
        )
    elif rater_identity is not None:
        expected = _read_json(session_dir / "session.json")["rater_id"]
        if _identity_hash(rater_identity) != expected:
            raise Phase2Error("rater identity does not match the sealed session")

    signed = session_dir / "signed_attestation.json"
    if signed.exists():
        return {
            "status": "COMPLETE_LOCKED_AND_SIGNED",
            "rater": rater,
            "attestation": _read_json(signed),
        }
    progress = rate_interactively(
        session_dir=session_dir,
        input_fn=input_fn,
        output_fn=output_fn,
        maximum_new_ratings=maximum_new_ratings,
    )
    if progress["remaining"]:
        return {"status": "IN_PROGRESS", "rater": rater, **progress}
    completed_dir.mkdir(parents=True, exist_ok=True)
    export_completed_form(session_dir=session_dir, output=completed_form)
    attestation = _sign_completion(session_dir, completed_form)
    return {
        "status": "COMPLETE_LOCKED_AND_SIGNED",
        "rater": rater,
        "ratings": progress["completed"],
        "completed_form": str(completed_form),
        "completed_form_sha256": sha256_file(completed_form),
        "attestation": attestation,
    }
