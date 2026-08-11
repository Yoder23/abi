"""Blind-safe, resumable local session for one Phase 2 human rater."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .capability_compiler_phase2_common import (
    Phase2Error,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .capability_compiler_phase2_human_ratings import (
    BOOL_FIELDS,
    PREFERENCE_VALUES,
    RATING_FIELDS,
    SCORE_FIELDS,
)


SESSION_FORMAT = "abi-capability-compiler-phase2-rater-session/1"
EVENT_FORMAT = "abi-capability-compiler-phase2-rater-event/1"
RECEIPT_FORMAT = "abi-capability-compiler-phase2-rater-completion/1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2Error(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for physical_line, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase2Error(f"invalid JSONL at {path}:{physical_line}") from exc
        if not isinstance(value, dict):
            raise Phase2Error(f"non-object JSONL row at {path}:{physical_line}")
        rows.append(value)
    return rows


def _rating_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != RATING_FIELDS:
        raise Phase2Error("rating field schema changed")
    preference = value["preference"]
    if preference not in PREFERENCE_VALUES:
        raise Phase2Error("invalid preference")
    for field in SCORE_FIELDS:
        score = value[field]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise Phase2Error(f"invalid required 1-to-5 score: {field}")
    for field in BOOL_FIELDS:
        if not isinstance(value[field], bool):
            raise Phase2Error(f"invalid required Boolean rating: {field}")
    if value["rater_comment"] is not None and not isinstance(value["rater_comment"], str):
        raise Phase2Error("rater comment must be text or null")
    return dict(value)


def initialize_session(
    *, packet_dir: Path, form_index: int, session_dir: Path, rater_id: str
) -> dict[str, Any]:
    if form_index not in (1, 2, 3):
        raise Phase2Error("form index must be 1, 2, or 3")
    if not rater_id.strip():
        raise Phase2Error("rater identifier is required")
    if session_dir.exists():
        raise Phase2Error("rater session directory already exists")
    manifest_path = packet_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("format") != "abi-capability-compiler-phase2-human-rating-packet/1"
        or manifest.get("status") != "AWAITING_THREE_INDEPENDENT_HUMAN_RATERS"
        or manifest.get("rater_forms") != 3
    ):
        raise Phase2Error("sealed rating packet changed")
    form_name = f"rater_form_{form_index}.jsonl"
    source = packet_dir / form_name
    binding = manifest.get("file_bindings", {}).get(form_name, {})
    if not source.is_file() or sha256_file(source) != binding.get("sha256"):
        raise Phase2Error("sealed rater form binding changed")
    rows = _read_jsonl(source)
    if len(rows) != binding.get("rows") or any(
        row.get("rater_form") != f"rater_form_{form_index}" for row in rows
    ):
        raise Phase2Error("sealed rater form identity changed")
    for row in rows:
        if "system_A" in row or "system_B" in row or any(row.get(field) is not None for field in RATING_FIELDS):
            raise Phase2Error("sealed rater form is not blind and unrated")
    session_dir.mkdir(parents=True)
    copied_form = session_dir / "blinded_form.jsonl"
    shutil.copyfile(source, copied_form)
    events = session_dir / "rating_events.jsonl"
    events.touch()
    session = {
        "format": SESSION_FORMAT,
        "status": "IN_PROGRESS",
        "packet_manifest_sha256": sha256_file(manifest_path),
        "rater_form": f"rater_form_{form_index}",
        "rater_id": rater_id,
        "ratings_required": len(rows),
        "started_utc": _utc_now(),
        "blinded_form": {
            "path": "blinded_form.jsonl",
            "sha256": sha256_file(copied_form),
            "rows": len(rows),
        },
        "event_log": "rating_events.jsonl",
        "blinding_key_read_by_session_tool": False,
        "answer_key_or_system_identity_present": False,
    }
    (session_dir / "session.json").write_bytes(canonical_json_bytes(session))
    return session


def _load_session(session_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = _read_json(session_dir / "session.json")
    if (
        session.get("format") != SESSION_FORMAT
        or session.get("status") != "IN_PROGRESS"
        or session.get("blinded_form", {}).get("path") != "blinded_form.jsonl"
        or session.get("event_log") != "rating_events.jsonl"
        or session.get("blinding_key_read_by_session_tool") is not False
        or session.get("answer_key_or_system_identity_present") is not False
    ):
        raise Phase2Error("rater session manifest changed")
    form_path = session_dir / str(session.get("blinded_form", {}).get("path"))
    if sha256_file(form_path) != session.get("blinded_form", {}).get("sha256"):
        raise Phase2Error("blinded session form changed")
    rows = _read_jsonl(form_path)
    if len(rows) != session.get("ratings_required"):
        raise Phase2Error("blinded session form depth changed")
    form_ids = {str(row.get("rating_id")) for row in rows}
    if len(form_ids) != len(rows) or any(
        row.get("rater_form") != session.get("rater_form") for row in rows
    ):
        raise Phase2Error("blinded session identities changed")
    return session, rows


def _event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "event_sha256"}


def load_progress(session_dir: Path) -> dict[str, Any]:
    session, rows = _load_session(session_dir)
    known = {str(row["rating_id"]) for row in rows}
    events = _read_jsonl(session_dir / str(session["event_log"]))
    latest: dict[str, dict[str, Any]] = {}
    previous = sha256_bytes(canonical_json_bytes(session))
    for index, event in enumerate(events, start=1):
        if (
            event.get("format") != EVENT_FORMAT
            or event.get("sequence") != index
            or event.get("previous_event_sha256") != previous
            or event.get("rating_id") not in known
            or event.get("rater_form") != session["rater_form"]
            or event.get("event_sha256") != sha256_bytes(canonical_json_bytes(_event_payload(event)))
        ):
            raise Phase2Error("rater event chain changed")
        rating = _rating_fields(event.get("rating", {}))
        rating_id = str(event["rating_id"])
        prior = latest.get(rating_id)
        if event.get("revision_of_sequence") != (prior["sequence"] if prior else None):
            raise Phase2Error("rater event revision lineage changed")
        latest[rating_id] = {"sequence": index, "rating": rating}
        previous = str(event["event_sha256"])
    return {
        "session": session,
        "rows": rows,
        "events": events,
        "latest": latest,
        "last_event_sha256": previous,
        "completed": len(latest),
        "remaining": len(rows) - len(latest),
    }


def record_rating(
    *, session_dir: Path, rating_id: str, rating: Mapping[str, Any], allow_revision: bool = False
) -> dict[str, Any]:
    progress = load_progress(session_dir)
    row_ids = {str(row["rating_id"]) for row in progress["rows"]}
    if rating_id not in row_ids:
        raise Phase2Error("rating identity is not in this blinded form")
    prior = progress["latest"].get(rating_id)
    if prior and not allow_revision:
        raise Phase2Error("rating already exists; an explicit revision is required")
    event = {
        "format": EVENT_FORMAT,
        "sequence": len(progress["events"]) + 1,
        "rating_id": rating_id,
        "rater_form": progress["session"]["rater_form"],
        "rating": _rating_fields(rating),
        "revision_of_sequence": prior["sequence"] if prior else None,
        "recorded_utc": _utc_now(),
        "previous_event_sha256": progress["last_event_sha256"],
    }
    event["event_sha256"] = sha256_bytes(canonical_json_bytes(event))
    with (session_dir / str(progress["session"]["event_log"])).open("ab") as handle:
        handle.write(canonical_json_bytes(event))
    return event


def export_completed_form(*, session_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise Phase2Error("immutable completed rater form already exists")
    receipt_path = session_dir / "completion_receipt.json"
    if receipt_path.exists():
        raise Phase2Error("immutable completion receipt already exists")
    progress = load_progress(session_dir)
    if progress["remaining"] != 0:
        raise Phase2Error(f"cannot export incomplete form: {progress['remaining']} ratings remain")
    completed_rows = []
    for template in progress["rows"]:
        row = dict(template)
        rating = progress["latest"][str(row["rating_id"])]["rating"]
        row.update(rating)
        completed_rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for row in completed_rows:
            handle.write(canonical_json_bytes(row))
    receipt = {
        "format": RECEIPT_FORMAT,
        "status": "COMPLETE_BLINDED_FORM_EXPORTED",
        "rater_id": progress["session"]["rater_id"],
        "rater_form": progress["session"]["rater_form"],
        "started_utc": progress["session"]["started_utc"],
        "completed_utc": _utc_now(),
        "ratings": len(completed_rows),
        "rating_events": len(progress["events"]),
        "final_event_sha256": progress["last_event_sha256"],
        "source_blinded_form_sha256": progress["session"]["blinded_form"]["sha256"],
        "completed_form": {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "rows": len(completed_rows),
        },
        "blinding_key_read_by_session_tool": False,
    }
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    return receipt


def _ask_score(prompt: str, input_fn: Callable[[str], str]) -> int:
    while True:
        value = input_fn(prompt).strip()
        if value in {"1", "2", "3", "4", "5"}:
            return int(value)


def _ask_bool(prompt: str, input_fn: Callable[[str], str]) -> bool:
    while True:
        value = input_fn(prompt).strip().lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False


def rate_interactively(
    *,
    session_dir: Path,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    maximum_new_ratings: int | None = None,
) -> dict[str, int]:
    progress = load_progress(session_dir)
    unrated = [row for row in progress["rows"] if str(row["rating_id"]) not in progress["latest"]]
    recorded = 0
    for row in unrated:
        if maximum_new_ratings is not None and recorded >= maximum_new_ratings:
            break
        output_fn(f"\n[{progress['completed'] + recorded + 1}/{len(progress['rows'])}] {row['capability']}")
        output_fn(f"PROMPT:\n{row['prompt']}\n\nOUTPUT A:\n{row['output_A']}\n\nOUTPUT B:\n{row['output_B']}")
        while True:
            raw = input_fn("Preference [A/B/T=tie/U=both unacceptable/Q=quit]: ").strip().upper()
            if raw == "Q":
                current = load_progress(session_dir)
                return {"recorded_this_run": recorded, "completed": current["completed"], "remaining": current["remaining"]}
            preference = {"T": "TIE", "U": "BOTH_UNACCEPTABLE"}.get(raw, raw)
            if preference in PREFERENCE_VALUES:
                break
        rating = {
            "preference": preference,
            "fluency_A_1_to_5": _ask_score("Fluency A [1-5]: ", input_fn),
            "fluency_B_1_to_5": _ask_score("Fluency B [1-5]: ", input_fn),
            "grounding_and_adherence_A_1_to_5": _ask_score("Grounding/adherence A [1-5]: ", input_fn),
            "grounding_and_adherence_B_1_to_5": _ask_score("Grounding/adherence B [1-5]: ", input_fn),
            "repetition_or_collapse_A": _ask_bool("Repetition/collapse in A? [y/n]: ", input_fn),
            "repetition_or_collapse_B": _ask_bool("Repetition/collapse in B? [y/n]: ", input_fn),
            "rater_comment": input_fn("Optional comment (Enter for none): ").strip() or None,
        }
        record_rating(session_dir=session_dir, rating_id=str(row["rating_id"]), rating=rating)
        recorded += 1
    current = load_progress(session_dir)
    return {"recorded_this_run": recorded, "completed": current["completed"], "remaining": current["remaining"]}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser("init")
    init_parser.add_argument("--packet-dir", required=True)
    init_parser.add_argument("--form-index", required=True, type=int)
    init_parser.add_argument("--session-dir", required=True)
    init_parser.add_argument("--rater-id", required=True)
    rate_parser = commands.add_parser("rate")
    rate_parser.add_argument("--session-dir", required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--session-dir", required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--session-dir", required=True)
    export_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    session_dir = Path(args.session_dir).resolve()
    if args.command == "init":
        result = initialize_session(
            packet_dir=Path(args.packet_dir).resolve(),
            form_index=args.form_index,
            session_dir=session_dir,
            rater_id=args.rater_id,
        )
    elif args.command == "rate":
        result = rate_interactively(session_dir=session_dir)
    elif args.command == "status":
        progress = load_progress(session_dir)
        result = {key: progress[key] for key in ("completed", "remaining")}
        result["rater_form"] = progress["session"]["rater_form"]
    else:
        result = export_completed_form(session_dir=session_dir, output=Path(args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
