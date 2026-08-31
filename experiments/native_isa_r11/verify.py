"""Fail-closed recomputation of R11 teacher-to-student output equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.copy_paste_r10.runtime import canonical_prediction
from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    committed_heldout_capabilities,
)

from .core import (
    RecurrentTransitionNeuralISA,
    load_package,
    sha256_bytes,
    sha256_file,
    transition_bytes,
)
from .run import _bind, _conditions, _json, _resolve, _rows


class R11VerificationError(RuntimeError):
    """Raised when R11 evidence is missing, stale, malformed, or fails a gate."""


FIELDS = {
    "host",
    "capability_id",
    "condition",
    "row_id",
    "prompt_sha256",
    "package_sha256",
    "backend_active",
    "codec_active",
    "prediction_token_id",
    "canonical_prediction",
    "canonical_output_utf8_hex",
    "canonical_probabilities",
    "neural_state",
}


def _fail(message: str) -> None:
    raise R11VerificationError(message)


def _evidence(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if stored != actual:
        _fail(f"{label} evidence hash changed")


def _jsonl(path: Path, expected_sha: str, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_sha:
        _fail(f"raw evidence unavailable: {path}")
    rows = []
    try:
        for line in path.open("r", encoding="utf-8"):
            value = json.loads(line)
            if not isinstance(value, dict):
                _fail("raw observation is not an object")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R11VerificationError(f"raw evidence unreadable: {path}") from exc
    if len(rows) != expected_rows:
        _fail(f"raw row count changed: {path}")
    return rows


def _probabilities(value: Any, label: str, *, normalized: bool = True) -> list[float]:
    if not isinstance(value, list) or len(value) != 8:
        _fail(f"{label} width changed")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        _fail(f"{label} is non-finite")
    if normalized and (
        any(item < -1e-6 or item > 1.000001 for item in result) or abs(sum(result) - 1.0) > 2e-5
    ):
        _fail(f"{label} is not a probability distribution")
    return result


def _accuracy(rows: Sequence[Mapping[str, Any]], answers: Mapping[str, int]) -> float:
    return sum(
        int(row["canonical_prediction"] == answers[str(row["row_id"])]) for row in rows
    ) / len(rows)


def _close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= 2e-6 for a, b in zip(left, right)
    )


def verify(config_path: Path, reveal_path: Path, run_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = _json(config_path)
    if config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL":
        _fail("R11 preregistration is not frozen")
    receipt = _json(run_dir / "receipt.json")
    _evidence(receipt, "run receipt")
    if (
        receipt.get("format") != "abi-native-neural-isa-r11-run/1"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("reveal_sha256") != sha256_file(reveal_path)
        or receipt.get("bindings") != _bind(root, config)
        or receipt.get("claim_ceiling") != "LEVEL_1_OUTPUT_EQUIVALENCE_ONLY"
    ):
        _fail("run identity or claim ceiling changed")
    reveal = _json(reveal_path)
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R11VerificationError("held-out reveal malformed") from exc
    if (
        len(secret) != 32
        or sha256_bytes(secret) != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        _fail("held-out commitment changed")
    capabilities = committed_heldout_capabilities(
        reveal["secret_hex"],
        expected_commitment=config["heldout_seed_commitment"],
        count=int(config["data"]["heldout_capabilities"]),
    )
    capability_ids = [item.capability_id for item in capabilities]
    _, evaluation = _rows(config, capabilities)
    expected = {str(row["row_id"]): row for rows in evaluation for row in rows}
    if len(expected) != sum(len(rows) for rows in evaluation):
        _fail("evaluation row identities collide")
    answers = {row_id: int(row["answer"]) for row_id, row in expected.items()}

    codec_receipt = _json(_resolve(root, config["codec_freeze"]["receipt"]))
    _evidence(codec_receipt, "codec freeze")
    if (
        codec_receipt.get("heldout_seed_revealed") is not False
        or codec_receipt.get("heldout_seed_commitment") != config["heldout_seed_commitment"]
        or codec_receipt["host_codecs"]["sha256"]
        != sha256_file(_resolve(root, config["codec_freeze"]["tensors"]))
    ):
        _fail("pre-capability codec freeze boundary changed")

    package_manifest = receipt.get("packages")
    if not isinstance(package_manifest, dict):
        _fail("package manifest missing")
    before_item = package_manifest["before"]
    after_items = package_manifest["after"]
    if [item.get("capability_id") for item in after_items] != capability_ids:
        _fail("package capability inventory changed")
    package_files = [before_item, *after_items]
    transitions = []
    for index, item in enumerate(package_files):
        path = run_dir / "packages" / str(item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            _fail("package identity changed")
        package, transition = load_package(path)
        if package["transition_sha256"] != item["transition_sha256"]:
            _fail("package transition hash changed")
        transitions.append(transition)
        if index > 0:
            training = receipt["teacher_training"][index - 1]
            if (
                package["provenance"]["teacher_before_sha256"] != training["teacher_before_sha256"]
                or package["provenance"]["teacher_after_sha256"] != training["teacher_after_sha256"]
                or sha256_bytes(transition_bytes(transition)) != training["teacher_after_sha256"]
            ):
                _fail("package does not bind teacher learning state")
    actual_packages = {path.name for path in (run_dir / "packages").glob("*.abipkg")}
    if actual_packages != {str(item["path"]) for item in package_files}:
        _fail("undeclared package file present")
    before = transitions[0]
    after = transitions[1:]

    teacher_execution = receipt["teacher_execution"]
    recipient_execution = receipt["recipient_execution"]
    teacher_reference = teacher_execution["observations"]
    recipient_reference = recipient_execution["observations"]
    teacher_rows = _jsonl(
        run_dir / teacher_reference["path"],
        teacher_reference["sha256"],
        teacher_reference["rows"],
    )
    recipient_rows = _jsonl(
        run_dir / recipient_reference["path"],
        recipient_reference["sha256"],
        recipient_reference["rows"],
    )
    conditions = list(config["conditions"])
    per_host_rows = len(expected) * len(conditions)
    if len(teacher_rows) != per_host_rows or len(recipient_rows) != per_host_rows * len(
        config["recipient_hosts"]
    ):
        _fail("observation matrix depth changed")
    host_receipts = [teacher_execution["host"], *recipient_execution["hosts"]]
    token_maps = {}
    for host_receipt in host_receipts:
        host = str(host_receipt["host"])
        token_ids = [int(value) for value in host_receipt["target_token_ids"]]
        if len(token_ids) != 8 or len(set(token_ids)) != 8:
            _fail(f"native token map invalid: {host}")
        if (
            host_receipt["model_state_sha256_before"] != host_receipt["model_state_sha256_after"]
            or host_receipt["codec_sha256_before"] != host_receipt["codec_sha256_after"]
            or host_receipt["recipient_optimizer_steps"] != 0
            or host_receipt["backend_learned_parameters"] != 0
        ):
            _fail(f"frozen native host boundary failed: {host}")
        token_maps[host] = token_ids
    if any(item.get("source_model_loaded") is not False for item in recipient_execution["hosts"]):
        _fail("source teacher was loaded by a recipient")

    executor = RecurrentTransitionNeuralISA()
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    all_rows = [*teacher_rows, *recipient_rows]
    for row in all_rows:
        if set(row) != FIELDS:
            _fail("raw observation schema changed")
        host = str(row["host"])
        capability_id = str(row["capability_id"])
        condition = str(row["condition"])
        row_id = str(row["row_id"])
        key = (host, capability_id, condition, row_id)
        if (
            key in index
            or host not in token_maps
            or capability_id not in capability_ids
            or condition not in conditions
        ):
            _fail("raw observation identity changed")
        reference = expected.get(row_id)
        if reference is None or row["prompt_sha256"] != reference["prompt_sha256"]:
            _fail("prompt commitment changed")
        canonical = canonical_prediction(int(row["prediction_token_id"]), token_maps[host])
        if canonical != row["canonical_prediction"]:
            _fail("authoritative native token accounting changed")
        _probabilities(row["canonical_probabilities"], "canonical probabilities")
        neural_state = _probabilities(
            row["neural_state"],
            "neural state",
            normalized=condition != "ZERO",
        )
        capability_index = capability_ids.index(capability_id)
        wrong_index = (capability_index + 1) % len(after)
        condition_transitions = _conditions(
            before, after[capability_index], after[wrong_index], capability_id
        )
        active = condition not in {
            "BASE",
            "REMOVED",
            "BACKEND_REMOVED",
            "CODEC_REMOVED",
        }
        if row["backend_active"] is not active or row["codec_active"] is not active:
            _fail("neural backend/codec intervention accounting changed")
        expected_hash = None
        if condition in {"AFTER", "RESTORED", "BACKEND_REMOVED", "CODEC_REMOVED"}:
            expected_hash = after_items[capability_index]["sha256"]
        elif condition == "BEFORE":
            expected_hash = before_item["sha256"]
        elif condition == "WRONG":
            expected_hash = after_items[wrong_index]["sha256"]
        elif condition in {"ZERO", "RANDOM", "SHUFFLED"}:
            expected_hash = "CONTROL_" + condition
        if row["package_sha256"] != expected_hash:
            _fail("package invocation hash changed")
        if active:
            recomputed = executor(condition_transitions[condition], [str(reference["prompt"])])[
                0
            ].tolist()
            if not _close(neural_state, recomputed):
                _fail("neural ISA state does not recompute")
        try:
            bytes.fromhex(str(row["canonical_output_utf8_hex"]))
        except ValueError as exc:
            raise R11VerificationError("canonical output byte encoding changed") from exc
        index[key] = row
        groups[(host, capability_id, condition)].append(row)

    gates = config["gates"]
    teacher_metrics = {}
    recipient_metrics = {}
    teacher_pass = True
    recipient_pass = True
    teacher_host = "source"
    for capability_id in capability_ids:
        before_accuracy = _accuracy(groups[(teacher_host, capability_id, "BEFORE")], answers)
        after_accuracy = _accuracy(groups[(teacher_host, capability_id, "AFTER")], answers)
        gain = after_accuracy - before_accuracy
        teacher_metrics[capability_id] = {
            "BEFORE": before_accuracy,
            "AFTER": after_accuracy,
            "gain": gain,
        }
        teacher_pass &= (
            before_accuracy <= gates["teacher_before_accuracy_maximum"]
            and after_accuracy == gates["teacher_after_accuracy"]
            and gain >= gates["teacher_gain_minimum"]
        )
    negative = {"BEFORE", "WRONG", "ZERO", "RANDOM", "SHUFFLED"}
    for host in config["recipient_hosts"]:
        recipient_metrics[host] = {}
        for capability_id in capability_ids:
            values = {
                condition: _accuracy(groups[(host, capability_id, condition)], answers)
                for condition in conditions
            }
            recipient_metrics[host][capability_id] = values
            recipient_pass &= values["AFTER"] == 1.0 and values["RESTORED"] == 1.0
            recipient_pass &= all(
                values[condition] <= gates["negative_control_accuracy_maximum"]
                for condition in negative
            )
            for row_id, reference in expected.items():
                if reference["capability_id"] != capability_id:
                    continue
                teacher_after = index[(teacher_host, capability_id, "AFTER", row_id)]
                for condition in ("AFTER", "RESTORED"):
                    student = index[(host, capability_id, condition, row_id)]
                    if (
                        student["canonical_output_utf8_hex"]
                        != teacher_after["canonical_output_utf8_hex"]
                    ):
                        _fail("recipient output is not losslessly teacher-equivalent")
                base = index[(host, capability_id, "BASE", row_id)]
                ignored = {"condition", "package_sha256", "backend_active", "codec_active"}
                base_comparable = {k: v for k, v in base.items() if k not in ignored}
                for condition in ("REMOVED", "BACKEND_REMOVED", "CODEC_REMOVED"):
                    removed = index[(host, capability_id, condition, row_id)]
                    if {k: v for k, v in removed.items() if k not in ignored} != base_comparable:
                        _fail(f"{condition} does not exactly restore BASE")
                restored = index[(host, capability_id, "RESTORED", row_id)]
                after_row = index[(host, capability_id, "AFTER", row_id)]
                if {**restored, "condition": "AFTER"} != after_row:
                    _fail("identical package bytes did not exactly restore output")

    times = (
        float(teacher_execution["started"]),
        float(teacher_execution["finished"]),
        float(recipient_execution["started"]),
        float(recipient_execution["finished"]),
    )
    temporal_pass = (
        all(math.isfinite(value) for value in times) and times[0] < times[1] <= times[2] < times[3]
    )
    passed = bool(teacher_pass and recipient_pass and temporal_pass)
    result = {
        "format": "abi-native-neural-isa-r11-verification/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": "LOSSLESS_LEVEL_1_OUTPUT_EQUIVALENCE",
        "claim_ceiling": "SYNTHETIC_ABI_NATIVE_TEACHER_ONLY",
        "teacher_metrics": teacher_metrics,
        "recipient_metrics": recipient_metrics,
        "gates": {
            "teacher_learning": bool(teacher_pass),
            "recipient_lossless_output_equivalence": bool(recipient_pass),
            "teacher_absent_after_learning": bool(temporal_pass),
        },
        "stronger_claims_open": [
            "arbitrary_open_weight_teacher",
            "english_or_natural_domain_transfer",
            "decision_or_distribution_equivalence",
            "information_minimality",
            "layercake_product_integration",
        ],
        "receipt_evidence_sha256": receipt["evidence_sha256"],
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if not passed:
        _fail("one or more recomputed R11 gates failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = verify(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.run_dir).resolve(),
        )
        output = Path(args.output).resolve()
        if output.exists():
            raise R11VerificationError(f"immutable verification exists: {output}")
        output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
