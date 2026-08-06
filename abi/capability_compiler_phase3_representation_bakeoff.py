"""No-training Unicode-atomic open-vocabulary representation bake-off."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_direct_core import _json


FORMAT = "abi-capability-compiler-phase3-representation-bakeoff/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_TRAINING"
        or protocol.get("training_allowed") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("representation bake-off governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"representation bake-off binding changed: {relative}")
    return protocol, sha256_file(path)


def rows(root: Path, protocol: Mapping[str, Any]):
    training = [
        {
            "record_id": str(row["ir_record_id"]),
            "capability": str(row["capability"]),
            "prompt": str(row["normalized_acquisition_prompt"]),
            "output": str(row["normalized_output"]),
        }
        for row in load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    ]
    teacher = {
        str(row["probe_id"]): row
        for row in (json.loads(line) for line in (root / protocol["teacher_reference"]).read_text(encoding="utf-8").splitlines() if line)
    }
    development = []
    for probe in development_probes((root / protocol["development_catalog"]).resolve()):
        probe_id = str(probe["probe_id"])
        development.append({
            "record_id": probe_id,
            "capability": str(probe["canonical_capability"]),
            "prompt": str(probe["prompt"]),
            "output": str(teacher[probe_id]["output"]),
        })
    if len(training) != 7000 or len(development) != 1400:
        raise Phase3Error("representation bake-off depth changed")
    return training, development


def tokenizer_type(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake_extensions.unicode_direct_neural_core import UnicodeAtomicLexemePointerTokenizer
    return UnicodeAtomicLexemePointerTokenizer


def spell_selected(piece: bytes, modulus: int) -> bool:
    return int.from_bytes(hashlib.sha256(piece).digest()[:8], "big") % modulus == 0


def inventory(training: list[Mapping[str, Any]], split) -> tuple[set[bytes], set[bytes]]:
    lexemes: set[bytes] = set()
    characters: set[bytes] = set()
    for row in training:
        lexemes.update(split(row["prompt"]))
        lexemes.update(split(row["output"]))
        characters.update(char.encode("utf-8") for char in row["output"])
    return lexemes, characters


def encode_row(row: Mapping[str, Any], split, lexemes: set[bytes], characters: set[bytes], *, mode: str, exposure_modulus: int | None):
    source = split(row["prompt"])
    output = split(row["output"])
    source_counts = Counter(source)
    actions: list[tuple[str, bytes]] = []
    for piece in output:
        if mode != "character_only" and source_counts[piece] == 1:
            actions.append(("pointer", piece))
        elif mode == "character_only" or piece not in lexemes or (exposure_modulus is not None and spell_selected(piece, exposure_modulus)):
            for char in piece.decode("utf-8", errors="strict"):
                encoded = char.encode("utf-8")
                if encoded not in characters:
                    raise KeyError(encoded)
                actions.append(("character", encoded))
        else:
            actions.append(("lexeme", piece))
    reconstructed = b"".join(payload for _, payload in actions)
    if reconstructed != str(row["output"]).encode("utf-8"):
        raise Phase3Error(f"representation is not lossless: {row['record_id']}")
    return source, actions


def evaluate(rows_: list[Mapping[str, Any]], split, lexemes: set[bytes], characters: set[bytes], candidate: Mapping[str, Any]):
    totals = Counter()
    character_actions = Counter()
    per_capability = {capability: Counter() for capability in CAPABILITIES}
    action_stream = hashlib.sha256()
    rejected = []
    for row in rows_:
        try:
            source, actions = encode_row(row, split, lexemes, characters, mode=candidate["mode"], exposure_modulus=candidate.get("exposure_modulus"))
        except (KeyError, UnicodeDecodeError) as exc:
            rejected.append({"record_id": row["record_id"], "capability": row["capability"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        counts = Counter(kind for kind, _ in actions)
        character_actions.update(value.hex() for kind, value in actions if kind == "character")
        values = {
            "records": 1,
            "actions": len(actions) + 1,
            "pointer_actions": counts["pointer"],
            "lexeme_actions": counts["lexeme"],
            "character_actions": counts["character"],
            "records_with_character_actions": int(counts["character"] > 0),
            "source_over_limit": int(len(source) > 128),
            "target_over_limit": int(len(actions) + 1 > 320),
        }
        totals.update(values); per_capability[row["capability"]].update(values)
        action_stream.update(canonical_json_bytes({"record_id": row["record_id"], "actions": [(kind, value.hex()) for kind, value in actions]}))
    return {
        **dict(totals),
        "representable": len(rows_) - len(rejected),
        "rejected": rejected,
        "mean_actions": totals["actions"] / totals["records"] if totals["records"] else None,
        "character_action_histogram": dict(sorted(character_actions.items())),
        "action_stream_sha256": action_stream.hexdigest(),
        "per_capability": {name: dict(values) for name, values in per_capability.items()},
    }


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    training, development = rows(root, protocol)
    tokenizer = tokenizer_type(root, protocol)
    lexemes, characters = inventory(training, tokenizer.split)
    candidates = {}
    for specification in protocol["candidates"]:
        name = specification["name"]
        train = evaluate(training, tokenizer.split, lexemes, characters, specification)
        dev = evaluate(development, tokenizer.split, lexemes, characters, specification)
        dev_needed_chars = {char.encode("utf-8") for row in development for char in row["output"]}
        unsupported_chars = sorted(value.hex() for value in dev_needed_chars - characters)
        dev_character_training_counts = {value.hex(): train["character_action_histogram"].get(value.hex(), 0) for value in sorted(dev_needed_chars)}
        minimum_dev_character_training_actions = min(dev_character_training_counts.values()) if dev_character_training_counts else 0
        total_fixed_actions = len(lexemes | characters) + 4
        qualifying = (
            train["representable"] == 7000 and dev["representable"] == 1400
            and train["source_over_limit"] == 0 and dev["source_over_limit"] == 0
            and train["target_over_limit"] == 0 and dev["target_over_limit"] == 0
            and not unsupported_chars
            and minimum_dev_character_training_actions >= protocol["qualification"]["minimum_training_actions_for_each_development_character"]
            and total_fixed_actions <= protocol["qualification"]["maximum_fixed_actions"]
            and all(value.decode("utf-8", errors="strict") is not None for value in lexemes | characters)
        )
        candidates[name] = {
            "mode": specification["mode"],
            "exposure_modulus": specification.get("exposure_modulus"),
            "fixed_lexemes": len(lexemes),
            "fixed_characters": len(characters),
            "total_fixed_actions": total_fixed_actions,
            "unsupported_development_characters": unsupported_chars,
            "minimum_training_actions_for_each_development_character": minimum_dev_character_training_actions,
            "training": train,
            "development": dev,
            "qualifying": qualifying,
        }
    qualified = [name for name, value in candidates.items() if value["qualifying"]]
    selected = min(qualified, key=lambda name: (candidates[name]["training"]["mean_actions"], candidates[name]["development"]["mean_actions"], name)) if qualified else None
    result: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-representation-bakeoff-result/1",
        "status": "PASS_NO_TRAINING" if selected else "FAIL_NO_QUALIFYING_REPRESENTATION",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "phase4_through_8": "LOCKED",
        "inventory": {"training_lexemes": len(lexemes), "training_output_characters": len(characters)},
        "candidates": candidates,
        "selection_rule": protocol["selection_rule"],
        "selected": selected,
        "decision": "REPRESENTATION_ONLY_NO_MODEL_AUTHORIZED",
        "layercake_host_changed": False,
        "claim_boundary": "Target representability and action-count result only. It proves no learned quality, performance, transfer, Phase 3 pass, or ABI superiority."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_REPRESENTATION_BAKEOFF_PROTOCOL_V28.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_representation_bakeoff/representation_bakeoff_v28.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); protocol = (root / args.protocol).resolve(); output = (root / args.output).resolve()
    expected = execute(root, protocol)
    if args.command == "execute":
        if output.exists(): raise Phase3Error(f"V28 output is immutable: {output}")
        _write_immutable(output, json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    elif _json(output) != expected:
        raise Phase3Error("stored V28 result differs from recomputation")
    print(json.dumps({"status": expected["status"], "selected": expected["selected"], "evidence_sha256": expected["evidence_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
