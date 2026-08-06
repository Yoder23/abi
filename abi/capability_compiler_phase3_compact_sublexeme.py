"""Training-derived compact-sublexeme representation bake-off."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_direct_core import _json
from .capability_compiler_phase3_representation_bakeoff import inventory, rows, tokenizer_type

FORMAT = "abi-capability-compiler-phase3-compact-sublexeme/1"


def load_protocol(root: Path, path: Path):
    p = _json(path)
    if p.get("format") != FORMAT or p.get("status") != "PREREGISTERED_NO_TRAINING" or p.get("training_allowed") is not False or p.get("final_test_access") != "PROHIBITED": raise Phase3Error("V32 governance changed")
    for relative, expected in p["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"V32 binding changed: {relative}")
    return p, sha256_file(path)


def ranked_sublexemes(training, split, existing, *, minimum_length, maximum_length):
    counts = Counter()
    for row in training:
        for piece in split(row["output"]):
            text = piece.decode("utf-8", errors="strict")
            observed = set()
            for start in range(len(text)):
                for width in range(minimum_length, min(maximum_length, len(text) - start) + 1):
                    value = text[start:start + width].encode("utf-8")
                    if value not in existing: observed.add(value)
            counts.update(observed)
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], -len(item[0].decode("utf-8")), item[0]))]


def segment(piece: bytes, vocabulary: set[bytes]):
    text = piece.decode("utf-8", errors="strict"); n = len(text); best = [None] * (n + 1); best[0] = []
    for end in range(1, n + 1):
        choices = []
        for start in range(end):
            value = text[start:end].encode("utf-8")
            if best[start] is not None and value in vocabulary: choices.append(best[start] + [value])
        if choices: best[end] = min(choices, key=lambda values: (len(values), [value.hex() for value in values]))
    if best[n] is None: raise Phase3Error("sublexeme vocabulary lost universal coverage")
    return best[n]


def evaluate(dataset, split, full_lexemes, fallback):
    totals = Counter(); stream = hashlib.sha256(); failing = []
    for row in dataset:
        source = split(row["prompt"]); source_counts = Counter(source); actions = []
        for piece in split(row["output"]):
            if source_counts[piece] == 1: actions.append(("pointer", piece))
            elif piece in full_lexemes: actions.append(("lexeme", piece))
            else: actions.extend(("sublexeme", value) for value in segment(piece, fallback))
        count = len(actions) + 1; totals.update(records=1, actions=count, over_320=int(count > 320));
        if count > 320: failing.append({"record_id": row["record_id"], "capability": row["capability"], "actions": count})
        stream.update(canonical_json_bytes({"record_id": row["record_id"], "actions": [(kind, value.hex()) for kind, value in actions]}))
    return {**dict(totals), "mean_actions": totals["actions"] / totals["records"], "maximum_actions": max([row["actions"] for row in failing], default=0), "failing": failing, "stream_sha256": stream.hexdigest()}


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    p, protocol_sha = load_protocol(root, protocol_path); training, development = rows(root, p); tokenizer = tokenizer_type(root, p)
    full, characters = inventory(training, tokenizer.split); ascii_set = {bytes((v,)) for v in range(0x20, 0x7F)}; full |= ascii_set; characters |= ascii_set
    ranked = ranked_sublexemes(training, tokenizer.split, full | characters, minimum_length=p["sublexemes"]["minimum_characters"], maximum_length=p["sublexemes"]["maximum_characters"])
    candidates = {}
    for budget in p["sublexemes"]["budgets"]:
        added = set(ranked[:budget]); fallback = characters | added
        train = evaluate(training, tokenizer.split, full, fallback); dev = evaluate(development, tokenizer.split, full, fallback)
        fixed = len(full | fallback) + 4; qualifies = train["over_320"] == 0 and dev["over_320"] == 0 and fixed <= p["qualification"]["maximum_fixed_actions"]
        candidates[str(budget)] = {"budget": budget, "added_sublexemes": len(added), "fixed_actions": fixed, "training": train, "development": dev, "qualifying": qualifies, "vocabulary_sha256": hashlib.sha256(b"".join(value + b"\0" for value in sorted(added))).hexdigest()}
    passing = [value for value in candidates.values() if value["qualifying"]]; selected = min(passing, key=lambda value: value["budget"])["budget"] if passing else None
    result = {"format": "abi-capability-compiler-phase3-compact-sublexeme-result/1", "status": "PASS_REPRESENTATION_ONLY" if selected else "FAIL_REPRESENTATION", "protocol": {"path": protocol_path.name, "sha256": protocol_sha}, "training_performed": False, "final_test_accessed": False, "phase3_certified": False, "candidates": candidates, "selected_budget": selected, "model_training_authorized": False, "host_change_authorized": False, "claim_boundary": "Training-derived representation efficiency only; no learned quality, host execution, Phase 3 pass, or superiority claim."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); return result


def main(argv: Iterable[str] | None = None) -> int:
    a=argparse.ArgumentParser(description=__doc__); a.add_argument("command",choices=("execute","verify")); a.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_COMPACT_SUBLEXEME_PROTOCOL_V32.json"); a.add_argument("--output",default="results/abi_capability_compiler_phase3_compact_sublexeme/compact_sublexeme_v32.json"); args=a.parse_args(argv); root=Path.cwd().resolve(); expected=execute(root,(root/args.protocol).resolve()); output=(root/args.output).resolve()
    if args.command=="execute":
        if output.exists(): raise Phase3Error(f"V32 output is immutable: {output}")
        _write_immutable(output,json.dumps(expected,indent=2,sort_keys=True).encode("utf-8")+b"\n")
    elif _json(output)!=expected: raise Phase3Error("stored V32 result differs from recomputation")
    print(json.dumps({"status":expected["status"],"selected_budget":expected["selected_budget"],"evidence_sha256":expected["evidence_sha256"]},indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
