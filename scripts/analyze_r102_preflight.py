"""Analyze the recorded proposal pool without new random draws or model work."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.coverage_exposure_r102 import core as c  # noqa: E402


def main():
    from transformers import AutoTokenizer

    directory = ROOT / "results/coverage_exposure_r102_preflight_v1"
    summary = c.read(directory / "summary.json")
    if c.sha(directory / "candidates.jsonl") != summary["candidate_sha256"]:
        raise RuntimeError("Preflight candidate pool changed")
    tokenizer = AutoTokenizer.from_pretrained(c.PARENT, local_files_only=True)
    training = c.read(c.previous.OUT / "train.json")
    old_dev = c.read(c.previous.OUT / "development.json") + c.read(c.previous.baseline.OUT / "development.json")
    occupied = {name for r in training + old_dev + c.development_catalog() for name in c.entities(r)}
    targets = {r["id"]: r["teacher_output"] for r in c.rows(c.previous.OUT / "source/rows.jsonl")}
    count = 0
    for row in training:
        for alias in range(4):
            if row["id"] == summary["row"]["id"] and alias == summary["alias"]:
                break
            c.construct_alias(row, targets[row["id"]], alias, tokenizer, occupied)
            count += 1
        else:
            continue
        break
    if count != summary["completed_aliases_before_failure"]:
        raise RuntimeError("Preflight history changed")
    row, answer = summary["row"], summary["target"]
    pc = len(tokenizer.encode(c.model_prompt(row, "distillation"), add_special_tokens=False))
    tc = len(tokenizer.encode(answer, add_special_tokens=False))
    choices = {"NOTE": [], "Person": []}
    for candidate in c.rows(directory / "candidates.jsonl"):
        for prefix in choices:
            mapping = {k: v for k, v in candidate["mapping"].items() if k.startswith(prefix)}
            derived, target = c.transform(row, mapping), c.replace_literals(answer, mapping)
            if (not set(mapping.values()) & occupied
                    and len(tokenizer.encode(c.model_prompt(derived, "distillation"), add_special_tokens=False)) == pc
                    and len(tokenizer.encode(target, add_special_tokens=False)) == tc):
                choices[prefix].append({"attempt": candidate["attempt"], "mapping": mapping})
    if not all(choices.values()):
        raise RuntimeError("No component-wise repair demonstrated")
    mapping = {**choices["NOTE"][0]["mapping"], **choices["Person"][0]["mapping"]}
    derived, target = c.transform(row, mapping), c.replace_literals(answer, mapping)
    if (len(tokenizer.encode(c.model_prompt(derived, "distillation"), add_special_tokens=False)) != pc
            or len(tokenizer.encode(target, add_special_tokens=False)) != tc
            or c.screen(derived, target) != c.screen(row, answer)
            or c.replace_literals(target, {v: k for k, v in mapping.items()}) != answer):
        raise RuntimeError("Combined repair violates original constraints")
    result = {"candidate_sha256": summary["candidate_sha256"], "component_matches": {k: len(v) for k, v in choices.items()},
              "selected_components": {k: v[0] for k, v in choices.items()}, "mapping": mapping,
              "prompt_tokens": pc, "target_tokens": tc, "new_random_draws": 0,
              "teacher_queries": 0, "optimizer_steps": 0,
              "interpretation": "Compatible components existed in the recorded pool; joint random pairing missed them"}
    c.write_once(directory / "analysis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
