"""Execute the fresh held-out R28 quorum acquisition."""

from __future__ import annotations

import argparse, gc, json, random, time
from pathlib import Path

import torch

from experiments.autonomous_labeling_r27.public_qualification import ACCEPTED_LABELS, _parse
from experiments.autonomous_labeling_r27.source_run import _control, _packages, _write_bundle
from experiments.factual_semantic_r16.public_qualification import _generate, _load_source, _render_chat
from experiments.foreign_capability_r14.core import R14Error, evidence_hash, sha256_file, write_json_once, write_jsonl_once
from .binding import load_config
from .facts import SYSTEM_PROMPT
from .isolation import run_isolated
from .protocol import answer_key, answer_packages, selected_facts, unique_quorum


def run(config_path: Path, reveal_path: Path, output: Path):
    root = Path(__file__).resolve().parents[2]
    if output.exists(): raise R14Error(f"immutable R28 output exists: {output}")
    config = load_config(root, config_path.resolve())
    if sha256_file(reveal_path) != config["reveal_sha256"]: raise R14Error("R28 reveal changed")
    reveal = json.loads(reveal_path.read_text(encoding="utf-8")); facts = selected_facts(reveal["secret_hex"], config["heldout_seed_commitment"], config["facts_per_domain"])
    output.mkdir(parents=True); tokenizer, model, snapshot = _load_source(config["source"]["model_id"], config["source"]["revision"]); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter(); rows = []; bundle = []; tokens = 0; output_bytes = 0
    for fact in facts:
        for split, questions in (("extraction", fact.extraction_questions), ("evaluation", fact.evaluation_questions)):
            for view, question in enumerate(questions):
                rendered = _render_chat(tokenizer, SYSTEM_PROMPT, question); completion, count = _generate(tokenizer, model, rendered, config["source"]["max_new_tokens"])
                try: response, label = _parse(completion); parsed = True
                except (ValueError, json.JSONDecodeError): response, label, parsed = "", "", False
                row = {"split": split, "fact_id": fact.fact_id, "oracle_domain": fact.oracle_domain, "subject": fact.subject, "view": view, "question": question, "oracle_answer": fact.answer, "completion": completion, "parse_exact": parsed, "teacher_answer": response, "answer_exact": answer_key(response) == answer_key(fact.answer), "free_label": label, "label_syntax_valid": bool(label and label.isascii() and label.isalpha() and label.islower()), "label_semantic_valid": label in ACCEPTED_LABELS[fact.oracle_domain]}
                rows.append(row)
                if split == "extraction": bundle.append({"subject": fact.subject, "question": question, "view": view, "answer": response, "label": label})
                tokens += count; output_bytes += len(completion.encode("utf-8"))
    del model, tokenizer; gc.collect(); torch.cuda.empty_cache(); random.Random(int(reveal["secret_hex"][:16], 16)).shuffle(bundle)
    rows_path = output / "source_rows.jsonl"; write_jsonl_once(rows_path, rows); bundle_path = output / "source_bundle.json"; _write_bundle(bundle_path, bundle)
    extraction = run_isolated(root, bundle_path, output / "extraction", config["physical_extraction"]["distribution"])
    control_rows, control_offset = _control(bundle); control_path = output / "control_bundle.json"; _write_bundle(control_path, control_rows); control = run_isolated(root, control_path, output / "control_extraction", config["physical_extraction"]["distribution"])
    packages = _packages(output / "extraction", extraction["result"]); controls = _packages(output / "control_extraction", control["result"])
    extraction_rows = [row for row in rows if row["split"] == "extraction"]; evaluation_rows = [row for row in rows if row["split"] == "evaluation"]
    fact_labels = {}; answer_quorums = {}; label_quorums = {}
    for fact in facts:
        selected = [row for row in extraction_rows if row["fact_id"] == fact.fact_id]
        answer_quorums[fact.fact_id] = unique_quorum([row["teacher_answer"] for row in selected], answer_key)
        label_quorums[fact.fact_id] = unique_quorum([row["free_label"] for row in selected])
        fact_labels[fact.subject] = label_quorums[fact.fact_id]
    package_domains = {package["namespace"]: {fact.oracle_domain for fact in facts if any(item["entity"] == fact.subject for item in package["facts"])} for package in packages}
    evaluation = []
    for row in evaluation_rows:
        namespace = f"teacher/{fact_labels[row['subject']]}"; selected = [item for item in packages if item["namespace"] == namespace]; others = [item for item in packages if item["namespace"] != namespace]; prediction = answer_packages(selected, row["question"])
        evaluation.append({"fact_id": row["fact_id"], "oracle_domain": row["oracle_domain"], "subject": row["subject"], "view": row["view"], "question": row["question"], "namespace": namespace, "oracle_answer": row["oracle_answer"], "teacher_answer": row["teacher_answer"], "package_answer": prediction, "other_answer": answer_packages(others, row["question"]), "control_answer": answer_packages(controls, row["question"])})
    evaluation_path = output / "evaluation.jsonl"; write_jsonl_once(evaluation_path, evaluation)
    metrics = {"selected_facts": len(facts), "source_rows": len(rows), "parse_exact": sum(row["parse_exact"] for row in rows), "source_answer_exact": sum(row["answer_exact"] for row in rows), "label_syntax_valid": sum(row["label_syntax_valid"] for row in rows), "label_semantic_valid": sum(row["label_semantic_valid"] for row in rows), "facts_with_answer_quorum": sum(value is not None for value in answer_quorums.values()), "facts_with_label_quorum": sum(value is not None for value in label_quorums.values()), "compiled_records": extraction["result"]["records_consumed"], "compiled_facts": sum(item["facts"] for item in extraction["result"]["packages"]), "discovered_labels": len(packages), "pure_discovered_labels": sum(len(domains) == 1 for domains in package_domains.values()), "package_oracle_exact": sum(answer_key(str(row["package_answer"] or "")) == answer_key(row["oracle_answer"]) for row in evaluation), "package_teacher_agreement": sum(answer_key(str(row["package_answer"] or "")) == answer_key(row["teacher_answer"]) for row in evaluation), "other_packages_abstain": sum(row["other_answer"] is None for row in evaluation), "control_oracle_exact": sum(answer_key(str(row["control_answer"] or "")) == answer_key(row["oracle_answer"]) for row in evaluation), "oracle_fields_consumed": extraction["result"]["oracle_fields_consumed"], "accepted_labels_consumed": extraction["result"]["accepted_labels_consumed"]}
    passed = metrics["selected_facts"] == 12 and metrics["source_rows"] == 72 and all(metrics[name] >= 68 for name in ("parse_exact", "source_answer_exact", "label_syntax_valid", "label_semantic_valid")) and metrics["facts_with_answer_quorum"] == 12 and metrics["facts_with_label_quorum"] == 12 and metrics["compiled_records"] == 36 and metrics["compiled_facts"] == 12 and metrics["discovered_labels"] >= 4 and metrics["pure_discovered_labels"] == metrics["discovered_labels"] and metrics["package_oracle_exact"] == 36 and metrics["package_teacher_agreement"] >= 32 and metrics["other_packages_abstain"] == 36 and metrics["control_oracle_exact"] / 36 <= config["max_control_accuracy"] and metrics["oracle_fields_consumed"] == 0 and metrics["accepted_labels_consumed"] == 0
    inventory = [{**item, "path": f"extraction/{item['path']}"} for item in extraction["result"]["packages"]]
    result = {"format": "abi-r28-heldout-quorum-source-result/1", "verdict": "PASS" if passed else "FAIL", "claim": "BOUNDED_ERROR_CORRECTING_FREE_LABEL_DISCOVERY_AND_COMPILATION", "claim_ceiling": "NOT_EXHAUSTIVE_TEACHER_DIAGNOSIS_OR_FLUENT_ENGLISH_EXTRACTION", "config_sha256": sha256_file(config_path), "reveal_sha256": sha256_file(reveal_path), "source": {"model_id": config["source"]["model_id"], "revision": config["source"]["revision"], "snapshot": str(snapshot), "training_steps": 0}, "metrics": metrics, "packages": inventory, "isolation": {"result_sha256": sha256_file(output / "extraction/result.json"), "launcher_sha256": sha256_file(output / "extraction/launcher.json"), "control_result_sha256": sha256_file(output / "control_extraction/result.json")}, "artifacts": {"source_rows": {"path": rows_path.name, "bytes": rows_path.stat().st_size, "sha256": sha256_file(rows_path)}, "source_bundle": {"path": bundle_path.name, "bytes": bundle_path.stat().st_size, "sha256": sha256_file(bundle_path)}, "evaluation": {"path": evaluation_path.name, "bytes": evaluation_path.stat().st_size, "sha256": sha256_file(evaluation_path)}}, "information_accounting": {"source_calls": len(rows), "generated_tokens": tokens, "output_bytes": output_bytes, "source_bundle_bytes": bundle_path.stat().st_size, "final_package_bytes": sum(item["bytes"] for item in inventory), "source_parameters_copied": 0, "bridge_parameters_trained": 0, "control_rotation_offset": control_offset, "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())}, "full_abi_moonshot": "OPEN"}; result["evidence_sha256"] = evidence_hash(result); write_json_once(output / "result.json", result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--reveal", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); print(json.dumps(run(args.config, args.reveal, args.output), indent=2))
if __name__ == "__main__": main()
