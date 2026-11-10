"""Once-only prospective source capture; training targets stay byte-identical."""

from __future__ import annotations

import time

from experiments.grounded_english_r99 import run as source_tools

from . import core as c


def acquire():
    import psutil
    import torch

    from experiments.factual_semantic_r16.public_qualification import _load_source, _render_chat

    c.binding()
    destination = c.OUT / "source"
    destination.mkdir()
    started = time.perf_counter()
    process = psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model, snapshot = _load_source(c.previous.SOURCE, c.previous.REVISION)
    expected_inventory = c.read(c.previous.r99.previous.OUT / "source/result.json")["accounting"]["original_acquisition"]["source_files"]
    inventory = {str(p.relative_to(snapshot)): {"sha256": c.sha(p), "bytes": p.stat().st_size}
                 for p in snapshot.rglob("*") if p.is_file()}
    if inventory != expected_inventory:
        raise RuntimeError("Pinned source model snapshot changed")
    prior_raw = {r["id"]: r for r in c.rows(c.previous.OUT / "source/raw.jsonl")}
    prior_normal = {r["id"]: r for r in c.rows(c.previous.OUT / "source/rows.jsonl")}
    population = c.read(c.OUT / "train.json") + c.read(c.OUT / "development.json")
    raw_rows, normalized_rows = [], []
    peak_rss = process.memory_info().rss
    with (destination / "raw.jsonl").open("x", encoding="utf-8") as raw_file, (
        destination / "rows.jsonl").open("x", encoding="utf-8") as normalized_file:
        for reference in population:
            if time.perf_counter() - started > 1800:
                raise RuntimeError("R102 source time budget")
            if reference["split"] == "train":
                raw, normalized = prior_raw[reference["id"]], prior_normal[reference["id"]]
            else:
                rendered = _render_chat(tokenizer, c.SYSTEM, reference["prompt"])
                encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
                before = time.perf_counter()
                with torch.inference_mode():
                    result = model.generate(**encoded, do_sample=False, max_new_tokens=384,
                                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                tokens = result[0, encoded.input_ids.shape[1]:].tolist()
                text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                raw = {**{key: reference[key] for key in ("id", "split", "task", "prompt")},
                       "teacher_output": text, "token_ids": tokens, "input_tokens": int(encoded.input_ids.numel()),
                       "wall_seconds": time.perf_counter() - before,
                       "truncated": len(tokens) == 384 and tokens[-1] != tokenizer.eos_token_id,
                       "screen": c.screen(reference, text)}
                selected = c.previous.normalize(reference, text)
                normalized = {**raw, "teacher_output": selected["text"], "normalization": selected,
                              "token_ids": tokenizer.encode(selected["text"], add_special_tokens=False),
                              "raw_source_token_count": len(tokens), "screen": c.screen(reference, selected["text"])}
                print(c.previous.canonical({"stage": "source", "completed": len(raw_rows)-191}).decode().strip(), flush=True)
            c.append(raw_file, raw)
            c.append(normalized_file, normalized)
            raw_rows.append(raw)
            normalized_rows.append(normalized)
            peak_rss = max(peak_rss, process.memory_info().rss)
    new = raw_rows[192:]
    c.write_once(destination / "receipt.json", {
        "raw_sha256": c.sha(destination / "raw.jsonl"), "normalized_sha256": c.sha(destination / "rows.jsonl"),
        "source_inventory": inventory, "summary": c.aggregate(normalized_rows), "raw_summary": c.aggregate(raw_rows),
        "new_queries": len(new), "input_tokens": sum(r["input_tokens"] for r in new),
        "generated_tokens": sum(len(r["token_ids"]) for r in new),
        "unique_prompt_utf8_bytes": sum(len(p.encode()) for p in {r["prompt"] for r in new}),
        "teacher_output_utf8_bytes": sum(len(r["teacher_output"].encode()) for r in new),
        "inference_seconds": sum(r["wall_seconds"] for r in new), "wall_seconds": time.perf_counter()-started,
        "cpu_process_seconds": sum(process.cpu_times()[:2])-cpu_start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "sampled_peak_rss_bytes": peak_rss,
        "source_parameters_loaded": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(), "precision": "float16",
        "source_parameters_copied": 0, "stored_logits": 0, "stored_activations": 0, "external_hardware": False,
        "inherited_source_costs": {"path": "results/acquisition_capacity_r100/source/result.json",
                                   "sha256": c.sha(c.previous.OUT / "source/result.json")},
        "separate_prior_r99_development_costs": {"path": "results/grounded_english_r99/source/result.json",
                                                 "sha256": c.sha(c.previous.r99.OUT / "source/result.json")},
    })


def verify():
    from experiments.factual_semantic_r16.public_qualification import _render_chat

    tokenizer = source_tools.tokenizer_from_snapshot()
    population = c.read(c.OUT / "train.json") + c.read(c.OUT / "development.json")
    directory = c.OUT / "source"
    receipt = c.read(directory / "receipt.json")
    if receipt["source_inventory"] != c.read(c.previous.r99.previous.OUT / "source/result.json")["accounting"]["original_acquisition"]["source_files"]:
        raise RuntimeError("Source inventory changed")
    if (c.sha(directory / "raw.jsonl") != receipt["raw_sha256"]
            or c.sha(directory / "rows.jsonl") != receipt["normalized_sha256"]):
        raise RuntimeError("Source raw payload changed")
    raw, normalized = c.rows(directory / "raw.jsonl"), c.rows(directory / "rows.jsonl")
    if len(raw) != 288 or len(normalized) != 288:
        raise RuntimeError("Incomplete source capture")
    prior_raw = {r["id"]: r for r in c.rows(c.previous.OUT / "source/raw.jsonl")}
    prior_normal = {r["id"]: r for r in c.rows(c.previous.OUT / "source/rows.jsonl")}
    vocab = len(tokenizer)
    for ref, original, selected in zip(population, raw, normalized):
        for field in ("id", "task", "split", "prompt"):
            if original[field] != ref[field] or selected[field] != ref[field]:
                raise RuntimeError("Source coverage mismatch")
        if ref["split"] == "train":
            if original != prior_raw[ref["id"]] or selected != prior_normal[ref["id"]]:
                raise RuntimeError("Inherited training data changed")
        else:
            tokens = original["token_ids"]
            if not tokens or len(tokens) > 384 or any(type(t) is not int or not 0 <= t < vocab for t in tokens):
                raise RuntimeError("Invalid teacher output tokens")
            if original["truncated"] != (len(tokens) == 384 and tokens[-1] != tokenizer.eos_token_id):
                raise RuntimeError("Incorrect source truncation")
            rendered = _render_chat(tokenizer, c.SYSTEM, ref["prompt"])
            if original["input_tokens"] != len(tokenizer.encode(rendered, add_special_tokens=False)):
                raise RuntimeError("Source input token accounting mismatch")
        if tokenizer.decode(original["token_ids"], skip_special_tokens=True).strip() != original["teacher_output"]:
            raise RuntimeError("Teacher token/text mismatch")
        expected = c.previous.normalize(ref, original["teacher_output"])
        if (selected["teacher_output"] != expected["text"] or selected["normalization"] != expected
                or selected["token_ids"] != tokenizer.encode(expected["text"], add_special_tokens=False)
                or selected["raw_source_token_count"] != len(original["token_ids"])
                or selected["truncated"] != original["truncated"]):
            raise RuntimeError("Unrecomputable source normalization")
        if original["screen"] != c.screen(ref, original["teacher_output"]) or selected["screen"] != c.screen(ref, selected["teacher_output"]):
            raise RuntimeError("Unrecomputable source screens")
    new = raw[192:]
    accounting = {"summary": c.aggregate(normalized), "raw_summary": c.aggregate(raw), "new_queries": len(new),
                  "input_tokens": sum(r["input_tokens"] for r in new), "generated_tokens": sum(len(r["token_ids"]) for r in new),
                  "unique_prompt_utf8_bytes": sum(len(p.encode()) for p in {r["prompt"] for r in new}),
                  "teacher_output_utf8_bytes": sum(len(r["teacher_output"].encode()) for r in new)}
    if any(receipt[k] != v for k, v in accounting.items()):
        raise RuntimeError("Source accounting mismatch")
    return normalized
