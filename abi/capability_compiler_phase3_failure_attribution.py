"""Read-only attribution of Phase 3 shared-output failures.

This diagnostic does not train, mutate, or promote a candidate.  It compares
the exact V11 C0 lineage, matched controls, and the sealed native LayerCake
parent on the same cached teacher continuations.  Its purpose is to decide
whether the next repair belongs to acquisition, integration/state recovery,
or a separately governed LayerCake host investigation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    repetition_collapse,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3_shared_output import (
    load_candidate,
    load_protocol as load_v11_protocol,
)
from .layercake_core_loader import load_layercake_core


FORMAT = "abi-capability-compiler-phase3-failure-attribution/1"
PROTOCOL_FORMAT = "abi-capability-compiler-phase3-failure-attribution-protocol/1"
SYSTEMS = ("P0", "C0", "C1", "C3", "C4")
HORIZONS = (1, 2, 4, 8, 16)


class DiagnosticError(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticError(f"expected JSON object: {path}")
    return value


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        raise DiagnosticError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_NO_PROMOTION"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("training_allowed") is not False
        or tuple(protocol.get("systems", ())) != SYSTEMS
        or tuple(protocol.get("corruption_recovery", {}).get("horizons", ())) != HORIZONS
    ):
        raise DiagnosticError("failure-attribution governance changed")
    for relative, expected in protocol.get("bindings", {}).items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise DiagnosticError(f"failure-attribution binding changed: {relative}")
    return protocol, sha256_file(path)


def classify_attribution(metrics: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    """Apply preregistered, non-exclusive attribution rules."""

    external = metrics["external_layercake_control"]
    if not external.get("identity_pass") or not external.get("sealed_certificates_pass"):
        return {
            "primary": "LAYERCAKE_HOST_REGRESSION_OR_IDENTITY_FAILURE",
            "owners": ["LAYERCAKE"],
            "next_action": "stop ABI experiments and investigate the sealed host separately",
            "layercake_regression": True,
        }

    systems = metrics["systems"]
    c0 = systems["C0"]["teacher_forced"]
    c3 = systems["C3"]["teacher_forced"]
    signal = (
        c0["mean_nll"] <= c3["mean_nll"] * (1.0 - float(rules["minimum_nll_reduction_vs_no_payload"]))
        and c0["token_accuracy"] >= c3["token_accuracy"] + float(rules["minimum_accuracy_gain_vs_no_payload"])
    )
    recovery = metrics["c0_corruption_recovery"]["aggregate"]
    prefix_sensitive = (
        recovery["nll_ratio_corrupted_to_clean"] >= float(rules["minimum_corrupted_nll_ratio"])
        and recovery["accuracy_delta_corrupted_minus_clean"] <= -float(rules["minimum_corrupted_accuracy_drop"])
    )
    native = systems["P0"]["autonomous"]
    scope_gap = (
        native["functional_rate"] < float(rules["native_new_suite_functional_floor"])
        or native["repetition_collapses"] > 0
    )
    owners: list[str] = []
    reasons: list[str] = []
    if signal:
        reasons.append("cached teacher payload is measurable against the no-payload control")
    else:
        owners.append("ABI")
        reasons.append("cached teacher payload does not clear the preregistered information-signal floor")
    if prefix_sensitive:
        owners.append("INTEGRATION")
        reasons.append("one model-generated wrong prefix token materially degrades continuation recovery")
    if scope_gap:
        owners.append("LAYERCAKE_SCOPE_REVIEW")
        reasons.append("the sealed native host has a non-regression scope gap on the broader ABI development suite")

    if signal and prefix_sensitive:
        primary = "ABI_SIGNAL_PRESENT_INTEGRATION_STATE_RECOVERY_LIMITING"
        action = "preregister one self-prefix-recovery bridge successor without modifying the sealed host"
    elif not signal:
        primary = "ABI_ACQUISITION_INFORMATION_DEFICIT"
        action = "repair the ABI representation or supervision before changing LayerCake"
    else:
        primary = "HOST_CAPACITY_OR_DECODING_LIMIT_UNRESOLVED"
        action = "run a separately preregistered oracle-fit capacity control before changing LayerCake"
    return {
        "primary": primary,
        "owners": sorted(set(owners)),
        "next_action": action,
        "layercake_regression": False,
        "native_new_suite_scope_gap": scope_gap,
        "abi_teacher_payload_signal": signal,
        "autonomous_prefix_sensitivity": prefix_sensitive,
        "host_representational_ceiling_proven": False,
        "reasons": reasons,
    }


def _external_control(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    cfg = protocol["external_layercake_control"]
    lc_root = (root / cfg["repository"]).resolve()
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={lc_root}", "rev-parse", "HEAD"],
        cwd=lc_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint = (root / cfg["checkpoint_path"]).resolve()
    certificates = []
    for item in cfg["certificates"]:
        path = (root / item["path"]).resolve()
        certificates.append(sha256_file(path) == item["sha256"])
    return {
        "repository_commit": commit,
        "required_repository_commit": cfg["repository_commit"],
        "identity_pass": commit == cfg["repository_commit"] and sha256_file(checkpoint) == cfg["checkpoint_sha256"],
        "sealed_certificates_pass": all(certificates),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _teacher_rows(root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    rows = []
    with (root / protocol["teacher_reference"]).resolve().open("r", encoding="utf-8") as stream:
        for line in stream:
            teacher = json.loads(line)
            probe_id = str(teacher["probe_id"])
            probe = probe_by_id.get(probe_id)
            if probe is None:
                raise DiagnosticError(f"teacher row has no development probe: {probe_id}")
            rows.append({"teacher": teacher, "probe": probe})
    if len(rows) != 1400 or len({row["teacher"]["probe_id"] for row in rows}) != 1400:
        raise DiagnosticError("teacher/development row count changed")
    per_capability = int(protocol.get("diagnostic_prompts_per_capability", 100))
    if per_capability == 100:
        return rows
    selected = []
    counts = Counter()
    for row in rows:
        capability = str(row["teacher"]["capability"])
        if counts[capability] < per_capability:
            selected.append(row)
            counts[capability] += 1
    if len(selected) != per_capability * len(CAPABILITIES) or set(counts) != set(CAPABILITIES):
        raise DiagnosticError("balanced diagnostic subset construction failed")
    return selected


def _encoded(row: Mapping[str, Any], tokenizer: Any, max_tokens: int) -> tuple[list[int], int]:
    prompt = str(row["probe"]["prompt"]).rstrip() + "\n"
    prompt_ids = [int(v) for v in tokenizer.encode(prompt, add_special_tokens=False)]
    response_ids = [int(v) for v in tokenizer.encode(str(row["teacher"]["output"]), add_special_tokens=False)]
    response_ids.append(int(tokenizer.eos_token_id))
    ids = prompt_ids + response_ids
    if len(ids) > max_tokens:
        raise DiagnosticError(f"teacher continuation exceeds host context: {row['teacher']['probe_id']}")
    return ids, len(prompt_ids)


def _forward_batch(model: Any, entries: list[tuple[list[int], int]], device: torch.device) -> list[torch.Tensor]:
    """Run exact right-padded causal rows and return unpadded logits."""

    lengths = [len(ids) for ids, _ in entries]
    maximum = max(lengths)
    # Padding is causally invisible and excluded by the exact attention mask.
    pad_id = 0
    ids = torch.full((len(entries), maximum), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(entries), maximum), dtype=torch.long, device=device)
    prompt_lengths = torch.tensor([prompt for _, prompt in entries], dtype=torch.long, device=device)
    for index, ((values, _), length) in enumerate(zip(entries, lengths)):
        ids[index, :length] = torch.tensor(values, dtype=torch.long, device=device)
        attention[index, :length] = 1
    with torch.autocast("cuda", dtype=torch.float16):
        result = model(ids, attention_mask=attention, prompt_lengths=prompt_lengths, use_cache=False)
    return [result["logits"][index, :length].float() for index, length in enumerate(lengths)]


@torch.inference_mode()
def _teacher_forced(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_prompt = []
    by_cap: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0
    repeat_errors = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        encoded = [_encoded(row, tokenizer, int(model.config.max_tokens)) for row in batch_rows]
        outputs = _forward_batch(model, encoded, device)
        for row, (ids, prompt_len), full_logits in zip(batch_rows, encoded, outputs):
            logits = full_logits[prompt_len - 1 : -1]
            targets = torch.tensor(ids[prompt_len:], dtype=torch.long, device=device)
            losses = F.cross_entropy(logits, targets, reduction="none")
            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(targets)
            repeated = torch.zeros_like(correct)
            for pos in range(len(targets)):
                if not bool(correct[pos]):
                    repeated[pos] = bool((targets[max(0, pos - 8) : pos] == predictions[pos]).any())
            item = {
                "probe_id": str(row["teacher"]["probe_id"]),
                "capability": str(row["teacher"]["capability"]),
                "tokens": len(targets),
                "nll_sum": float(losses.sum().item()),
                "correct_tokens": int(correct.sum().item()),
                "wrong_recent_repeat_predictions": int(repeated.sum().item()),
                "predictions": [int(v) for v in predictions.cpu().tolist()],
                "targets": [int(v) for v in targets.cpu().tolist()],
                "prompt_length": prompt_len,
                "input_ids": ids,
            }
            per_prompt.append(item)
            by_cap[item["capability"]].append(item)
            total_nll += item["nll_sum"]
            total_correct += item["correct_tokens"]
            total_tokens += item["tokens"]
            repeat_errors += item["wrong_recent_repeat_predictions"]
        if min(start + batch_size, len(rows)) % 200 == 0:
            print(json.dumps({"teacher_forced": min(start + batch_size, len(rows))}), flush=True)

    def summary(values: list[dict[str, Any]]) -> dict[str, Any]:
        tokens = sum(v["tokens"] for v in values)
        nll = sum(v["nll_sum"] for v in values)
        return {
            "prompts": len(values),
            "tokens": tokens,
            "mean_nll": nll / tokens,
            "perplexity": math.exp(min(20.0, nll / tokens)),
            "token_accuracy": sum(v["correct_tokens"] for v in values) / tokens,
            "wrong_recent_repeat_predictions": sum(v["wrong_recent_repeat_predictions"] for v in values),
        }
    aggregate = summary(per_prompt)
    aggregate["per_capability"] = {cap: summary(by_cap[cap]) for cap in CAPABILITIES}
    return aggregate, per_prompt


@torch.inference_mode()
def _autonomous(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    from .capability_compiler_phase3_sequence_bridge import _generate

    observations = []
    for index, row in enumerate(rows):
        output, token_ids, route = _generate(
            model, tokenizer, str(row["probe"]["prompt"]), int(row["probe"]["max_new_tokens"]), device
        )
        observations.append({
            "probe_id": str(row["teacher"]["probe_id"]),
            "capability": str(row["teacher"]["capability"]),
            "functional_pass": evaluate_functional(output, row["probe"]["evaluator"]),
            "repetition_collapse": repetition_collapse(output),
            "output_tokens": len(token_ids),
            "automatic_route": route,
        })
        if (index + 1) % 200 == 0:
            print(json.dumps({"autonomous": index + 1}), flush=True)
    grouped = {cap: [v for v in observations if v["capability"] == cap] for cap in CAPABILITIES}
    return {
        "observations": len(observations),
        "functional_passes": sum(bool(v["functional_pass"]) for v in observations),
        "functional_rate": sum(bool(v["functional_pass"]) for v in observations) / len(observations),
        "repetition_collapses": sum(bool(v["repetition_collapse"]) for v in observations),
        "per_capability": {
            cap: {
                "passes": sum(bool(v["functional_pass"]) for v in values),
                "collapses": sum(bool(v["repetition_collapse"]) for v in values),
            }
            for cap, values in grouped.items()
        },
        "route_counts": dict(sorted(Counter(v["automatic_route"] for v in observations).items())),
    }


@torch.inference_mode()
def _corruption_recovery(model: Any, clean_rows: list[dict[str, Any]], device: torch.device, per_capability: int, batch_size: int) -> dict[str, Any]:
    selected = []
    counts = Counter()
    for row in clean_rows:
        cap = row["capability"]
        if counts[cap] < per_capability:
            selected.append(row)
            counts[cap] += 1
    prepared = []
    for row in selected:
        wrong = next((i for i, (p, t) in enumerate(zip(row["predictions"], row["targets"])) if p != t), None)
        if wrong is None or wrong + 1 >= len(row["targets"]):
            continue
        corrupted = list(row["input_ids"])
        absolute = row["prompt_length"] + wrong
        corrupted[absolute] = row["predictions"][wrong]
        prepared.append((row, wrong, absolute, corrupted))
    observations = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        corrupt_logits = _forward_batch(model, [(v[3], v[0]["prompt_length"]) for v in batch], device)
        clean_logits = _forward_batch(model, [(v[0]["input_ids"], v[0]["prompt_length"]) for v in batch], device)
        for (row, wrong, absolute, _), corrupted_full, clean_full in zip(batch, corrupt_logits, clean_logits):
            targets = torch.tensor(row["targets"][wrong + 1 :], dtype=torch.long, device=device)
            limit = min(len(targets), max(HORIZONS))
            if limit == 0:
                continue
            corrupted_slice = corrupted_full[absolute:-1][:limit]
            clean_slice = clean_full[absolute:-1][:limit]
            corrupt_losses = F.cross_entropy(corrupted_slice, targets[:limit], reduction="none")
            clean_losses = F.cross_entropy(clean_slice, targets[:limit], reduction="none")
            corrupt_predictions = corrupted_slice.argmax(dim=-1)
            clean_predictions = clean_slice.argmax(dim=-1)
            item = {"probe_id": row["probe_id"], "capability": row["capability"], "available": limit, "horizons": {}}
            for horizon in HORIZONS:
                used = min(horizon, limit)
                item["horizons"][str(horizon)] = {
                    "tokens": used,
                    "clean_correct": int(clean_predictions[:used].eq(targets[:used]).sum().item()),
                    "corrupted_correct": int(corrupt_predictions[:used].eq(targets[:used]).sum().item()),
                    "corrupted_nll_sum": float(corrupt_losses[:used].sum().item()),
                    "clean_nll_sum": float(clean_losses[:used].sum().item()),
                }
            observations.append(item)
        if min(start + batch_size, len(prepared)) % 70 == 0:
            print(json.dumps({"corruption_recovery": min(start + batch_size, len(prepared))}), flush=True)

    summaries = {}
    for horizon in HORIZONS:
        values = [item["horizons"][str(horizon)] for item in observations]
        tokens = sum(v["tokens"] for v in values)
        clean_nll = sum(v["clean_nll_sum"] for v in values)
        corrupted_nll = sum(v["corrupted_nll_sum"] for v in values)
        clean_accuracy = sum(v["clean_correct"] for v in values) / tokens
        corrupted_accuracy = sum(v["corrupted_correct"] for v in values) / tokens
        summaries[str(horizon)] = {
            "observations": len(values), "tokens": tokens,
            "clean_mean_nll": clean_nll / tokens, "corrupted_mean_nll": corrupted_nll / tokens,
            "nll_ratio_corrupted_to_clean": corrupted_nll / clean_nll,
            "clean_accuracy": clean_accuracy, "corrupted_accuracy": corrupted_accuracy,
            "accuracy_delta_corrupted_minus_clean": corrupted_accuracy - clean_accuracy,
        }
    return {"selection": "first lexical 20 prompts per capability; first wrong C0 teacher-forced prediction", "observations": len(observations), "by_horizon": summaries, "aggregate": summaries[str(max(HORIZONS))]}


def run(root: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_dir.exists():
        raise DiagnosticError(f"diagnostic output is immutable: {output_dir}")
    if not torch.cuda.is_available():
        raise DiagnosticError("registered GPU diagnostic device unavailable")
    device = torch.device("cuda")
    rows = _teacher_rows(root, protocol)
    v11, _ = load_v11_protocol(root, (root / protocol["v11_protocol"]).resolve())
    results: dict[str, Any] = {}
    c0_details = None
    started = time.perf_counter()
    for system in SYSTEMS:
        if system == "P0":
            parent = (root / v11["host"]["parent_path"]).resolve()
            model, tokenizer, _ = load_layercake_core(parent, layercake_root=(root / v11["host"]["layercake_root"]).resolve(), device=device)
        else:
            candidate = (root / protocol["candidate_paths"][system]).resolve()
            metadata = _json(candidate / "metadata.json")
            if sha256_file(candidate / "model.safetensors") != metadata["checkpoint"]["sha256"]:
                raise DiagnosticError(f"candidate checkpoint changed: {system}")
            model, tokenizer = load_candidate(root=root, protocol=v11, candidate_dir=candidate, device=device)
        batch_size = int(protocol["execution"]["batch_size"])
        teacher_forced, details = _teacher_forced(model, tokenizer, rows, device, batch_size)
        autonomous = _autonomous(model, tokenizer, rows, device) if system == "P0" else protocol["sealed_autonomous_results"][system]
        results[system] = {"teacher_forced": teacher_forced, "autonomous": autonomous}
        if system == "C0":
            c0_details = details
            recovery = _corruption_recovery(model, details, device, int(protocol["corruption_recovery"]["prompts_per_capability"]), batch_size)
        del model
        torch.cuda.empty_cache()
    assert c0_details is not None
    evidence = {
        "format": FORMAT, "status": "COMPLETE_DIAGNOSTIC_NO_PROMOTION",
        "protocol_sha256": protocol_sha, "final_test_accessed": False, "training_performed": False,
        "external_layercake_control": _external_control(root, protocol),
        "systems": results, "c0_corruption_recovery": recovery,
        "wall_seconds": time.perf_counter() - started,
    }
    evidence["attribution"] = classify_attribution(evidence, protocol["attribution_rules"])
    evidence["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    output_dir.mkdir(parents=True)
    _write_immutable(output_dir / "evidence.json", json.dumps(evidence, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FAILURE_ATTRIBUTION_PROTOCOL_V13.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_failure_attribution/v13")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps({"status": result["status"], "attribution": result["attribution"], "evidence_sha256": result["evidence_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
