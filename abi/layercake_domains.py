"""Conform ABI-selected domain records to LayerCake's sealed direct-cake ABI.

This module is deliberately an adapter, not a copy of LayerCake.  It imports a
caller-supplied, exact LayerCake checkout, verifies the pinned commit, and uses
that checkout's public token-plan and package APIs.  Search records may train a
candidate.  Validation records may only evaluate it.  Final-test records are
never read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import psutil
import torch
import torch.nn.functional as F

from .capability_pipeline import TRAINING_ARTIFACT_ROLE, read_extraction_bundle
from .hf_extraction import evaluate_output
from .layercake_host import strip_source_chat_template


PROTOCOL_FORMAT = "abi-layercake-domain-conformance-protocol/1"
ARTIFACT_FORMAT = "abi-layercake-domain-candidate/1"
TRAINING_EVIDENCE_FORMAT = "abi-layercake-domain-training-evidence/1"
VALIDATION_EVIDENCE_FORMAT = "abi-layercake-domain-validation-evidence/1"
PACKAGE_EVIDENCE_FORMAT = "abi-layercake-domain-package-evidence/1"
LAYERCAKE_COMMIT = "04cf2927a16fba686cd640e18a78708e5658bbda"
DIRECT_ABI_VERSION = "lc-direct-neural-decoder/1"
DIRECT_ABI_SHA256 = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)
SUPPORTED_DOMAINS = frozenset(
    {"chemistry", "civics", "mathematics", "python"}
)


class DomainConformanceError(ValueError):
    """A fail-closed domain acquisition or certification error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise DomainConformanceError(f"evidence is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("format") != PROTOCOL_FORMAT
        or value.get("status")
        != "PREREGISTERED_BEFORE_DOMAIN_CONFORMANCE_TRAINING"
    ):
        raise DomainConformanceError("domain conformance protocol is invalid")
    target = value.get("immutable_layercake_target", {})
    if target.get("repository_commit") != LAYERCAKE_COMMIT:
        raise DomainConformanceError("protocol LayerCake commit is not pinned")
    if target.get("sealed_repository_may_be_modified") is not False:
        raise DomainConformanceError("protocol does not protect sealed LayerCake")
    return value


def _layercake_commit(root: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _import_layercake(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not (root / "layercake" / "portable_token_plan.py").is_file():
        raise DomainConformanceError("LayerCake runtime root is invalid")
    commit = _layercake_commit(root)
    if commit != LAYERCAKE_COMMIT:
        raise DomainConformanceError(
            f"LayerCake commit mismatch: expected {LAYERCAKE_COMMIT}, got {commit}"
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from layercake.cake.manifest import CakeManifest
    from layercake.cake.package import build_package, load_package, tensor_specs
    from layercake.cake.signing import generate_keypair
    from layercake.models.direct_cake_host import DirectCakeHost
    from layercake.models.portable_decoder import (
        portable_token_plan_manifest_architecture,
    )
    from layercake.portable_token_plan import (
        LosslessLexemePointerTokenizer,
        PortableTokenPlan,
        build_token_plan_artifact,
        load_token_plan_artifact,
    )

    return {
        "CakeManifest": CakeManifest,
        "build_package": build_package,
        "load_package": load_package,
        "tensor_specs": tensor_specs,
        "generate_keypair": generate_keypair,
        "DirectCakeHost": DirectCakeHost,
        "portable_token_plan_manifest_architecture": (
            portable_token_plan_manifest_architecture
        ),
        "LosslessLexemePointerTokenizer": LosslessLexemePointerTokenizer,
        "PortableTokenPlan": PortableTokenPlan,
        "build_token_plan_artifact": build_token_plan_artifact,
        "load_token_plan_artifact": load_token_plan_artifact,
        "commit": commit,
    }


def _selected_domain_item(bundle: Mapping[str, Any], domain: str) -> dict[str, Any]:
    items = [
        item
        for item in bundle["selection"]["selected_items"]
        if item["destination_scope"] == "domain_cake"
        and item["domain"] == domain
    ]
    if len(items) != 1:
        raise DomainConformanceError(
            f"selection must contain exactly one source for domain {domain!r}"
        )
    return dict(items[0])


def _candidate_copy_lexemes(
    prompt: str,
    response: str,
    evaluator: Mapping[str, Any],
) -> list[str]:
    """Choose one exact prompt identity lexeme without inspecting validation."""

    preferred = evaluator.get("function_name")
    if (
        isinstance(preferred, str)
        and prompt.count(preferred) == 1
        and preferred in response
    ):
        return [preferred]
    import re

    prompt_words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", prompt)
    response_words = set(
        re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", response)
    )
    candidates = [
        value
        for value in prompt_words
        if value in response_words
        and prompt.count(value) == 1
        and response.count(value) >= 1
    ]
    if candidates:
        candidates.sort(key=lambda value: (-len(value), value.encode("utf-8")))
        return [candidates[0]]
    # LayerCake's generic tokenizer requires a declared, unique source lexeme
    # even when the target needs no pointer action.  A prompt-only sentinel is
    # valid: it is excluded from the fixed vocabulary, occurs once in source,
    # and is never emitted.  This preserves the source answer byte-for-byte.
    prompt_only = [
        value
        for value in prompt_words
        if prompt.count(value) == 1
    ]
    if not prompt_only:
        raise DomainConformanceError(
            "training row has no unique source lexeme for pointer conformance"
        )
    prompt_only.sort(key=lambda value: (-len(value), value.encode("utf-8")))
    return [prompt_only[0]]


def load_domain_training_rows(
    bundle_path: str | Path,
    *,
    domain: str,
    budget_index: int = -1,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load selected, passing, search-only rows for one domain."""

    if domain not in SUPPORTED_DOMAINS:
        raise DomainConformanceError(f"unsupported domain: {domain}")
    bundle = read_extraction_bundle(bundle_path)
    if (
        bundle["verification"]["artifact_role"] != TRAINING_ARTIFACT_ROLE
        or bundle["verification"]["training_eligible"] is not True
    ):
        raise DomainConformanceError("bundle is not current training material")
    budgets = bundle["budgets"]
    if budget_index < 0:
        budget_index += len(budgets)
    if not 0 <= budget_index < len(budgets):
        raise DomainConformanceError("budget index is outside the bundle")
    budget = budgets[budget_index]
    if budget["split"] != "search":
        raise DomainConformanceError("domain training may use search only")
    selected = _selected_domain_item(bundle, domain)
    allowed = set(budget["record_ids"])
    results = {
        str(result["record_id"]): result for result in bundle["probe_results"]
    }
    rows: list[dict[str, Any]] = []
    for record in bundle["records"]:
        if record["record_id"] not in allowed:
            continue
        if (
            record["destination_scope"] != "domain_cake"
            or record["domain"] != domain
        ):
            continue
        if record["split"] != "search":
            raise DomainConformanceError("non-search row crossed training boundary")
        if (
            record["source_model"] != selected["source_model"]
            or record["source_model_revision"]
            != selected["source_model_revision"]
        ):
            raise DomainConformanceError("unselected source crossed domain boundary")
        result = results.get(str(record["record_id"]))
        if result is None or result["passed"] is not True:
            raise DomainConformanceError(
                "failed or unscored source row crossed training boundary"
            )
        prompt = strip_source_chat_template(str(record["prompt"]))
        response = str(record["output"])
        rows.append(
            {
                "id": str(record["record_id"]),
                "domain_id": domain,
                "capability": str(record["capability"]),
                "prompt": prompt,
                "response": response,
                "copy_lexemes": _candidate_copy_lexemes(
                    prompt, response, result["evaluator"]
                ),
                "evaluator": dict(result["evaluator"]),
                "teacher_tokens": int(record["teacher_tokens"]),
                "prompt_utf8_bytes": int(record["prompt_utf8_bytes"]),
                "output_utf8_bytes": int(record["output_utf8_bytes"]),
                "source_model": str(record["source_model"]),
                "source_model_revision": str(record["source_model_revision"]),
                "provenance": str(record["provenance"]),
            }
        )
    rows.sort(key=lambda row: row["id"])
    if not rows:
        raise DomainConformanceError(f"budget contains no rows for {domain}")
    return rows, budget, bundle


def build_domain_validation_rows(
    *,
    training_bundle_path: str | Path,
    validation_bundle_paths: Sequence[str | Path],
    domain: str,
) -> list[dict[str, Any]]:
    """Bind 100 held-out rows to the exact source selected for training."""

    training = read_extraction_bundle(training_bundle_path)
    selected = _selected_domain_item(training, domain)
    training_catalogs = {
        str(record["provenance"]).partition(":")[0]
        for record in training["records"]
        if record["destination_scope"] == "domain_cake"
        and record["domain"] == domain
        and record["source_model"] == selected["source_model"]
        and record["source_model_revision"] == selected["source_model_revision"]
    }
    if len(training_catalogs) != 1:
        raise DomainConformanceError("training catalog provenance is ambiguous")
    rows: dict[str, dict[str, Any]] = {}
    for path in validation_bundle_paths:
        bundle = read_extraction_bundle(path)
        results = {
            str(result["record_id"]): result for result in bundle["probe_results"]
        }
        for record in bundle["records"]:
            if (
                record["destination_scope"] != "domain_cake"
                or record["domain"] != domain
                or record["split"] != "validation"
                or record["source_model"] != selected["source_model"]
                or record["source_model_revision"]
                != selected["source_model_revision"]
                or str(record["provenance"]).partition(":")[0]
                not in training_catalogs
            ):
                continue
            result = results.get(str(record["record_id"]))
            if result is None:
                raise DomainConformanceError("validation row lacks source result")
            row = {
                "id": str(record["record_id"]),
                "probe_id": str(result["probe_id"]),
                "domain_id": domain,
                "capability": str(record["capability"]),
                "prompt": strip_source_chat_template(str(record["prompt"])),
                "source_output": str(record["output"]),
                "source_passed": bool(result["passed"]),
                "source_score": float(result["score"]),
                "evaluator": dict(result["evaluator"]),
                "source_model": str(record["source_model"]),
                "source_model_revision": str(record["source_model_revision"]),
                "provenance": str(record["provenance"]),
            }
            if row["id"] in rows and rows[row["id"]] != row:
                raise DomainConformanceError("conflicting validation record")
            rows[row["id"]] = row
    if len(rows) != 100:
        raise DomainConformanceError(
            f"held-out domain evidence is incomplete: {domain}={len(rows)}"
        )
    return sorted(rows.values(), key=lambda row: row["probe_id"])


def _batch(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    maximum_source_lexemes: int,
    maximum_target_actions: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sources: list[list[int]] = []
    targets: list[list[int]] = []
    raw_bytes = 0
    for row in rows:
        source, lexemes = tokenizer.encode_source(str(row["prompt"]) + "\n")
        target = tokenizer.encode_target(
            str(row["response"]),
            copy_lexemes=row["copy_lexemes"],
            source_lexemes=lexemes,
        )
        if len(source) > maximum_source_lexemes:
            raise DomainConformanceError("source exceeds protocol lexeme maximum")
        if len(target) > maximum_target_actions:
            raise DomainConformanceError("target exceeds protocol action maximum")
        if tokenizer.decode_actions(target, lexemes) != str(
            row["response"]
        ).encode("utf-8"):
            raise DomainConformanceError("lossless target roundtrip failed")
        sources.append(source)
        targets.append(target)
        raw_bytes += len((str(row["prompt"]) + "\n").encode("utf-8"))
        raw_bytes += len(str(row["response"]).encode("utf-8"))
    source_width = max(map(len, sources))
    target_width = max(map(len, targets))
    source_tensor = torch.zeros(
        len(rows), source_width, dtype=torch.long, device=device
    )
    target_tensor = torch.full(
        (len(rows), target_width), -100, dtype=torch.long, device=device
    )
    for index, (source, target) in enumerate(zip(sources, targets)):
        source_tensor[index, : len(source)] = torch.tensor(
            source, dtype=torch.long, device=device
        )
        target_tensor[index, : len(target)] = torch.tensor(
            target, dtype=torch.long, device=device
        )
    return {
        "source_ids": source_tensor,
        "target_actions": target_tensor,
        "raw_bytes": torch.tensor(raw_bytes, device=device),
    }


def _loss(model: Any, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, Any]:
    result = model(batch["source_ids"], batch["target_actions"])
    mask = batch["target_actions"].ge(0)
    return (
        F.nll_loss(result["log_probs"][mask], batch["target_actions"][mask]),
        result,
    )


class _Sampler:
    def __init__(self, rows: list[dict[str, Any]], batch_size: int, seed: int):
        if not rows or batch_size <= 0:
            raise DomainConformanceError("invalid training batch size")
        self.rows = rows
        self.batch_size = batch_size
        self.random = random.Random(seed)
        self.order = list(range(len(rows)))
        self.cursor = len(rows)

    def next(self) -> list[dict[str, Any]]:
        chosen: list[int] = []
        while len(chosen) < self.batch_size:
            if self.cursor >= len(self.order):
                self.random.shuffle(self.order)
                self.cursor = 0
            take = min(
                self.batch_size - len(chosen),
                len(self.order) - self.cursor,
            )
            chosen.extend(self.order[self.cursor : self.cursor + take])
            self.cursor += take
        return [self.rows[index] for index in chosen]


def _deterministic_math_rows(
    repair: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if repair.get("kind") != "deterministic_elementary_algebra_closure":
        raise DomainConformanceError("unsupported mathematics repair")
    addend_range = repair.get("addend_inclusive_range")
    solution_range = repair.get("solution_inclusive_range")
    if (
        not isinstance(addend_range, list)
        or len(addend_range) != 2
        or not isinstance(solution_range, list)
        or len(solution_range) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in addend_range + solution_range
        )
    ):
        raise DomainConformanceError("mathematics repair ranges are invalid")
    rows: list[dict[str, Any]] = []
    for addend in range(addend_range[0], addend_range[1] + 1):
        for solution in range(solution_range[0], solution_range[1] + 1):
            total = addend + solution
            identity = f"derived-algebra-a{addend:03d}-x{solution:03d}"
            prompt = (
                f"Solve x + {addend} = {total}. "
                "Give only the numerical value of x."
            )
            rows.append(
                {
                    "id": hashlib.sha256(identity.encode("ascii")).hexdigest(),
                    "domain_id": "mathematics",
                    "capability": "elementary_algebra",
                    "prompt": prompt,
                    "response": f"The numerical value of x is {solution}.",
                    "copy_lexemes": ["numerical"],
                    "evaluator": {
                        "kind": "numeric_equal",
                        "value": solution,
                        "absolute_tolerance": 0,
                    },
                    "teacher_tokens": 0,
                    "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                    "output_utf8_bytes": 0,
                    "source_model": "deterministic_algebra_rule",
                    "source_model_revision": _canonical_sha(repair),
                    "provenance": identity,
                    "derived_without_teacher": True,
                }
            )
    if len(rows) != int(repair["generated_rows"]):
        raise DomainConformanceError("mathematics generated-row count is stale")
    return rows


def train_domain_candidate(
    *,
    protocol_path: str | Path,
    bundle_path: str | Path,
    layercake_root: str | Path,
    domain: str,
    budget_index: int,
    seed: int,
    output_directory: str | Path,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    bundle_path = Path(bundle_path).resolve()
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise DomainConformanceError(
            f"candidate directory is immutable: {output_directory}"
        )
    protocol = _read_protocol(protocol_path)
    settings = protocol["bounded_training"]
    if seed not in settings["seeds"]:
        raise DomainConformanceError("seed is outside preregistered set")
    source_rows, budget, bundle = load_domain_training_rows(
        bundle_path, domain=domain, budget_index=budget_index
    )
    rows = list(source_rows)
    repair = protocol.get("domain_repairs", {}).get(domain)
    derived_rows: list[dict[str, Any]] = []
    if repair is not None:
        if domain != "mathematics":
            raise DomainConformanceError("domain repair is not implemented")
        derived_rows = _deterministic_math_rows(repair)
        rows.extend(derived_rows)
    lc = _import_layercake(Path(layercake_root))
    tokenizer = lc["LosslessLexemePointerTokenizer"].build_generic(rows)
    architecture = dict(protocol["architecture"])
    if architecture.pop("name") != "portable_token_plan_pointer_transformer":
        raise DomainConformanceError("unsupported protocol architecture")
    architecture.pop("physical_execution")
    if not torch.cuda.is_available():
        raise DomainConformanceError("preregistered primary CUDA is unavailable")
    device = torch.device(str(settings["primary_device"]))
    device_index = device.index or 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    model = lc["PortableTokenPlan"](
        fixed_vocab_size=tokenizer.vocab_size, **architecture
    ).to(device)
    resume_evidence = None
    resume = protocol.get("resume_candidate")
    if resume is not None:
        resume_path = Path(str(resume["path"]))
        if not resume_path.is_absolute():
            resume_path = protocol_path.parent / resume_path
        resume_path = resume_path.resolve()
        if _sha256_file(resume_path) != resume.get("sha256"):
            raise DomainConformanceError("resume candidate hash mismatch")
        resume_artifact = torch.load(
            resume_path, map_location="cpu", weights_only=True
        )
        old_spec, old_tokenizer, old_model = lc[
            "load_token_plan_artifact"
        ](resume_artifact, "cpu")
        if (
            old_spec["domain_id"] != domain
            or old_tokenizer.hash() != tokenizer.hash()
            or old_model.canonical_config() != model.canonical_config()
            or resume_artifact["payload_hash"] != resume.get("payload_hash")
        ):
            raise DomainConformanceError(
                "resume candidate representation differs from protocol"
            )
        model.load_state_dict(old_model.state_dict(), strict=True)
        resume_evidence = {
            "path": str(resume_path),
            "sha256": _sha256_file(resume_path),
            "payload_hash": resume_artifact["payload_hash"],
            "prior_optimizer_steps": int(resume["prior_optimizer_steps"]),
            "optimizer_state_reused": False,
        }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    sampler = _Sampler(rows, int(settings["batch_size"]), seed + 1)
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    visible_actions = 0
    raw_bytes = 0
    model.train()
    steps = int(settings["optimizer_steps"])
    for step in range(1, steps + 1):
        batch = _batch(
            sampler.next(),
            tokenizer,
            maximum_source_lexemes=model.maximum_source_lexemes,
            maximum_target_actions=model.maximum_target_actions,
            device=device,
        )
        loss, result = _loss(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        value = float(loss.item())
        raw_bytes += int(batch["raw_bytes"].item())
        visible_actions += int(batch["target_actions"].ge(0).sum().item())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if step == 1 or step % 100 == 0 or step == steps:
            mask = batch["target_actions"].ge(0)
            predictions = result["log_probs"][mask].argmax(dim=-1)
            targets = batch["target_actions"][mask]
            pointer_mask = targets.ge(tokenizer.vocab_size)
            item = {
                "step": step,
                "action_negative_log_likelihood": value,
                "action_accuracy": float(predictions.eq(targets).float().mean()),
                "pointer_action_accuracy": (
                    float(
                        predictions[pointer_mask]
                        .eq(targets[pointer_mask])
                        .float()
                        .mean()
                    )
                    if bool(pointer_mask.any())
                    else None
                ),
                "gradient_norm_before_clip": float(gradient_norm),
                "gpu_wall_seconds": time.perf_counter() - started,
            }
            history.append(item)
            print(json.dumps({"domain": domain, **item}), flush=True)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    if best_state is None:
        raise DomainConformanceError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    torch.cuda.synchronize(device_index)
    gpu_seconds = time.perf_counter() - started
    peak_gpu = int(torch.cuda.max_memory_allocated(device_index))
    cpu_model = model.cpu()
    smoke_optimizer = torch.optim.AdamW(cpu_model.parameters(), lr=1e-5)
    smoke_batch = _batch(
        rows[: min(4, len(rows))],
        tokenizer,
        maximum_source_lexemes=cpu_model.maximum_source_lexemes,
        maximum_target_actions=cpu_model.maximum_target_actions,
        device=torch.device("cpu"),
    )
    smoke_started = time.perf_counter()
    smoke_loss, _ = _loss(cpu_model.train(), smoke_batch)
    smoke_optimizer.zero_grad(set_to_none=True)
    smoke_loss.backward()
    smoke_optimizer.step()
    cpu_smoke = {
        "status": "PASS",
        "optimizer_steps": 1,
        "loss": float(smoke_loss),
        "wall_seconds": time.perf_counter() - smoke_started,
    }
    cpu_model.load_state_dict(best_state)
    cpu_model.eval()
    artifact = lc["build_token_plan_artifact"](
        cpu_model,
        tokenizer,
        domain_id=domain,
        training={
            "seed": seed,
            "optimizer_steps": steps,
            "primary_device": str(device),
            "primary_device_name": torch.cuda.get_device_name(device_index),
            "protocol_sha256": _sha256_file(protocol_path),
            "training_bundle_sha256": bundle["verification"]["archive_sha256"],
            "budget_id": budget["budget_id"],
        },
    )
    output_directory.mkdir(parents=True)
    artifact_path = output_directory / "domain_candidate.pt"
    torch.save(artifact, artifact_path)
    selected = _selected_domain_item(bundle, domain)
    accounting = {
        "raw_source_prompts": len(source_rows),
        "unique_prompt_utf8_bytes": sum(
            row["prompt_utf8_bytes"] for row in source_rows
        ),
        "teacher_generated_output_bytes": sum(
            row["output_utf8_bytes"] for row in source_rows
        ),
        "teacher_tokens": sum(row["teacher_tokens"] for row in source_rows),
        "deterministic_derived_training_rows": len(derived_rows),
        "deterministic_derived_teacher_tokens": 0,
        "deterministic_derived_teacher_outputs": 0,
        "logits_stored": 0,
        "hidden_activations_stored": 0,
        "frozen_source_parameters_copied": 0,
        "final_imported_substrate_parameters": cpu_model.parameter_count(),
        "bridge_parameters_trained": cpu_model.parameter_count(),
        "source_transformer_blocks_retained": 0,
        "teacher_present_at_inference": False,
        "artifact_disk_footprint_bytes": artifact_path.stat().st_size,
        "peak_process_resident_memory_bytes": peak_rss,
        "peak_accelerator_memory_bytes": peak_gpu,
        "source_model_inference_hours": 0.0,
        "source_extraction_cost_reused": True,
        "external_hardware": {
            "used": True,
            "device": torch.cuda.get_device_name(device_index),
            "role": "domain conformance training only",
        },
    }
    evidence = {
        "format": TRAINING_EVIDENCE_FORMAT,
        "status": "TRAINED_NOT_VALIDATED",
        "domain": domain,
        "seed": seed,
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "layercake_target": {
            "root": str(Path(layercake_root).resolve()),
            "commit": lc["commit"],
            "abi_version": DIRECT_ABI_VERSION,
            "abi_sha256": DIRECT_ABI_SHA256,
            "modified": False,
        },
        "source": selected,
        "training_bundle": {
            "path": str(bundle_path),
            "sha256": bundle["verification"]["archive_sha256"],
            "budget_id": budget["budget_id"],
            "budget_index": budget_index,
        },
        "training_rows": len(rows),
        "source_search_training_rows": len(source_rows),
        "deterministic_derived_training_rows": len(derived_rows),
        "parameter_count": cpu_model.parameter_count(),
        "fixed_vocabulary_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.hash(),
        "best_action_negative_log_likelihood": best_loss,
        "optimizer_steps": steps,
        "resume_candidate": resume_evidence,
        "gpu_wall_seconds": gpu_seconds,
        "raw_utf8_training_bytes_exposed_with_repetition": raw_bytes,
        "model_visible_target_actions_with_repetition": visible_actions,
        "cpu_fallback_smoke": cpu_smoke,
        "accounting": accounting,
        "artifact": {
            "path": str(artifact_path),
            "sha256": _sha256_file(artifact_path),
            "payload_hash": artifact["payload_hash"],
            "spec_hash": artifact["spec_hash"],
        },
        "history": history,
        "validation_accessed": False,
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    _write_json(output_directory / "training_evidence.json", evidence)
    return evidence


@torch.inference_mode()
def evaluate_domain_candidate(
    *,
    protocol_path: str | Path,
    training_bundle_path: str | Path,
    validation_bundle_paths: Sequence[str | Path],
    layercake_root: str | Path,
    domain: str,
    candidate_directory: str | Path,
    output_path: str | Path,
    device_name: str,
) -> dict[str, Any]:
    protocol = _read_protocol(Path(protocol_path).resolve())
    candidate_directory = Path(candidate_directory).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise DomainConformanceError(f"validation evidence is immutable: {output_path}")
    training = json.loads(
        (candidate_directory / "training_evidence.json").read_text(encoding="utf-8")
    )
    if training.get("format") != TRAINING_EVIDENCE_FORMAT:
        raise DomainConformanceError("candidate training evidence is invalid")
    artifact_path = candidate_directory / "domain_candidate.pt"
    if _sha256_file(artifact_path) != training["artifact"]["sha256"]:
        raise DomainConformanceError("candidate artifact hash mismatch")
    rows = build_domain_validation_rows(
        training_bundle_path=training_bundle_path,
        validation_bundle_paths=validation_bundle_paths,
        domain=domain,
    )
    lc = _import_layercake(Path(layercake_root))
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DomainConformanceError("requested CUDA validation is unavailable")
    spec, tokenizer, model = lc["load_token_plan_artifact"](artifact, device)
    if spec["domain_id"] != domain:
        raise DomainConformanceError("candidate domain identity mismatch")
    observations = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        request_started = time.perf_counter()
        try:
            raw = model.generate_bytes(row["prompt"] + "\n")
            output = raw.decode("utf-8")
            invalid_utf8 = False
        except UnicodeDecodeError:
            raw = b""
            output = ""
            invalid_utf8 = True
        passed, score = evaluate_output(output, row["evaluator"])
        observations.append(
            {
                **row,
                "layercake_output": output,
                "layercake_output_sha256": hashlib.sha256(raw).hexdigest(),
                "layercake_passed": passed,
                "layercake_score": score,
                "invalid_utf8": invalid_utf8,
                "latency_seconds": time.perf_counter() - request_started,
            }
        )
        if index % 25 == 0:
            print(
                json.dumps(
                    {
                        "domain": domain,
                        "evaluated": index,
                        "passes": sum(item["layercake_passed"] for item in observations),
                    }
                ),
                flush=True,
            )
    source_passes = sum(row["source_passed"] for row in observations)
    layercake_passes = sum(row["layercake_passed"] for row in observations)
    regressions = sum(
        row["source_passed"] and not row["layercake_passed"]
        for row in observations
    )
    invalid_utf8 = sum(row["invalid_utf8"] for row in observations)
    passed = (
        len(observations)
        == int(protocol["data_boundary"]["minimum_distinct_validation_prompts_per_domain"])
        and regressions == 0
        and layercake_passes >= source_passes
        and invalid_utf8 == 0
    )
    evidence = {
        "format": VALIDATION_EVIDENCE_FORMAT,
        "status": "PASS" if passed else "FAIL",
        "domain": domain,
        "seed": training["seed"],
        "split": "validation",
        "protocol_sha256": _sha256_file(Path(protocol_path).resolve()),
        "candidate_artifact_sha256": training["artifact"]["sha256"],
        "training_bundle_sha256": _sha256_file(Path(training_bundle_path)),
        "validation_bundle_sha256": [
            _sha256_file(Path(path)) for path in validation_bundle_paths
        ],
        "device": str(device),
        "observation_count": len(observations),
        "source_passes": source_passes,
        "layercake_passes": layercake_passes,
        "source_passing_regressions": regressions,
        "invalid_utf8_outputs": invalid_utf8,
        "median_latency_seconds": statistics.median(
            row["latency_seconds"] for row in observations
        ),
        "wall_seconds": time.perf_counter() - started,
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "validation_used_for_training": False,
        "final_test_accessed": False,
        "observations": observations,
        "claim_boundary": (
            "Paired retention on 100 declared validation prompts for the exact "
            "selected source/domain. Not final-test or universal semantic identity."
        ),
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    _write_json(output_path, evidence)
    return evidence


def package_validated_candidate(
    *,
    protocol_path: str | Path,
    layercake_root: str | Path,
    domain: str,
    candidate_directory: str | Path,
    validation_path: str | Path,
    package_path: str | Path,
    public_key_path: str | Path,
) -> dict[str, Any]:
    """Sign a validation-passing candidate using an ephemeral research key."""

    _read_protocol(Path(protocol_path).resolve())
    candidate_directory = Path(candidate_directory).resolve()
    validation_path = Path(validation_path).resolve()
    package_path = Path(package_path).resolve()
    public_key_path = Path(public_key_path).resolve()
    if package_path.exists() or public_key_path.exists():
        raise DomainConformanceError("package outputs are immutable")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("format") != VALIDATION_EVIDENCE_FORMAT
        or validation.get("status") != "PASS"
        or validation.get("domain") != domain
    ):
        raise DomainConformanceError("only a passing domain candidate may be packaged")
    training = json.loads(
        (candidate_directory / "training_evidence.json").read_text(encoding="utf-8")
    )
    artifact_path = candidate_directory / "domain_candidate.pt"
    if validation["candidate_artifact_sha256"] != _sha256_file(artifact_path):
        raise DomainConformanceError("validation does not bind candidate artifact")
    lc = _import_layercake(Path(layercake_root))
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in artifact["state_dict"].items()
    }
    private, public, key_id = lc["generate_keypair"]()
    manifest = lc["CakeManifest"](
        schema_version="1",
        cake_id=f"abi-{domain}-token-plan",
        name=f"ABI {domain.title()} Capability",
        description=(
            f"ABI-extracted, source-free, directly selected {domain} capability"
        ),
        version="1.0.0",
        publisher={
            "id": "abi-research",
            "name": "ABI Research",
            "key_id": key_id,
        },
        abi_version=DIRECT_ABI_VERSION,
        abi_hash=DIRECT_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={
            "mode": "direct_selected_portable_decoder",
            "external": "UTF-8 bytes",
            "canonical_semantic_abi_consumed": False,
        },
        output_contract={
            "mode": "autonomous_extended_vocabulary_actions",
            "external": "UTF-8 bytes",
            "composition": "direct_selected_one_cake_no_router",
        },
        architecture=lc["portable_token_plan_manifest_architecture"](
            artifact["spec"]
        ),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={
            "features": ["byte_input", "safe_tensors", "incremental"]
        },
        tensor_payload_hash="",
        tensor_shapes=lc["tensor_specs"](state),
        package_hash="",
        training_data_provenance={
            "abi_training_bundle_sha256": training["training_bundle"]["sha256"],
            "source_model": training["source"]["source_model"],
            "source_model_revision": training["source"]["source_model_revision"],
            "teacher_tokens": training["accounting"]["teacher_tokens"],
            "seed": training["seed"],
            "source_teacher_in_package": False,
        },
        evaluation_evidence={
            "status": "VALIDATION_PASS_NOT_FINAL",
            "validation_evidence_sha256": validation["evidence_sha256"],
            "validation_prompts": validation["observation_count"],
            "source_passing_regressions": validation[
                "source_passing_regressions"
            ],
            "final_test_accessed": False,
        },
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id},
        domains=(domain,),
        keywords=(domain, "abi", "portable", "token-plan"),
        permissions=("local-inference",),
    )
    package_path.parent.mkdir(parents=True, exist_ok=True)
    lc["build_package"](package_path, manifest, state, private_key=private)
    with tempfile.TemporaryDirectory(prefix=f"abi-{domain}-rebuild-") as temp:
        rebuilt = Path(temp) / package_path.name
        lc["build_package"](rebuilt, manifest, state, private_key=private)
        deterministic = rebuilt.read_bytes() == package_path.read_bytes()
    if not deterministic:
        raise DomainConformanceError("package rebuild was not byte-identical")
    public_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.write_bytes(public)
    loaded = lc["load_package"](
        package_path, trust_store={key_id: public}
    )
    evidence = {
        "format": PACKAGE_EVIDENCE_FORMAT,
        "status": "VALIDATION_CANDIDATE_NOT_PROMOTED",
        "domain": domain,
        "seed": training["seed"],
        "package_path": str(package_path),
        "archive_sha256": _sha256_file(package_path),
        "archive_bytes": package_path.stat().st_size,
        "package_hash": loaded.manifest.package_hash,
        "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        "public_key_path": str(public_key_path),
        "public_key_sha256": _sha256_file(public_key_path),
        "key_id": key_id,
        "signed": loaded.signed,
        "deterministic_rebuild_identity": deterministic,
        "non_executable_safetensors_only": True,
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "validation_evidence_sha256": validation["evidence_sha256"],
        "final_test_accessed": False,
        "promotion_blockers": [
            "three_seed_validation",
            "receiver_matrix",
            "tamper_and_lifecycle",
            "cpu_cuda_identity",
            "unopened_final_test",
        ],
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    _write_json(package_path.with_suffix(".package.json"), evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--protocol", required=True)
    train.add_argument("--bundle", required=True)
    train.add_argument("--layercake-root", required=True)
    train.add_argument("--domain", choices=sorted(SUPPORTED_DOMAINS), required=True)
    train.add_argument("--budget-index", type=int, default=-1)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--validation-bundle", action="append", required=True)
    evaluate.add_argument("--layercake-root", required=True)
    evaluate.add_argument(
        "--domain", choices=sorted(SUPPORTED_DOMAINS), required=True
    )
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda")
    package = subparsers.add_parser("package")
    package.add_argument("--protocol", required=True)
    package.add_argument("--layercake-root", required=True)
    package.add_argument(
        "--domain", choices=sorted(SUPPORTED_DOMAINS), required=True
    )
    package.add_argument("--candidate", required=True)
    package.add_argument("--validation", required=True)
    package.add_argument("--output", required=True)
    package.add_argument("--public-key", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        result = train_domain_candidate(
            protocol_path=args.protocol,
            bundle_path=args.bundle,
            layercake_root=args.layercake_root,
            domain=args.domain,
            budget_index=args.budget_index,
            seed=args.seed,
            output_directory=args.output,
        )
    elif args.command == "evaluate":
        result = evaluate_domain_candidate(
            protocol_path=args.protocol,
            training_bundle_path=args.bundle,
            validation_bundle_paths=args.validation_bundle,
            layercake_root=args.layercake_root,
            domain=args.domain,
            candidate_directory=args.candidate,
            output_path=args.output,
            device_name=args.device,
        )
    else:
        result = package_validated_candidate(
            protocol_path=args.protocol,
            layercake_root=args.layercake_root,
            domain=args.domain,
            candidate_directory=args.candidate,
            validation_path=args.validation,
            package_path=args.output,
            public_key_path=args.public_key,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
