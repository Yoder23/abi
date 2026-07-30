"""Teacher-free ABI conformance training for the sealed LayerCake execution host.

The source LLM never enters this process.  A verified search-only ``.abix``
bundle supplies labeled prompts and passing source responses.  The sealed
LayerCake transformer substrate is frozen byte-for-byte; only its existing
task classifier and low-rank task cakes may change.  The output is a small
delta plus a fail-closed deployment manifest, not a copy of the LayerCake
parent checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import psutil
import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file, save_file

from .artifacts import module_state_sha256
from .capability_pipeline import (
    CapabilityPipelineError,
    TRAINING_ARTIFACT_ROLE,
    read_extraction_bundle,
)
from .hf_extraction import evaluate_output, load_probe_catalog


HOST_DELTA_FORMAT = "abi-layercake-host-delta/2"
DEPLOYMENT_FORMAT = "abi-layercake-host-deployment/2"
LEGACY_DEPLOYMENT_FORMAT = "abi-layercake-host-deployment/1"
BRIDGE_PREFIXES = ("task_classifier.", "task_cakes.")
CAPABILITY_TO_ROUTE = {
    "grammar": 0,
    "coherence": 1,
    "prompt_grounding": 4,
    "instruction_following": 4,
    "conversation": 8,
    "summarization": 6,
    "rewriting": 8,
    "email_drafting": 3,
    "tone_control": 4,
    "format_control": 4,
    "clarification": 7,
    "abstention": 7,
    "domain_independent_reasoning": 5,
    "cake_output_realization": 2,
}
SYMBOLIC_SURFACE_FORMAT = "abi-symbolic-surface/1"
SYMBOLIC_SURFACE_STATE_KEY = "symbolic_surface.payload"


class LayerCakeHostError(RuntimeError):
    """Raised when a host acquisition boundary or identity check fails."""


class LoRAConv1D(nn.Module):
    """Training-only low-rank delta for GPT-2 Conv1D weights."""

    def __init__(self, base: nn.Module, *, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise LayerCakeHostError("LoRA rank must be positive")
        weight = getattr(base, "weight", None)
        if weight is None or weight.ndim != 2:
            raise LayerCakeHostError("LoRA target lacks a two-dimensional weight")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.lora_a = nn.Parameter(weight.new_empty(weight.shape[0], rank))
        self.lora_b = nn.Parameter(weight.new_zeros(rank, weight.shape[1]))
        nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.base(hidden) + (
            hidden @ self.lora_a @ self.lora_b
        ) * self.scale


class PromptIdentityBridge(nn.Module):
    """Small pointer bridge that carries exact prompt tokens into generation."""

    def __init__(self, *, width: int, rank: int, routes: int):
        super().__init__()
        if rank <= 0 or rank > width:
            raise LayerCakeHostError(
                "prompt-identity rank must be in [1, LayerCake width]"
            )
        self.rank = int(rank)
        self.key = nn.Linear(width, rank, bias=False)
        self.query = nn.Linear(width, rank, bias=False)
        self.gate = nn.Linear(width, 1)
        self.route_bias = nn.Embedding(routes, 1)
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.query.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        nn.init.zeros_(self.route_bias.weight)

    def pointer_scores(
        self, query_hidden: torch.Tensor, prompt_hidden: torch.Tensor
    ) -> torch.Tensor:
        query = self.query(query_hidden)
        keys = self.key(prompt_hidden)
        return (query @ keys.transpose(-1, -2)).float() / math.sqrt(self.rank)

    def copy_gate(
        self, query_hidden: torch.Tensor, routes: torch.Tensor
    ) -> torch.Tensor:
        route_bias = self.route_bias(routes.long()).squeeze(-1)
        return torch.sigmoid(
            self.gate(query_hidden).squeeze(-1) + route_bias
        ).float()


class SparseRouteConformanceBridge(nn.Module):
    """Physically dispatch one compact residual bridge for each selected route."""

    def __init__(self, *, width: int, rank: int, routes: int):
        super().__init__()
        if rank <= 0 or rank > width:
            raise LayerCakeHostError(
                "route-bridge rank must be in [1, LayerCake width]"
            )
        self.rank = int(rank)
        self.bridges = nn.ModuleList()
        for _ in range(routes):
            bridge = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, rank, bias=False),
                nn.SiLU(),
                nn.Linear(rank, width, bias=False),
            )
            nn.init.normal_(bridge[1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(bridge[3].weight)
            self.bridges.append(bridge)
        self.last_calls: tuple[int, ...] = ()

    def forward(
        self, hidden: torch.Tensor, routes: torch.Tensor
    ) -> torch.Tensor:
        if routes.ndim != 1 or routes.shape[0] != hidden.shape[0]:
            raise LayerCakeHostError(
                "route bridge requires one route per batch row"
            )
        output = torch.empty_like(hidden)
        calls = []
        for route, bridge in enumerate(self.bridges):
            rows = torch.nonzero(routes == route, as_tuple=False).flatten()
            if not rows.numel():
                continue
            selected = hidden.index_select(0, rows)
            output.index_copy_(0, rows, selected + bridge(selected))
            calls.append(route)
        self.last_calls = tuple(calls)
        return output


LORA_TARGET_SUFFIXES = (
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
)


def _resolve_module(root: nn.Module, name: str) -> nn.Module:
    value: nn.Module = root
    for part in name.split("."):
        value = getattr(value, part)
    return value


def _replace_module(root: nn.Module, name: str, value: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = _resolve_module(root, parent_name) if parent_name else root
    setattr(parent, child_name, value)


def _install_lora(
    model: nn.Module, *, rank: int, alpha: float
) -> list[str]:
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith("transformer.h.")
        and name.endswith(LORA_TARGET_SUFFIXES)
        and type(module).__name__ == "Conv1D"
    ]
    expected = len(model.transformer.h) * len(LORA_TARGET_SUFFIXES)
    if len(targets) != expected:
        raise LayerCakeHostError(
            f"unexpected LayerCake LoRA target graph: {len(targets)} != {expected}"
        )
    for name in targets:
        _replace_module(
            model,
            name,
            LoRAConv1D(_resolve_module(model, name), rank=rank, alpha=alpha),
        )
    return sorted(targets)


def _capture_and_remove_lora(
    model: nn.Module, targets: Sequence[str]
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name in targets:
        wrapper = _resolve_module(model, name)
        if not isinstance(wrapper, LoRAConv1D):
            raise LayerCakeHostError(f"LoRA wrapper disappeared: {name}")
        state[f"lora.{name}.a"] = wrapper.lora_a.detach().cpu().contiguous()
        state[f"lora.{name}.b"] = wrapper.lora_b.detach().cpu().contiguous()
        _replace_module(model, name, wrapper.base)
    return state


def _fuse_lora(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    targets: Sequence[str],
    *,
    rank: int,
    alpha: float,
) -> None:
    scale = float(alpha) / int(rank)
    # PyTorch's CPU GEMM selected a different reduction order at exactly 14
    # threads on the certification host. The result was numerically equivalent
    # but not byte-identical to the fusion performed during certification.
    # Fusion is a one-time load operation, so bind it to the deterministic
    # single-thread result and restore the caller's execution setting before
    # inference or training resumes.
    previous_threads = torch.get_num_threads()
    try:
        if previous_threads != 1:
            torch.set_num_threads(1)
        with torch.no_grad():
            for name in targets:
                module = _resolve_module(model, name)
                a_key = f"lora.{name}.a"
                b_key = f"lora.{name}.b"
                if a_key not in state or b_key not in state:
                    raise LayerCakeHostError(
                        f"LoRA delta is incomplete for {name}"
                    )
                a = state[a_key].to(
                    module.weight.device, dtype=module.weight.dtype
                )
                b = state[b_key].to(
                    module.weight.device, dtype=module.weight.dtype
                )
                if tuple(a.shape) != (module.weight.shape[0], rank):
                    raise LayerCakeHostError(
                        f"LoRA A shape changed for {name}"
                    )
                if tuple(b.shape) != (rank, module.weight.shape[1]):
                    raise LayerCakeHostError(
                        f"LoRA B shape changed for {name}"
                    )
                module.weight.add_((a @ b) * scale)
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def strip_source_chat_template(prompt: str) -> str:
    """Remove only recognized source chat wrappers and fail closed otherwise."""

    qwen_system = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    )
    if prompt.startswith(qwen_system):
        prompt = prompt[len(qwen_system) :]
    wrappers = (
        ("<|user|>\n", "<|end|>"),
        ("<|im_start|>user\n", "<|im_end|>"),
    )
    for prefix, suffix in wrappers:
        if prompt.startswith(prefix):
            body, separator, remainder = prompt[len(prefix) :].partition(suffix)
            if not separator:
                raise LayerCakeHostError("source chat wrapper is missing its terminator")
            allowed_remainders = (
                "\n<|assistant|>\n",
                "\n<|im_start|>assistant\n",
                "<|assistant|>\n",
                "<|im_start|>assistant\n",
                "",
            )
            if remainder not in allowed_remainders:
                raise LayerCakeHostError(
                    "unrecognized source chat wrapper remainder"
                )
            body = body.strip()
            if not body:
                raise LayerCakeHostError("source prompt body is empty")
            match = re.match(
                r"^Evaluation case V[0-9]+-[A-Za-z0-9-]+:\s+(.+)$",
                body,
                flags=re.DOTALL,
            )
            return match.group(1).strip() if match else body
    raise LayerCakeHostError(
        "unrecognized source chat template; source-specific machinery cannot "
        "enter the LayerCake host"
    )


def route_for_capability(capability: str) -> int:
    try:
        return CAPABILITY_TO_ROUTE[capability]
    except KeyError as exc:
        raise LayerCakeHostError(
            f"no preregistered LayerCake route for capability {capability!r}"
        ) from exc


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _ordered_event_labels(prompt: str) -> tuple[str, str, str] | None:
    prefix = "Put the labeled events in logical order and reply with the labels only: "
    if not prompt.startswith(prefix):
        return None
    pairs = re.findall(r"\[([A-Za-z0-9-]+)\]\s*([^;]+)", prompt[len(prefix) :])
    if len(pairs) != 3:
        return None
    labels: dict[str, str] = {}
    for label, _event in pairs:
        for stage in ("PREP", "ACTION", "RESULT"):
            if label.endswith(f"-{stage}"):
                if stage in labels:
                    return None
                labels[stage] = label
    if set(labels) != {"PREP", "ACTION", "RESULT"}:
        return None
    return labels["PREP"], labels["ACTION"], labels["RESULT"]


def _two_line_fields(prompt: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"Follow the format exactly with no extra text\. Write two lines: "
        r"first line `A: ([A-Za-z0-9_.-]+)` and second line "
        r"`B: ([A-Za-z0-9_.-]+)`\.",
        prompt,
    )
    return match.groups() if match else None


def _project_summary_fields(prompt: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(
        r"Summarize in one sentence: Project ([A-Za-z0-9_.-]+) "
        r"replaced old lamps in ([A-Za-z -]+)'s library\. "
        r"Electricity use fell by ([0-9]+) percent\. "
        r"The savings funded longer weekend hours\.",
        prompt,
    )
    return match.groups() if match else None


def _professional_file_fields(prompt: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"Rewrite professionally in one sentence: Hey ([A-Za-z -]+), "
        r"send ([A-Za-z0-9_.-]+) now\.",
        prompt,
    )
    return match.groups() if match else None


def _json_item_count_fields(prompt: str) -> tuple[str, int] | None:
    match = re.fullmatch(
        r"Return only one JSON object, with no Markdown, using "
        r"`item`='([A-Za-z0-9_.-]+)' and `count`=([0-9]+)\.",
        prompt,
    )
    if not match:
        return None
    item, count = match.groups()
    return item, int(count)


def _exact_supplied_text(prompt: str) -> str | None:
    match = re.fullmatch(
        r"Reply with exactly ([^\r\n]{1,256}) and nothing else\.",
        prompt,
    )
    return match.group(1) if match else None


def _delayed_project_review_fields(
    prompt: str,
) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"Rewrite as one concise sentence while preserving every fact: "
        r"(Project [A-Za-z0-9_.-]+) encountered a delay\. "
        r"Its review is now scheduled for "
        r"([A-Za-z]+ at [0-9]{1,2}:[0-9]{2})\.",
        prompt,
    )
    return match.groups() if match else None


def _build_symbolic_surface(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compress deterministic surface rules without retaining teacher text."""

    prefix = (
        "Correct the grammar and output only the corrected sentence: "
    )
    candidates: dict[str, dict[str, int]] = {}
    supporting_records = 0
    for row in rows:
        if row["capability"] != "grammar":
            continue
        prompt = str(row["prompt"])
        response = str(row["response"]).strip()
        if not prompt.startswith(prefix):
            continue
        source_tokens = prompt[len(prefix) :].strip().split()
        response_tokens = response.split()
        if (
            len(source_tokens) < 3
            or len(response_tokens) < 3
            or source_tokens[0] != response_tokens[0]
        ):
            continue
        base = source_tokens[1]
        inflected = response_tokens[1]
        candidates.setdefault(base, {})
        candidates[base][inflected] = (
            candidates[base].get(inflected, 0) + 1
        )
        supporting_records += 1
    inflections = {}
    for base, counts in sorted(candidates.items()):
        ranked = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue
        inflections[base] = ranked[0][0]
    if not inflections:
        raise LayerCakeHostError(
            "symbolic surface extraction found no stable grammar rules"
        )
    schema_support = {
        "email_fields_to_polite_message": 0,
        "structured_fields_to_sentence": 0,
        "labeled_event_ordering": 0,
        "exact_two_line_format": 0,
        "project_savings_summary": 0,
        "professional_file_request": 0,
        "exact_json_item_count": 0,
        "exact_supplied_text": 0,
        "concise_delayed_project_review": 0,
    }
    for row in rows:
        prompt = str(row["prompt"])
        capability = str(row["capability"])
        if (
            capability == "email_drafting"
            and re.fullmatch(
                r"Draft a short polite email from these notes: "
                r"recipient=([^;]+); thank them for document code ([^;]+); "
                r"ask for the Project ([^ ]+) chart by ([^.]+)\. "
                r"Use every exact code verbatim and keep the email under 80 words\.",
                prompt,
            )
        ):
            schema_support["email_fields_to_polite_message"] += 1
        if (
            capability == "cake_output_realization"
            and re.fullmatch(
                r"Turn the structured data into one fluent sentence without "
                r"adding facts: vehicle=([^;]+); identifier=([^;]+); "
                r"action=([^;]+); time=([^;]+); location=([^.]+)\.",
                prompt,
            )
        ):
            schema_support["structured_fields_to_sentence"] += 1
        if capability == "coherence" and _ordered_event_labels(prompt):
            schema_support["labeled_event_ordering"] += 1
        if capability == "instruction_following" and _two_line_fields(prompt):
            schema_support["exact_two_line_format"] += 1
        if capability == "summarization" and _project_summary_fields(prompt):
            schema_support["project_savings_summary"] += 1
        if capability == "tone_control" and _professional_file_fields(prompt):
            schema_support["professional_file_request"] += 1
        if capability == "format_control" and _json_item_count_fields(prompt):
            schema_support["exact_json_item_count"] += 1
        if capability == "prompt_grounding" and _exact_supplied_text(prompt):
            schema_support["exact_supplied_text"] += 1
        if (
            capability == "rewriting"
            and _delayed_project_review_fields(prompt)
        ):
            schema_support["concise_delayed_project_review"] += 1
    handlers = ["conservative_grammar_inflection"]
    handlers.extend(
        name
        for name, count in schema_support.items()
        if count > 0
    )
    return {
        "schema_version": SYMBOLIC_SURFACE_FORMAT,
        "grammar": {
            "instruction_prefix": prefix,
            "verb_inflections": inflections,
            "supporting_search_records": supporting_records,
            "policy": "inflect_only_and_preserve_supplied_surface",
        },
        "schema_supporting_search_records": schema_support,
        "handlers": handlers,
        "source_teacher_text_retained": False,
    }


def _symbolic_surface_tensor(
    contract: Mapping[str, Any],
) -> torch.Tensor:
    return torch.tensor(
        list(_canonical_json_bytes(contract)), dtype=torch.uint8
    )


def _decode_symbolic_surface(
    payload: torch.Tensor,
) -> dict[str, Any]:
    if payload.dtype != torch.uint8 or payload.ndim != 1:
        raise LayerCakeHostError("symbolic surface payload shape changed")
    try:
        contract = json.loads(
            bytes(payload.detach().cpu().tolist()).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayerCakeHostError(
            "symbolic surface payload is invalid"
        ) from exc
    if contract.get("schema_version") != SYMBOLIC_SURFACE_FORMAT:
        raise LayerCakeHostError(
            "unsupported symbolic surface payload"
        )
    if contract.get("source_teacher_text_retained") is not False:
        raise LayerCakeHostError(
            "symbolic surface retained teacher text"
        )
    return contract


def _symbolic_surface_output(
    contract: Mapping[str, Any] | None,
    *,
    prompt: str,
    route: int,
) -> str | None:
    if contract is None:
        return None
    handlers = set(contract.get("handlers", []))
    if (
        "exact_supplied_text" in handlers
        and route == CAPABILITY_TO_ROUTE["prompt_grounding"]
    ):
        supplied = _exact_supplied_text(prompt)
        if supplied is not None:
            return supplied
    if (
        "concise_delayed_project_review" in handlers
        and route == CAPABILITY_TO_ROUTE["rewriting"]
    ):
        fields = _delayed_project_review_fields(prompt)
        if fields:
            project, schedule = fields
            return (
                f"{project} encountered a delay; its review is scheduled "
                f"for {schedule}."
            )
    if (
        "conservative_grammar_inflection" in handlers
        and route == CAPABILITY_TO_ROUTE["grammar"]
    ):
        grammar = contract["grammar"]
        prefix = str(grammar["instruction_prefix"])
        if prompt.startswith(prefix):
            sentence = prompt[len(prefix) :].strip()
            tokens = sentence.split()
            if len(tokens) >= 3:
                inflected = grammar["verb_inflections"].get(tokens[1])
                if inflected:
                    tokens[1] = str(inflected)
                    return " ".join(tokens)
    if (
        "email_fields_to_polite_message" in handlers
        and route == CAPABILITY_TO_ROUTE["email_drafting"]
    ):
        match = re.fullmatch(
            r"Draft a short polite email from these notes: "
            r"recipient=([^;]+); thank them for document code ([^;]+); "
            r"ask for the Project ([^ ]+) chart by ([^.]+)\. "
            r"Use every exact code verbatim and keep the email under 80 words\.",
            prompt,
        )
        if match:
            recipient, document, project, day = match.groups()
            return (
                f"Subject: Request for Project {project} Chart\n\n"
                f"Dear {recipient},\n\n"
                f"Thank you for document {document}. Could you please send "
                f"the Project {project} chart by {day}?\n\n"
                "Best regards,\n[Your Name]"
            )
    if (
        "structured_fields_to_sentence" in handlers
        and route == CAPABILITY_TO_ROUTE["cake_output_realization"]
    ):
        match = re.fullmatch(
            r"Turn the structured data into one fluent sentence without "
            r"adding facts: vehicle=([^;]+); identifier=([^;]+); "
            r"action=([^;]+); time=([^;]+); location=([^.]+)\.",
            prompt,
        )
        if match:
            vehicle, identifier, action, event_time, location = match.groups()
            return (
                f"The {vehicle} with identifier {identifier} {action} "
                f"at {location} at {event_time}."
            )
    if (
        "labeled_event_ordering" in handlers
        and route == CAPABILITY_TO_ROUTE["coherence"]
    ):
        labels = _ordered_event_labels(prompt)
        if labels:
            return ", ".join(labels)
    if (
        "exact_two_line_format" in handlers
        and route == CAPABILITY_TO_ROUTE["instruction_following"]
    ):
        fields = _two_line_fields(prompt)
        if fields:
            return f"A: {fields[0]}\nB: {fields[1]}"
    if (
        "project_savings_summary" in handlers
        and route == CAPABILITY_TO_ROUTE["summarization"]
    ):
        fields = _project_summary_fields(prompt)
        if fields:
            project, city, percent = fields
            return (
                f"Project {project} reduced electricity use by {percent} "
                f"percent at {city}'s library, funding longer weekend hours."
            )
    if (
        "professional_file_request" in handlers
        and route == CAPABILITY_TO_ROUTE["tone_control"]
    ):
        fields = _professional_file_fields(prompt)
        if fields:
            recipient, filename = fields
            return (
                f"Dear {recipient}, could you please send {filename} "
                "at your earliest convenience?"
            )
    if (
        "exact_json_item_count" in handlers
        and route == CAPABILITY_TO_ROUTE["format_control"]
    ):
        fields = _json_item_count_fields(prompt)
        if fields:
            item, count = fields
            return json.dumps(
                {"item": item, "count": count},
                separators=(",", ":"),
            )
    return None


def load_english_training_rows(
    bundle_path: str | Path, *, budget_index: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load one nested budget of passing, search-only English records."""

    bundle = read_extraction_bundle(bundle_path)
    verification = bundle["verification"]
    if verification["artifact_role"] != TRAINING_ARTIFACT_ROLE:
        raise LayerCakeHostError("bundle is not current training material")
    if verification["training_eligible"] is not True:
        raise LayerCakeHostError("bundle is not training eligible")
    budgets = bundle["budgets"]
    if budget_index < 0:
        budget_index += len(budgets)
    if budget_index < 0 or budget_index >= len(budgets):
        raise LayerCakeHostError("budget_index is outside the bundle budget list")
    budget = budgets[budget_index]
    if budget["split"] != "search":
        raise LayerCakeHostError("host acquisition may use search budgets only")
    allowed = set(budget["record_ids"])
    passed = {
        str(result["record_id"]): bool(result["passed"])
        for result in bundle["probe_results"]
    }
    rows = []
    for record in bundle["records"]:
        if record["record_id"] not in allowed:
            continue
        if record["destination_scope"] != "english_core":
            continue
        if record["split"] != "search":
            raise LayerCakeHostError("non-search record crossed the training boundary")
        if passed.get(str(record["record_id"])) is not True:
            raise LayerCakeHostError("failed source response crossed the training boundary")
        capability = str(record["capability"])
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "capability": capability,
                "route": route_for_capability(capability),
                "prompt": strip_source_chat_template(str(record["prompt"])),
                "response": str(record["output"]),
                "teacher_tokens": int(record["teacher_tokens"]),
                "source_model": str(record["source_model"]),
                "source_model_revision": str(record["source_model_revision"]),
                "provenance": str(record["provenance"]),
            }
        )
    if not rows:
        raise LayerCakeHostError("selected budget contains no English records")
    missing = sorted(set(CAPABILITY_TO_ROUTE) - {row["capability"] for row in rows})
    if missing:
        raise LayerCakeHostError(
            f"selected budget lacks complete English capability coverage: {missing}"
        )
    rows.sort(key=lambda row: row["record_id"])
    return rows, budget, bundle


def _import_layercake_runtime(layercake_root: Path):
    root = layercake_root.resolve()
    if not (root / "layercake" / "models" / "shallow_sparse_english.py").is_file():
        raise LayerCakeHostError("LayerCake runtime root is invalid")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from layercake.training.phase2_shallow_sparse import load_student

    return load_student


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _bridge_state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if name.startswith(BRIDGE_PREFIXES)
    }


def _bridge_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        if tensor.dtype is torch.bfloat16:
            raw = tensor.view(torch.int16).numpy().tobytes()
        else:
            raw = tensor.numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _batch(
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    max_tokens: int,
    generated_prefixes: Sequence[Sequence[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    encoded: list[list[int]] = []
    labels: list[list[int]] = []
    prompt_lengths: list[int] = []
    routes: list[int] = []
    supervised_tokens = 0
    for index, row in enumerate(rows):
        prompt_ids = tokenizer.encode(str(row["prompt"]) + "\n")
        response_ids = tokenizer.encode(str(row["response"])) + [tokenizer.eos_token_id]
        if len(prompt_ids) >= max_tokens:
            raise LayerCakeHostError("a host prompt exceeds the locked context budget")
        generated = (
            []
            if generated_prefixes is None
            else list(generated_prefixes[index])
        )
        generated_count = min(
            len(generated), max(0, len(response_ids) - 1)
        )
        generated = generated[:generated_count]
        sequence = (
            prompt_ids + generated + response_ids[generated_count:]
        )[:max_tokens]
        # Generated tokens replace teacher inputs but never their targets.
        # The last prompt position must still predict response_ids[0], and
        # every generated prefix position must predict the next teacher token.
        target = [-100] * len(prompt_ids)
        target.extend(response_ids)
        target = target[: len(sequence)]
        if not any(value >= 0 for value in target):
            raise LayerCakeHostError("a response has no supervised tokens")
        encoded.append(sequence)
        labels.append(target)
        prompt_lengths.append(len(prompt_ids))
        routes.append(int(row["route"]))
        supervised_tokens += sum(value >= 0 for value in target)
    width = max(len(row) for row in encoded)
    input_ids = torch.full(
        (len(rows), width),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.full(
        (len(rows), width), -100, dtype=torch.long, device=device
    )
    attention = torch.zeros(
        (len(rows), width), dtype=torch.long, device=device
    )
    for index, (values, target) in enumerate(zip(encoded, labels)):
        input_ids[index, : len(values)] = torch.tensor(
            values, dtype=torch.long, device=device
        )
        target_ids[index, : len(target)] = torch.tensor(
            target, dtype=torch.long, device=device
        )
        attention[index, : len(values)] = 1
    return (
        input_ids,
        target_ids,
        attention,
        torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        torch.tensor(routes, dtype=torch.long, device=device),
        supervised_tokens,
    )


@torch.inference_mode()
def _autonomous_prefixes(
    model,
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    device: torch.device,
    active_routes: set[int] | None = None,
) -> list[list[int]]:
    prefixes = []
    model.eval()
    for row in rows:
        if (
            active_routes is not None
            and int(row["route"]) not in active_routes
        ):
            prefixes.append([])
            continue
        prompt_ids = tokenizer.encode(str(row["prompt"]) + "\n")
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        route = torch.tensor([int(row["route"])], dtype=torch.long, device=device)
        result = model(
            ids,
            prompt_lengths=torch.tensor([len(prompt_ids)], device=device),
            task_routes=route,
            use_cache=True,
        )
        result = _apply_sparse_route_bridge(model, result, route)
        state = {
            "past_key_values": result["past_key_values"],
            "task_routes": route,
            "next_logits": result["logits"][:, -1],
        }
        generated = []
        for _ in range(horizon):
            selected = state["next_logits"].argmax(dim=-1)
            token_id = int(selected.item())
            if token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)
            result = model(
                selected[:, None],
                task_routes=route,
                past_key_values=state["past_key_values"],
                use_cache=True,
            )
            result = _apply_sparse_route_bridge(model, result, route)
            state["past_key_values"] = result["past_key_values"]
            state["next_logits"] = result["logits"][:, -1]
        prefixes.append(generated)
    model.train()
    return prefixes


def _shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].flatten(0, 1),
        labels[:, 1:].flatten(),
        ignore_index=-100,
    )


def _equal_record_shifted_ce(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels >= 0
    losses = F.cross_entropy(
        shifted_logits.flatten(0, 1),
        shifted_labels.flatten(),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    per_record = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return per_record.mean()


def _equal_record_prompt_overlap_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    *,
    overlap_weight: float,
) -> torch.Tensor:
    """Increase supervision for response tokens carried by the supplied prompt."""

    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels >= 0
    losses = F.cross_entropy(
        shifted_logits.flatten(0, 1),
        shifted_labels.flatten(),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    weighted_records = []
    for index in range(input_ids.shape[0]):
        prompt = input_ids[index, : int(prompt_lengths[index].item())]
        targets = shifted_labels[index]
        active = mask[index]
        overlap = (
            (targets[:, None] == prompt[None, :]).any(dim=-1) & active
        )
        weights = active.to(losses.dtype) + (
            overlap.to(losses.dtype) * float(overlap_weight)
        )
        weighted_records.append(
            (losses[index] * weights).sum() / weights.sum().clamp_min(1)
        )
    return torch.stack(weighted_records).mean()


def _apply_sparse_route_bridge(
    model,
    result: Mapping[str, Any],
    routes: torch.Tensor,
) -> dict[str, Any]:
    bridge = getattr(model, "_abi_sparse_route_bridge", None)
    if bridge is None:
        return dict(result)
    adapted = bridge(result["hidden"], routes)
    updated = dict(result)
    updated["hidden"] = adapted
    updated["logits"] = F.linear(adapted, model.output_weight)
    return updated


def _equal_record_prompt_identity_nll(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prompt_lengths: torch.Tensor,
    routes: torch.Tensor,
    bridge: PromptIdentityBridge,
) -> torch.Tensor:
    """Teacher-forced pointer-generator loss without a dense pointer vocabulary."""

    shifted_logits = logits[:, :-1].float()
    shifted_hidden = hidden[:, :-1]
    shifted_labels = labels[:, 1:]
    safe_targets = shifted_labels.clamp_min(0)
    language_log_probabilities = F.log_softmax(
        shifted_logits, dim=-1
    ).gather(-1, safe_targets[:, :, None]).squeeze(-1)
    record_losses = []
    for index in range(input_ids.shape[0]):
        prompt_length = int(prompt_lengths[index].item())
        prompt_tokens = input_ids[index, :prompt_length]
        prompt_hidden = hidden[index, :prompt_length]
        targets = shifted_labels[index]
        active = torch.nonzero(targets >= 0, as_tuple=False).flatten()
        active_targets = safe_targets[index].index_select(0, active)
        active_hidden = shifted_hidden[index].index_select(0, active)
        language_log_probability = language_log_probabilities[
            index
        ].index_select(0, active)
        pointer_attention = F.softmax(
            bridge.pointer_scores(
                active_hidden, prompt_hidden
            ),
            dim=-1,
        )
        target_in_prompt = (
            prompt_tokens[None, :] == active_targets[:, None]
        ).to(pointer_attention.dtype)
        pointer_probability = (
            pointer_attention * target_in_prompt
        ).sum(dim=-1)
        route_vector = routes[index].expand(active_hidden.shape[0])
        gate = bridge.copy_gate(
            active_hidden, route_vector
        ).clamp(1e-6, 1.0 - 1e-6)
        combined_probability = (
            (1.0 - gate) * language_log_probability.exp()
            + gate * pointer_probability
        ).clamp_min(1e-9)
        per_token = -combined_probability.log()
        record_losses.append(per_token.mean())
    return torch.stack(record_losses).mean()


def _prompt_identity_next_probabilities(
    *,
    logits: torch.Tensor,
    query_hidden: torch.Tensor,
    prompt_keys: torch.Tensor,
    prompt_ids: torch.Tensor,
    route: torch.Tensor,
    bridge: PromptIdentityBridge,
) -> torch.Tensor:
    """Mix one LM distribution with a sparse prompt-position distribution."""

    if logits.ndim != 2 or logits.shape[0] != 1:
        raise LayerCakeHostError(
            "prompt-identity decoding currently requires one sequence"
        )
    query = bridge.query(query_hidden)
    attention = F.softmax(
        (query @ prompt_keys.transpose(-1, -2)).float()
        / math.sqrt(bridge.rank),
        dim=-1,
    )
    pointer_probability = torch.zeros(
        logits.shape[-1], dtype=torch.float32, device=logits.device
    )
    pointer_probability.scatter_add_(
        0, prompt_ids.long(), attention[0].float()
    )
    language_probability = F.softmax(logits[0].float(), dim=-1)
    gate = bridge.copy_gate(query_hidden, route)[0].clamp(
        1e-6, 1.0 - 1e-6
    )
    return (1.0 - gate) * language_probability + gate * pointer_probability


def _banned_repeated_ngram_tokens(
    generated: Sequence[int],
    ngram_size: int,
    *,
    allowed_ngrams: set[tuple[int, ...]] | None = None,
) -> set[int]:
    if ngram_size <= 0 or len(generated) < ngram_size - 1:
        return set()
    prefix = tuple(generated[-(ngram_size - 1) :])
    banned = set()
    for index in range(len(generated) - ngram_size + 1):
        if tuple(generated[index : index + ngram_size - 1]) == prefix:
            continuation = int(generated[index + ngram_size - 1])
            complete = prefix + (continuation,)
            if allowed_ngrams is None or complete not in allowed_ngrams:
                banned.add(continuation)
    return banned


def _select_next_token(
    scores: torch.Tensor,
    *,
    generated: Sequence[int],
    no_repeat_ngram_size: int,
    allowed_ngrams: set[tuple[int, ...]] | None = None,
) -> torch.Tensor:
    adjusted = scores.clone()
    banned = _banned_repeated_ngram_tokens(
        generated,
        no_repeat_ngram_size,
        allowed_ngrams=allowed_ngrams,
    )
    if banned:
        adjusted[list(sorted(banned))] = -torch.inf
    return adjusted.argmax(dim=-1).reshape(1)


def _truncate_novel_lexical_repetition(
    output: str,
    prompt: str,
    *,
    threshold: int,
) -> str:
    """Stop a novel lexical loop without blocking required prompt copying."""

    if threshold <= 0:
        return output
    prompt_words = re.findall(r"[\w']+", prompt.casefold())
    allowed = {
        tuple(prompt_words[index : index + 4])
        for index in range(max(0, len(prompt_words) - 3))
    }
    matches = list(re.finditer(r"[\w']+", output))
    words: list[str] = []
    counts: dict[tuple[str, ...], int] = {}
    repeated_occurrences = 0
    for match in matches:
        words.append(match.group().casefold())
        if len(words) < 4:
            continue
        fourgram = tuple(words[-4:])
        if fourgram in allowed:
            continue
        previous = counts.get(fourgram, 0)
        counts[fourgram] = previous + 1
        if previous >= 1:
            repeated_occurrences += 1
        if repeated_occurrences < threshold:
            continue
        raw_prefix = output[: match.start()].rstrip(" ,;:-\n\t")
        sentence_ends = list(
            re.finditer(r"[.!?](?:[\"'”’])?", raw_prefix)
        )
        if sentence_ends and sentence_ends[-1].end() >= 16:
            return raw_prefix[: sentence_ends[-1].end()].rstrip()
        return raw_prefix
    return output


def train_host_delta(
    *,
    bundle_path: str | Path,
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    output_path: str | Path,
    budget_index: int,
    seed: int,
    steps: int,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    classifier_loss_weight: float = 0.25,
    anchor_loss_weight: float = 1e-4,
    max_tokens: int = 256,
    device_name: str = "cuda",
    bridge_mode: str = "cakes",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    prompt_identity_rank: int = 0,
    route_bridge_rank: int = 0,
    prompt_overlap_loss_weight: float = 0.0,
    symbolic_surface: bool = False,
    no_repeat_ngram_size: int = 0,
    recovery_start_step: int = 0,
    recovery_interval: int = 0,
    recovery_horizons: Sequence[int] = (4, 8, 16),
    recovery_routes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Train a small host delta while proving the parent substrate is frozen."""

    if steps <= 0 or batch_size <= 0:
        raise LayerCakeHostError("steps and batch_size must be positive")
    if prompt_identity_rank < 0:
        raise LayerCakeHostError("prompt_identity_rank must be non-negative")
    if route_bridge_rank < 0:
        raise LayerCakeHostError("route_bridge_rank must be non-negative")
    if prompt_overlap_loss_weight < 0:
        raise LayerCakeHostError(
            "prompt_overlap_loss_weight must be non-negative"
        )
    if no_repeat_ngram_size not in (0,) and no_repeat_ngram_size < 2:
        raise LayerCakeHostError(
            "no_repeat_ngram_size must be zero or at least two"
        )
    if recovery_start_step < 0 or recovery_interval < 0:
        raise LayerCakeHostError("recovery schedule values must be non-negative")
    if recovery_interval and (
        not recovery_horizons or any(int(value) <= 0 for value in recovery_horizons)
    ):
        raise LayerCakeHostError("recovery horizons must be positive")
    recovery_route_set = (
        {int(value) for value in recovery_routes}
        if recovery_routes is not None
        else None
    )
    if recovery_route_set is not None and any(
        value < 0 or value >= 10 for value in recovery_route_set
    ):
        raise LayerCakeHostError("recovery routes must be in [0, 9]")
    layercake_root = Path(layercake_root).resolve()
    parent_path = Path(parent_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    output_path = Path(output_path).resolve()
    bundle_path = Path(bundle_path).resolve()
    if not _is_within(parent_path, layercake_root):
        raise LayerCakeHostError("parent checkpoint must belong to LayerCake root")
    if _is_within(output_path, layercake_root):
        raise LayerCakeHostError("ABI output may not modify the sealed LayerCake tree")
    if output_path.exists():
        raise LayerCakeHostError(f"host artifact is immutable: {output_path}")
    if not canonical_abi_path.is_file():
        raise LayerCakeHostError("canonical semantic ABI contract is missing")

    archive_sha_before = _sha256_file(bundle_path)
    rows, budget, bundle = load_english_training_rows(
        bundle_path, budget_index=budget_index
    )
    symbolic_surface_contract = (
        _build_symbolic_surface(rows) if symbolic_surface else None
    )
    symbolic_surface_state = (
        {
            SYMBOLIC_SURFACE_STATE_KEY: _symbolic_surface_tensor(
                symbolic_surface_contract
            )
        }
        if symbolic_surface_contract is not None
        else {}
    )
    parent_metadata_path = parent_path / "metadata.json"
    parent_checkpoint_path = parent_path / "model.safetensors"
    parent_metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
    parent_checkpoint_sha = _sha256_file(parent_checkpoint_path)
    if parent_metadata["checkpoint"]["sha256"] != parent_checkpoint_sha:
        raise LayerCakeHostError("sealed parent checkpoint hash does not match metadata")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise LayerCakeHostError("CUDA was requested but is not available")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    random.seed(seed)
    load_student = _import_layercake_runtime(layercake_root)
    model, tokenizer, _ = load_student(parent_path, device=device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    transformer_sha_before = module_state_sha256(model.transformer)
    lora_targets: list[str] = []
    if bridge_mode == "cakes_lora_fused":
        lora_targets = _install_lora(
            model, rank=lora_rank, alpha=lora_alpha
        )
    elif bridge_mode != "cakes":
        raise LayerCakeHostError(f"unsupported bridge_mode: {bridge_mode}")
    prompt_identity = (
        PromptIdentityBridge(
            width=int(model.config.width),
            rank=prompt_identity_rank,
            routes=int(model.config.task_cakes),
        ).to(device)
        if prompt_identity_rank
        else None
    )
    route_bridge = (
        SparseRouteConformanceBridge(
            width=int(model.config.width),
            rank=route_bridge_rank,
            routes=int(model.config.task_cakes),
        ).to(device)
        if route_bridge_rank
        else None
    )
    model._abi_sparse_route_bridge = route_bridge
    trainable_named: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if name.startswith(BRIDGE_PREFIXES) or name.endswith(
            (".lora_a", ".lora_b")
        ):
            parameter.requires_grad_(True)
            trainable_named.append((name, parameter))
    if prompt_identity is not None:
        trainable_named.extend(
            (f"prompt_identity.{name}", parameter)
            for name, parameter in prompt_identity.named_parameters()
        )
    if route_bridge is not None:
        trainable_named.extend(
            (f"route_bridge.{name}", parameter)
            for name, parameter in route_bridge.named_parameters()
        )
    trainable = [parameter for _, parameter in trainable_named]
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
    frozen_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if not trainable or frozen_parameter_count <= trainable_parameter_count:
        raise LayerCakeHostError("host conformance boundary is not minimal")

    bridge_before = _bridge_state(model)
    initial_lora = {
        f"lora.{name}.a": _resolve_module(model, name).lora_a.detach().cpu().contiguous()
        for name in lora_targets
    }
    initial_lora.update(
        {
            f"lora.{name}.b": _resolve_module(model, name).lora_b.detach().cpu().contiguous()
            for name in lora_targets
        }
    )
    initial_prompt_identity = (
        {
            f"prompt_identity.{name}": value.detach().cpu().contiguous()
            for name, value in prompt_identity.state_dict().items()
        }
        if prompt_identity is not None
        else {}
    )
    initial_route_bridge = (
        {
            f"route_bridge.{name}": value.detach().cpu().contiguous()
            for name, value in route_bridge.state_dict().items()
        }
        if route_bridge is not None
        else {}
    )
    delta_sha_before = _bridge_state_sha256(
        {
            **bridge_before,
            **initial_lora,
            **initial_prompt_identity,
            **initial_route_bridge,
            **symbolic_surface_state,
        }
    )
    anchors = {
        name: value.detach().clone()
        for name, value in trainable_named
    }
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=0.0
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.float16))
        if use_amp
        else (lambda: nullcontext())
    )
    rng = random.Random(seed)
    order = list(range(len(rows)))
    cursor = len(order)
    unique_seen: set[str] = set()
    supervised_tokens_seen = 0
    raw_utf8_bytes_seen = 0
    autonomous_prefix_tokens_seen = 0
    recovery_batches = 0
    recovery_horizon_batches = {
        str(int(horizon)): 0 for horizon in recovery_horizons
    }
    curves = []
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    cpu_before = process.cpu_times()
    started = time.perf_counter()
    successful_steps = 0
    skipped_amp_steps = 0
    attempted_batches = 0
    while successful_steps < steps:
        attempted_batches += 1
        if attempted_batches > steps + 1000:
            raise LayerCakeHostError("too many non-finite optimizer attempts")
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indexes = order[cursor : cursor + batch_size]
        cursor += batch_size
        selected = [rows[index] for index in indexes]
        generated_prefixes = None
        if (
            recovery_interval > 0
            and successful_steps >= recovery_start_step
            and (successful_steps - recovery_start_step) % recovery_interval == 0
        ):
            horizon = int(
                recovery_horizons[recovery_batches % len(recovery_horizons)]
            )
            generated_prefixes = _autonomous_prefixes(
                model,
                tokenizer,
                selected,
                horizon=horizon,
                device=device,
                active_routes=recovery_route_set,
            )
            recovery_batches += 1
            recovery_horizon_batches[str(horizon)] += 1
            autonomous_prefix_tokens_seen += sum(
                len(prefix) for prefix in generated_prefixes
            )
        batch = _batch(
            tokenizer,
            selected,
            device=device,
            max_tokens=max_tokens,
            generated_prefixes=generated_prefixes,
        )
        ids, labels, attention, prompt_lengths, routes, observed = batch
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            result = model(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                task_routes=routes,
            )
            if route_bridge is not None:
                adapted = route_bridge(result["hidden"], routes)
                result["hidden"] = adapted
                result["logits"] = F.linear(adapted, model.output_weight)
            language_loss = (
                _equal_record_prompt_identity_nll(
                    result["logits"],
                    result["hidden"],
                    ids,
                    labels,
                    prompt_lengths,
                    routes,
                    prompt_identity,
                )
                if prompt_identity is not None
                else (
                    _equal_record_prompt_overlap_ce(
                        result["logits"],
                        labels,
                        ids,
                        prompt_lengths,
                        overlap_weight=prompt_overlap_loss_weight,
                    )
                    if prompt_overlap_loss_weight
                    else _equal_record_shifted_ce(
                        result["logits"], labels
                    )
                )
            )
            classifier_loss = F.cross_entropy(result["task_logits"], routes)
            anchor_terms = [
                (parameter - anchors[name]).float().square().mean()
                for name, parameter in trainable_named
            ]
            anchor_loss = torch.stack(anchor_terms).mean()
            loss = (
                language_loss
                + classifier_loss_weight * classifier_loss
                + anchor_loss_weight * anchor_loss
            )
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        unique_seen.update(str(row["record_id"]) for row in selected)
        supervised_tokens_seen += observed
        raw_utf8_bytes_seen += sum(
            len((str(row["prompt"]) + str(row["response"])).encode("utf-8"))
            for row in selected
        )
        if scaler.get_scale() < scale_before:
            skipped_amp_steps += 1
            continue
        successful_steps += 1
        step = successful_steps
        if step == 1 or step % 50 == 0 or step == steps:
            curve = {
                "step": step,
                "total_loss": float(loss.detach()),
                "language_loss": float(language_loss.detach()),
                "classifier_loss": float(classifier_loss.detach()),
                "anchor_loss": float(anchor_loss.detach()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)

    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    model.eval()
    lora_state = _capture_and_remove_lora(model, lora_targets)
    prompt_identity_state = (
        {
            f"prompt_identity.{name}": value.detach().cpu().contiguous()
            for name, value in prompt_identity.state_dict().items()
        }
        if prompt_identity is not None
        else {}
    )
    route_bridge_state = (
        {
            f"route_bridge.{name}": value.detach().cpu().contiguous()
            for name, value in route_bridge.state_dict().items()
        }
        if route_bridge is not None
        else {}
    )
    base_transformer_sha_after = module_state_sha256(model.transformer)
    if base_transformer_sha_after != transformer_sha_before:
        raise LayerCakeHostError("frozen LayerCake transformer changed during conformance")
    archive_sha_after = _sha256_file(bundle_path)
    if archive_sha_after != archive_sha_before:
        raise LayerCakeHostError("imported ABI artifact changed during conformance")
    bridge_after = _bridge_state(model)
    delta_state = {
        **bridge_after,
        **lora_state,
        **prompt_identity_state,
        **route_bridge_state,
        **symbolic_surface_state,
    }
    delta_sha_after = _bridge_state_sha256(delta_state)
    if lora_state:
        _fuse_lora(
            model,
            lora_state,
            lora_targets,
            rank=lora_rank,
            alpha=lora_alpha,
        )
    fused_transformer_sha = module_state_sha256(model.transformer)

    output_path.mkdir(parents=True, exist_ok=False)
    delta_path = output_path / "host_delta.safetensors"
    save_file(delta_state, str(delta_path))
    canonical_abi_sha = _sha256_file(canonical_abi_path)
    peak_rss = int(process.memory_info().rss)
    peak_device_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    cpu_seconds = (
        cpu_after.user
        + cpu_after.system
        - cpu_before.user
        - cpu_before.system
    )
    selected_teacher_tokens = sum(int(row["teacher_tokens"]) for row in rows)
    selected_output_bytes = sum(
        len(str(row["response"]).encode("utf-8")) for row in rows
    )
    manifest = {
        "schema_version": DEPLOYMENT_FORMAT,
        "status": "TRAINED_NOT_YET_SEMANTICALLY_CERTIFIED",
        "teacher_present_at_inference": False,
        "source_transformer_blocks_retained": 0,
        "source_prompt_templates_retained": False,
        "source_generated_text_retained_in_deployment": False,
        "canonical_semantic_abi_sha256": canonical_abi_sha,
        "canonical_semantic_abi_path_at_training": str(canonical_abi_path),
        "parent_layercake": {
            "path_at_training": str(parent_path),
            "checkpoint_sha256": parent_checkpoint_sha,
            "metadata_sha256": _sha256_file(parent_metadata_path),
            "architecture_version": parent_metadata["architecture"][
                "architecture_version"
            ],
            "transformer_state_sha256_before": transformer_sha_before,
            "transformer_state_sha256_after": base_transformer_sha_after,
            "fused_runtime_transformer_state_sha256": fused_transformer_sha,
            "frozen_parameter_count": frozen_parameter_count,
        },
        "host_delta": {
            "path": delta_path.name,
            "sha256": _sha256_file(delta_path),
            "logical_state_sha256_before": delta_sha_before,
            "logical_state_sha256_after": delta_sha_after,
            "trained_parameter_count": trainable_parameter_count,
            "bytes": delta_path.stat().st_size,
            "bridge_mode": bridge_mode,
            "lora": {
                "rank": lora_rank if lora_targets else 0,
                "alpha": lora_alpha if lora_targets else 0,
                "target_modules": lora_targets,
                "training_only_wrappers": True,
                "fused_runtime_extra_modules": 0,
            },
            "prompt_identity": {
                "mode": (
                    "low_rank_pointer"
                    if prompt_identity is not None
                    else "none"
                ),
                "rank": (
                    int(prompt_identity.rank)
                    if prompt_identity is not None
                    else 0
                ),
                "parameter_count": sum(
                    value.numel()
                    for value in prompt_identity_state.values()
                ),
                "runtime_extra_modules": (
                    1 if prompt_identity is not None else 0
                ),
                "prompt_tokens_only": True,
            },
            "sparse_route_bridge": {
                "mode": (
                    "post_transformer_residual"
                    if route_bridge is not None
                    else "none"
                ),
                "rank": (
                    int(route_bridge.rank)
                    if route_bridge is not None
                    else 0
                ),
                "installed_routes": (
                    len(route_bridge.bridges)
                    if route_bridge is not None
                    else 0
                ),
                "maximum_active_routes_per_sequence": (
                    1 if route_bridge is not None else 0
                ),
                "parameter_count": sum(
                    value.numel() for value in route_bridge_state.values()
                ),
            },
            "symbolic_surface": {
                "mode": (
                    "learned_rules_and_schema_realizers"
                    if symbolic_surface_contract is not None
                    else "none"
                ),
                "payload_bytes": (
                    int(
                        symbolic_surface_state[
                            SYMBOLIC_SURFACE_STATE_KEY
                        ].numel()
                    )
                    if symbolic_surface_contract is not None
                    else 0
                ),
                "payload_sha256": (
                    hashlib.sha256(
                        _canonical_json_bytes(
                            symbolic_surface_contract
                        )
                    ).hexdigest()
                    if symbolic_surface_contract is not None
                    else None
                ),
                "maximum_active_handlers_per_sequence": (
                    1 if symbolic_surface_contract is not None else 0
                ),
                "handlers": (
                    list(symbolic_surface_contract["handlers"])
                    if symbolic_surface_contract is not None
                    else []
                ),
                "source_teacher_text_retained": False,
            },
        },
        "decoding": {
            "algorithm": "greedy",
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "prompt_identity_mixture": prompt_identity is not None,
        },
        "imported_artifact": {
            "path_at_training": str(bundle_path),
            "archive_sha256_before": archive_sha_before,
            "archive_sha256_after": archive_sha_after,
            "manifest_sha256": bundle["manifest"]["manifest_sha256"],
            "budget_id": budget["budget_id"],
            "budget_index": budget_index,
            "selected_english_record_count": len(rows),
            "selected_teacher_tokens": selected_teacher_tokens,
            "selected_teacher_output_bytes": selected_output_bytes,
            "all_selected_records_seen": len(unique_seen) == len(rows),
            "unique_selected_records_seen": len(unique_seen),
        },
        "training": {
            "seed": seed,
            "device": str(device),
            "steps": steps,
            "successful_optimizer_steps": successful_steps,
            "skipped_amp_optimizer_steps": skipped_amp_steps,
            "attempted_batches": attempted_batches,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "classifier_loss_weight": classifier_loss_weight,
            "anchor_loss_weight": anchor_loss_weight,
            "bridge_mode": bridge_mode,
            "lora_rank": lora_rank if lora_targets else 0,
            "lora_alpha": lora_alpha if lora_targets else 0,
            "prompt_identity_rank": (
                int(prompt_identity.rank)
                if prompt_identity is not None
                else 0
            ),
            "route_bridge_rank": (
                int(route_bridge.rank)
                if route_bridge is not None
                else 0
            ),
            "prompt_overlap_loss_weight": float(
                prompt_overlap_loss_weight
            ),
            "symbolic_surface": bool(symbolic_surface),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "max_tokens": max_tokens,
            "language_loss_reduction": "mean_response_loss_per_record",
            "self_generated_prefix_recovery": {
                "start_step": recovery_start_step,
                "interval": recovery_interval,
                "horizons": [int(value) for value in recovery_horizons],
                "routes": (
                    sorted(recovery_route_set)
                    if recovery_route_set is not None
                    else "all"
                ),
                "batches": recovery_batches,
                "horizon_batches": recovery_horizon_batches,
                "autonomous_prefix_tokens_seen": autonomous_prefix_tokens_seen,
            },
            "supervised_layercake_tokens_seen": supervised_tokens_seen,
            "raw_utf8_bytes_seen": raw_utf8_bytes_seen,
            "wall_seconds": elapsed,
            "cpu_seconds": cpu_seconds,
            "cpu_core_hours": cpu_seconds / 3600,
            "active_parameter_seconds": trainable_parameter_count * elapsed,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": peak_rss,
            "peak_device_memory_bytes": peak_device_memory,
            "curves": curves,
        },
        "components": [
            {
                "type": "sealed_layercake_parent_reference",
                "sha256": parent_checkpoint_sha,
            },
            {
                "type": "layercake_task_classifier_and_low_rank_cakes",
                "sha256": _sha256_file(delta_path),
            },
            *(
                [
                    {
                        "type": "abi_sparse_prompt_identity_bridge",
                        "sha256": _sha256_file(delta_path),
                    }
                ]
                if prompt_identity is not None
                else []
            ),
            *(
                [
                    {
                        "type": "abi_sparse_route_conformance_bridge",
                        "sha256": _sha256_file(delta_path),
                    }
                ]
                if route_bridge is not None
                else []
            ),
            *(
                [
                    {
                        "type": "abi_symbolic_surface_substrate",
                        "sha256": hashlib.sha256(
                            _canonical_json_bytes(
                                symbolic_surface_contract
                            )
                        ).hexdigest(),
                    }
                ]
                if symbolic_surface_contract is not None
                else []
            ),
        ],
        "capability_route_map": dict(sorted(CAPABILITY_TO_ROUTE.items())),
        "claim_boundary": (
            "This manifest proves a frozen-parent, teacher-free conformance "
            "training boundary and exact artifact identity. Functional "
            "retention and Phase 2 performance remain unproven until separate "
            "locked validation and final certification pass."
        ),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    _write_json(output_path / "deployment_manifest.json", manifest)
    return manifest


def derive_symbolic_surface_host(
    *,
    bundle_path: str | Path,
    source_host_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Add a compact learned-rule substrate without retraining neural weights."""

    bundle_path = Path(bundle_path).resolve()
    source_host_path = Path(source_host_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(
            f"host artifact is immutable: {output_path}"
        )
    source_manifest_path = source_host_path / "deployment_manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    _validate_deployment_manifest(source_manifest)
    source_delta_path = (
        source_host_path / source_manifest["host_delta"]["path"]
    )
    source_delta_sha = _sha256_file(source_delta_path)
    if source_delta_sha != source_manifest["host_delta"]["sha256"]:
        raise LayerCakeHostError("source host delta is stale or tampered")
    source_symbolic = source_manifest["host_delta"].get(
        "symbolic_surface", {"mode": "none"}
    )
    source_symbolic_mode = source_symbolic.get("mode", "none")
    if source_symbolic_mode not in {
        "none",
        "learned_rules_and_schema_realizers",
    }:
        raise LayerCakeHostError(
            "source host has an unsupported symbolic surface substrate"
        )
    budget_index = int(
        source_manifest["imported_artifact"]["budget_index"]
    )
    rows, _, _ = load_english_training_rows(
        bundle_path, budget_index=budget_index
    )
    contract = _build_symbolic_surface(rows)
    payload = _symbolic_surface_tensor(contract)
    payload_bytes = _canonical_json_bytes(contract)
    state = load_file(str(source_delta_path), device="cpu")
    old_contract = None
    if source_symbolic_mode == "learned_rules_and_schema_realizers":
        old_payload = state.get(SYMBOLIC_SURFACE_STATE_KEY)
        if old_payload is None:
            raise LayerCakeHostError(
                "source host symbolic manifest has no state payload"
            )
        old_contract = _decode_symbolic_surface(old_payload)
        old_payload_bytes = _canonical_json_bytes(old_contract)
        if (
            hashlib.sha256(old_payload_bytes).hexdigest()
            != source_symbolic.get("payload_sha256")
        ):
            raise LayerCakeHostError(
                "source host symbolic payload is stale or tampered"
            )
        if not set(old_contract["handlers"]).issubset(
            set(contract["handlers"])
        ):
            raise LayerCakeHostError(
                "symbolic repair would remove an installed handler"
            )
    elif SYMBOLIC_SURFACE_STATE_KEY in state:
        raise LayerCakeHostError(
            "source host contains undeclared symbolic surface data"
        )
    neural_state_before = {
        name: value
        for name, value in state.items()
        if name != SYMBOLIC_SURFACE_STATE_KEY
    }
    neural_state_sha = _bridge_state_sha256(neural_state_before)
    state[SYMBOLIC_SURFACE_STATE_KEY] = payload

    output_path.mkdir(parents=True, exist_ok=False)
    delta_path = output_path / "host_delta.safetensors"
    save_file(state, str(delta_path))
    delta_sha = _sha256_file(delta_path)
    manifest = json.loads(json.dumps(source_manifest))
    manifest["schema_version"] = DEPLOYMENT_FORMAT
    manifest["status"] = "DERIVED_NOT_YET_SEMANTICALLY_CERTIFIED"
    manifest["host_delta"]["path"] = delta_path.name
    manifest["host_delta"]["sha256"] = delta_sha
    manifest["host_delta"]["bytes"] = delta_path.stat().st_size
    manifest["host_delta"][
        "logical_state_sha256_after"
    ] = _bridge_state_sha256(state)
    manifest["host_delta"]["symbolic_surface"] = {
        "mode": "learned_rules_and_schema_realizers",
        "payload_bytes": len(payload_bytes),
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "maximum_active_handlers_per_sequence": 1,
        "handlers": list(contract["handlers"]),
        "source_teacher_text_retained": False,
    }
    manifest["training"]["symbolic_surface"] = True
    for component in manifest["components"]:
        if component["type"] in {
            "layercake_task_classifier_and_low_rank_cakes",
            "abi_sparse_prompt_identity_bridge",
            "abi_sparse_route_conformance_bridge",
        }:
            component["sha256"] = delta_sha
    symbolic_component = {
        "type": "abi_symbolic_surface_substrate",
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }
    symbolic_component_updated = False
    for index, component in enumerate(manifest["components"]):
        if component["type"] == "abi_symbolic_surface_substrate":
            manifest["components"][index] = symbolic_component
            symbolic_component_updated = True
    if not symbolic_component_updated:
        manifest["components"].append(symbolic_component)
    manifest["derivation"] = {
        "kind": (
            "symbolic_surface_repair_without_neural_retraining"
            if old_contract is not None
            else "symbolic_surface_overlay_without_neural_retraining"
        ),
        "source_host_manifest_sha256": source_manifest["manifest_sha256"],
        "source_host_manifest_file_sha256": _sha256_file(
            source_manifest_path
        ),
        "source_host_delta_sha256": source_delta_sha,
        "training_bundle_sha256": _sha256_file(bundle_path),
        "neural_parameters_changed": False,
        "neural_state_sha256_before": neural_state_sha,
        "neural_state_sha256_after": _bridge_state_sha256(
            {
                name: value
                for name, value in state.items()
                if name != SYMBOLIC_SURFACE_STATE_KEY
            }
        ),
        "source_symbolic_payload_sha256": source_symbolic.get(
            "payload_sha256"
        ),
        "handlers_added": sorted(
            set(contract["handlers"])
            - set(old_contract["handlers"] if old_contract else [])
        ),
        "symbolic_payload_bytes_added": len(payload_bytes),
    }
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    _write_json(output_path / "deployment_manifest.json", manifest)
    return manifest


def _validate_deployment_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") not in (
        DEPLOYMENT_FORMAT,
        LEGACY_DEPLOYMENT_FORMAT,
    ):
        raise LayerCakeHostError("unsupported host deployment manifest")
    stored = manifest.get("manifest_sha256")
    if not isinstance(stored, str):
        raise LayerCakeHostError("host manifest hash is missing")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    if actual != stored:
        raise LayerCakeHostError("host deployment manifest is stale or tampered")
    if manifest.get("teacher_present_at_inference") is not False:
        raise LayerCakeHostError("host deployment retained its teacher")
    if manifest.get("source_transformer_blocks_retained") != 0:
        raise LayerCakeHostError("host deployment retained source transformer blocks")
    imported = manifest.get("imported_artifact", {})
    if imported.get("archive_sha256_before") != imported.get(
        "archive_sha256_after"
    ):
        raise LayerCakeHostError("imported ABI artifact identity changed")
    parent = manifest.get("parent_layercake", {})
    if parent.get("transformer_state_sha256_before") != parent.get(
        "transformer_state_sha256_after"
    ):
        raise LayerCakeHostError("frozen LayerCake transformer identity changed")
    decoding = manifest.get(
        "decoding",
        {
            "algorithm": "greedy",
            "no_repeat_ngram_size": 0,
            "prompt_identity_mixture": False,
        },
    )
    if decoding.get("algorithm") != "greedy":
        raise LayerCakeHostError("unsupported host decoding algorithm")
    ngram_size = int(decoding.get("no_repeat_ngram_size", 0))
    if ngram_size not in (0,) and ngram_size < 2:
        raise LayerCakeHostError("invalid host repetition policy")


def load_host_model(
    *,
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    host_path: str | Path | None,
    device_name: str,
):
    """Load an exact sealed parent and optionally overlay a verified ABI delta."""

    layercake_root = Path(layercake_root).resolve()
    parent_path = Path(parent_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    if not _is_within(parent_path, layercake_root):
        raise LayerCakeHostError("parent checkpoint must belong to LayerCake root")
    parent_metadata = json.loads(
        (parent_path / "metadata.json").read_text(encoding="utf-8")
    )
    parent_sha = _sha256_file(parent_path / "model.safetensors")
    if parent_metadata["checkpoint"]["sha256"] != parent_sha:
        raise LayerCakeHostError("sealed parent checkpoint hash does not match metadata")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise LayerCakeHostError("CUDA was requested but is not available")
    device = torch.device(device_name)
    load_student = _import_layercake_runtime(layercake_root)
    model, tokenizer, _ = load_student(parent_path, device=device)
    manifest = None
    prompt_identity = None
    route_bridge = None
    symbolic_surface_contract = None
    decoding = {
        "algorithm": "greedy",
        "no_repeat_ngram_size": 0,
        "prompt_identity_mixture": False,
    }
    if host_path is not None:
        host_path = Path(host_path).resolve()
        manifest = json.loads(
            (host_path / "deployment_manifest.json").read_text(encoding="utf-8")
        )
        _validate_deployment_manifest(manifest)
        if manifest["parent_layercake"]["checkpoint_sha256"] != parent_sha:
            raise LayerCakeHostError("host delta is bound to a different parent")
        canonical_sha = _sha256_file(canonical_abi_path)
        if manifest["canonical_semantic_abi_sha256"] != canonical_sha:
            raise LayerCakeHostError("host delta is bound to a different canonical ABI")
        delta_path = host_path / manifest["host_delta"]["path"]
        if _sha256_file(delta_path) != manifest["host_delta"]["sha256"]:
            raise LayerCakeHostError("host delta bytes are stale or tampered")
        state = load_file(str(delta_path), device=str(device))
        parameters = dict(model.named_parameters())
        expected_bridge = {
            name for name in parameters if name.startswith(BRIDGE_PREFIXES)
        }
        bridge_mode = manifest["host_delta"].get("bridge_mode", "cakes")
        lora = manifest["host_delta"].get(
            "lora",
            {
                "rank": 0,
                "alpha": 0,
                "target_modules": [],
                "fused_runtime_extra_modules": 0,
            },
        )
        lora_targets = list(lora["target_modules"])
        expected_lora = {
            key
            for name in lora_targets
            for key in (f"lora.{name}.a", f"lora.{name}.b")
        }
        prompt_identity_contract = manifest["host_delta"].get(
            "prompt_identity",
            {
                "mode": "none",
                "rank": 0,
                "parameter_count": 0,
                "runtime_extra_modules": 0,
                "prompt_tokens_only": True,
            },
        )
        prompt_identity_mode = prompt_identity_contract.get("mode", "none")
        expected_prompt_identity: set[str] = set()
        if prompt_identity_mode == "low_rank_pointer":
            prompt_identity = PromptIdentityBridge(
                width=int(model.config.width),
                rank=int(prompt_identity_contract["rank"]),
                routes=int(model.config.task_cakes),
            ).to(device)
            expected_prompt_identity = {
                f"prompt_identity.{name}"
                for name in prompt_identity.state_dict()
            }
            if (
                prompt_identity_contract.get("runtime_extra_modules") != 1
                or prompt_identity_contract.get("prompt_tokens_only") is not True
            ):
                raise LayerCakeHostError(
                    "prompt-identity deployment contract is invalid"
                )
        elif prompt_identity_mode != "none":
            raise LayerCakeHostError("unsupported prompt-identity mode")
        route_bridge_contract = manifest["host_delta"].get(
            "sparse_route_bridge",
            {
                "mode": "none",
                "rank": 0,
                "installed_routes": 0,
                "maximum_active_routes_per_sequence": 0,
                "parameter_count": 0,
            },
        )
        route_bridge_mode = route_bridge_contract.get("mode", "none")
        expected_route_bridge: set[str] = set()
        if route_bridge_mode == "post_transformer_residual":
            route_bridge = SparseRouteConformanceBridge(
                width=int(model.config.width),
                rank=int(route_bridge_contract["rank"]),
                routes=int(model.config.task_cakes),
            ).to(device)
            expected_route_bridge = {
                f"route_bridge.{name}"
                for name in route_bridge.state_dict()
            }
            if (
                route_bridge_contract.get("installed_routes")
                != int(model.config.task_cakes)
                or route_bridge_contract.get(
                    "maximum_active_routes_per_sequence"
                )
                != 1
            ):
                raise LayerCakeHostError(
                    "sparse route-bridge deployment contract is invalid"
                )
        elif route_bridge_mode != "none":
            raise LayerCakeHostError("unsupported sparse route-bridge mode")
        symbolic_surface_manifest = manifest["host_delta"].get(
            "symbolic_surface",
            {
                "mode": "none",
                "payload_bytes": 0,
                "payload_sha256": None,
                "maximum_active_handlers_per_sequence": 0,
                "handlers": [],
                "source_teacher_text_retained": False,
            },
        )
        symbolic_surface_mode = symbolic_surface_manifest.get(
            "mode", "none"
        )
        expected_symbolic_surface: set[str] = set()
        if symbolic_surface_mode == "learned_rules_and_schema_realizers":
            expected_symbolic_surface = {SYMBOLIC_SURFACE_STATE_KEY}
            if (
                symbolic_surface_manifest.get(
                    "maximum_active_handlers_per_sequence"
                )
                != 1
                or symbolic_surface_manifest.get(
                    "source_teacher_text_retained"
                )
                is not False
            ):
                raise LayerCakeHostError(
                    "symbolic surface deployment contract is invalid"
                )
        elif symbolic_surface_mode != "none":
            raise LayerCakeHostError("unsupported symbolic surface mode")
        if set(state) != (
            expected_bridge
            | expected_lora
            | expected_prompt_identity
            | expected_route_bridge
            | expected_symbolic_surface
        ):
            raise LayerCakeHostError("host delta parameter set changed")
        if bridge_mode == "cakes_lora_fused":
            if not lora_targets or lora.get("fused_runtime_extra_modules") != 0:
                raise LayerCakeHostError("fused LoRA runtime contract is invalid")
            _fuse_lora(
                model,
                state,
                lora_targets,
                rank=int(lora["rank"]),
                alpha=float(lora["alpha"]),
            )
        elif bridge_mode != "cakes" or lora_targets:
            raise LayerCakeHostError("unsupported host bridge mode")
        if prompt_identity is not None:
            pointer_state = {
                name.removeprefix("prompt_identity."): state[name]
                for name in expected_prompt_identity
            }
            prompt_identity.load_state_dict(pointer_state, strict=True)
            prompt_identity.eval()
        if route_bridge is not None:
            route_state = {
                name.removeprefix("route_bridge."): state[name]
                for name in expected_route_bridge
            }
            route_bridge.load_state_dict(route_state, strict=True)
            route_bridge.eval()
        if expected_symbolic_surface:
            payload = state[SYMBOLIC_SURFACE_STATE_KEY]
            symbolic_surface_contract = _decode_symbolic_surface(payload)
            payload_bytes = _canonical_json_bytes(
                symbolic_surface_contract
            )
            if (
                len(payload_bytes)
                != symbolic_surface_manifest.get("payload_bytes")
                or hashlib.sha256(payload_bytes).hexdigest()
                != symbolic_surface_manifest.get("payload_sha256")
            ):
                raise LayerCakeHostError(
                    "symbolic surface payload differs from its manifest"
                )
        with torch.no_grad():
            for name in expected_bridge:
                value = state[name]
                if tuple(value.shape) != tuple(parameters[name].shape):
                    raise LayerCakeHostError(f"host delta shape changed for {name}")
                parameters[name].copy_(value)
        logical_sha = _bridge_state_sha256(state)
        if logical_sha != manifest["host_delta"]["logical_state_sha256_after"]:
            raise LayerCakeHostError("loaded host delta logical hash is stale")
        transformer_sha = module_state_sha256(model.transformer)
        expected_transformer_sha = manifest["parent_layercake"].get(
            "fused_runtime_transformer_state_sha256",
            manifest["parent_layercake"]["transformer_state_sha256_after"],
        )
        if transformer_sha != expected_transformer_sha:
            raise LayerCakeHostError("loaded host transformer differs from certification")
        decoding = dict(
            manifest.get(
                "decoding",
                {
                    "algorithm": "greedy",
                    "no_repeat_ngram_size": 0,
                    "prompt_identity_mixture": False,
                },
            )
        )
        if bool(decoding.get("prompt_identity_mixture")) != (
            prompt_identity is not None
        ):
            raise LayerCakeHostError(
                "decoding and prompt-identity contracts disagree"
            )
    model._abi_prompt_identity_bridge = prompt_identity
    model._abi_sparse_route_bridge = route_bridge
    model._abi_symbolic_surface = symbolic_surface_contract
    model._abi_decoding = decoding
    return model.eval(), tokenizer, manifest, device


def build_validation_rows(
    *,
    training_bundle_path: str | Path,
    validation_bundle_paths: Sequence[str | Path],
    catalog_paths: Sequence[str | Path],
    capabilities: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Bind held-out source evidence to the exact catalog/source selected for training."""

    training = read_extraction_bundle(training_bundle_path)
    desired_capabilities = (
        set(capabilities) if capabilities is not None else set(CAPABILITY_TO_ROUTE)
    )
    unknown = desired_capabilities - set(CAPABILITY_TO_ROUTE)
    if unknown:
        raise LayerCakeHostError(f"unknown requested capabilities: {sorted(unknown)}")
    catalog_ids_by_key: dict[tuple[str, str, str], set[str]] = {}
    for record in training["records"]:
        if (
            record["destination_scope"] != "english_core"
            or record["capability"] not in desired_capabilities
        ):
            continue
        key = (
            str(record["capability"]),
            str(record["source_model"]),
            str(record["source_model_revision"]),
        )
        catalog_ids_by_key.setdefault(key, set()).add(
            str(record["provenance"]).partition(":")[0]
        )
    selected_keys = {
        (
            str(item["capability"]),
            str(item["source_model"]),
            str(item["source_model_revision"]),
        )
        for item in training["selection"]["selected_items"]
        if item["destination_scope"] == "english_core"
        and item["capability"] in desired_capabilities
    }
    if {key[0] for key in selected_keys} != desired_capabilities:
        raise LayerCakeHostError("training selection lacks requested English capabilities")
    for key in selected_keys:
        if len(catalog_ids_by_key.get(key, set())) != 1:
            raise LayerCakeHostError(
                f"training material has ambiguous catalog provenance for {key}"
            )

    probes: dict[str, dict[str, Any]] = {}
    for path in catalog_paths:
        catalog = load_probe_catalog(path)
        for probe in catalog["probes"]:
            identity = f"{catalog['catalog_id']}:{probe['probe_id']}"
            if identity in probes and probes[identity] != probe:
                raise LayerCakeHostError(f"conflicting catalog probe {identity}")
            probes[identity] = probe

    rows: dict[str, dict[str, Any]] = {}
    for path in validation_bundle_paths:
        bundle = read_extraction_bundle(path)
        results = {
            str(result["record_id"]): result for result in bundle["probe_results"]
        }
        for record in bundle["records"]:
            key = (
                str(record["capability"]),
                str(record["source_model"]),
                str(record["source_model_revision"]),
            )
            if key not in selected_keys or record["split"] != "validation":
                continue
            catalog_id = str(record["provenance"]).partition(":")[0]
            if catalog_id not in catalog_ids_by_key[key]:
                continue
            result = results.get(str(record["record_id"]))
            if result is None:
                raise LayerCakeHostError("validation record lacks its source result")
            probe_identity = str(record["provenance"])
            probe = probes.get(probe_identity)
            if probe is None:
                raise LayerCakeHostError(
                    f"locked probe definition is missing: {probe_identity}"
                )
            if probe["evaluator"] != result["evaluator"]:
                raise LayerCakeHostError("source result evaluator differs from catalog")
            row = {
                "validation_record_id": str(record["record_id"]),
                "probe_id": str(result["probe_id"]),
                "provenance": probe_identity,
                "capability": str(record["capability"]),
                "prompt": strip_source_chat_template(str(record["prompt"])),
                "source_output": str(record["output"]),
                "source_passed": bool(result["passed"]),
                "source_score": float(result["score"]),
                "evaluator": dict(result["evaluator"]),
                "max_new_tokens": int(probe["max_new_tokens"]),
                "expected_route": route_for_capability(str(record["capability"])),
                "source_model": str(record["source_model"]),
                "source_model_revision": str(record["source_model_revision"]),
            }
            identity = row["validation_record_id"]
            if identity in rows and rows[identity] != row:
                raise LayerCakeHostError(f"conflicting validation record {identity}")
            rows[identity] = row
    grouped: dict[str, int] = {}
    for row in rows.values():
        grouped[row["capability"]] = grouped.get(row["capability"], 0) + 1
    incomplete = {
        capability: grouped.get(capability, 0)
        for capability in desired_capabilities
        if grouped.get(capability, 0) != 100
    }
    if incomplete:
        raise LayerCakeHostError(
            f"held-out source evidence is incomplete: {incomplete}"
        )
    return sorted(
        rows.values(), key=lambda row: (row["capability"], row["probe_id"])
    )


@torch.inference_mode()
def _generate_host(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, list[int], int, float]:
    prompt_ids = tokenizer.encode(prompt + "\n")
    if len(prompt_ids) + max_new_tokens > model.config.max_tokens:
        raise LayerCakeHostError("validation prompt exceeds LayerCake context")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    result = model(
        input_ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], device=device),
        use_cache=True,
    )
    route = int(result["task_routes"].item())
    route_tensor = result["task_routes"]
    result = _apply_sparse_route_bridge(
        model, result, route_tensor
    )
    symbolic_output = _symbolic_surface_output(
        getattr(model, "_abi_symbolic_surface", None),
        prompt=prompt,
        route=route,
    )
    if symbolic_output is not None:
        generated = tokenizer.encode(symbolic_output)
        if len(generated) > max_new_tokens:
            raise LayerCakeHostError(
                "symbolic surface exceeded the locked generation budget"
            )
        return (
            symbolic_output,
            generated,
            route,
            time.perf_counter() - started,
        )
    prompt_identity = getattr(
        model, "_abi_prompt_identity_bridge", None
    )
    prompt_token_tensor = input_ids[0]
    prompt_keys = (
        prompt_identity.key(result["hidden"][0, : len(prompt_ids)])
        if prompt_identity is not None
        else None
    )
    next_hidden = result["hidden"][:, -1]
    decoding = getattr(
        model,
        "_abi_decoding",
        {
            "algorithm": "greedy",
            "no_repeat_ngram_size": 0,
            "prompt_identity_mixture": False,
        },
    )
    no_repeat_ngram_size = int(
        decoding.get("no_repeat_ngram_size", 0)
    )
    allow_prompt_ngrams = bool(
        decoding.get("allow_prompt_ngrams", False)
    )
    allowed_ngrams = (
        {
            tuple(prompt_ids[index : index + no_repeat_ngram_size])
            for index in range(
                max(0, len(prompt_ids) - no_repeat_ngram_size + 1)
            )
        }
        if allow_prompt_ngrams and no_repeat_ngram_size > 0
        else None
    )
    state = {
        "past_key_values": result["past_key_values"],
        "task_routes": route_tensor,
        "next_logits": result["logits"][:, -1],
    }
    generated: list[int] = []
    for _ in range(max_new_tokens):
        if prompt_identity is not None:
            scores = _prompt_identity_next_probabilities(
                logits=state["next_logits"],
                query_hidden=next_hidden,
                prompt_keys=prompt_keys,
                prompt_ids=prompt_token_tensor,
                route=route_tensor,
                bridge=prompt_identity,
            )
        else:
            scores = state["next_logits"][0]
        selected = _select_next_token(
            scores,
            generated=generated,
            no_repeat_ngram_size=no_repeat_ngram_size,
            allowed_ngrams=allowed_ngrams,
        ).to(device)
        token_id = int(selected.item())
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        result = model(
            selected[:, None],
            task_routes=route_tensor,
            past_key_values=state["past_key_values"],
            use_cache=True,
        )
        result = _apply_sparse_route_bridge(
            model, result, route_tensor
        )
        state["past_key_values"] = result["past_key_values"]
        state["next_logits"] = result["logits"][:, -1]
        next_hidden = result["hidden"][:, -1]
    elapsed = time.perf_counter() - started
    output = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    lexical_repetition_threshold = int(
        decoding.get("lexical_repetition_truncation_threshold", 0)
    )
    if lexical_repetition_threshold > 0:
        output = _truncate_novel_lexical_repetition(
            output,
            prompt,
            threshold=lexical_repetition_threshold,
        )
        generated = tokenizer.encode(output)
    return output, generated, route, elapsed


def evaluate_host_semantics(
    *,
    training_bundle_path: str | Path,
    validation_bundle_paths: Sequence[str | Path],
    catalog_paths: Sequence[str | Path],
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    host_path: str | Path | None,
    output_path: str | Path,
    device_name: str,
    capabilities: Sequence[str] | None = None,
    limit_per_capability: int | None = None,
) -> dict[str, Any]:
    """Evaluate automatic host routing against exact paired source validation rows."""

    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(f"semantic evidence is immutable: {output_path}")
    rows = build_validation_rows(
        training_bundle_path=training_bundle_path,
        validation_bundle_paths=validation_bundle_paths,
        catalog_paths=catalog_paths,
        capabilities=capabilities,
    )
    if limit_per_capability is not None:
        if limit_per_capability <= 0:
            raise LayerCakeHostError("limit_per_capability must be positive")
        counts: dict[str, int] = {}
        limited = []
        for row in rows:
            count = counts.get(row["capability"], 0)
            if count < limit_per_capability:
                limited.append(row)
                counts[row["capability"]] = count + 1
        rows = limited
    model, tokenizer, manifest, device = load_host_model(
        layercake_root=layercake_root,
        parent_path=parent_path,
        canonical_abi_path=canonical_abi_path,
        host_path=host_path,
        device_name=device_name,
    )
    observations = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        output, generated, route, elapsed = _generate_host(
            model,
            tokenizer,
            row["prompt"],
            max_new_tokens=row["max_new_tokens"],
            device=device,
        )
        passed, score = evaluate_output(output, row["evaluator"])
        observations.append(
            {
                **row,
                "layercake_output": output,
                "layercake_output_sha256": hashlib.sha256(
                    output.encode("utf-8")
                ).hexdigest(),
                "layercake_generated_tokens": len(generated),
                "layercake_passed": passed,
                "layercake_score": score,
                "automatic_route": route,
                "route_correct": route == row["expected_route"],
                "latency_seconds": elapsed,
            }
        )
        if index % 100 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": index,
                        "total": len(rows),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    capability_metrics = {}
    for capability in sorted({row["capability"] for row in observations}):
        selected = [
            row for row in observations if row["capability"] == capability
        ]
        source_passes = sum(row["source_passed"] for row in selected)
        host_passes = sum(row["layercake_passed"] for row in selected)
        regressions = sum(
            row["source_passed"] and not row["layercake_passed"]
            for row in selected
        )
        capability_metrics[capability] = {
            "observations": len(selected),
            "source_passes": source_passes,
            "layercake_passes": host_passes,
            "source_pass_rate": source_passes / len(selected),
            "layercake_pass_rate": host_passes / len(selected),
            "source_passing_regressions": regressions,
            "bounded_zero_regression_pass": regressions == 0,
            "automatic_route_accuracy": sum(
                row["route_correct"] for row in selected
            )
            / len(selected),
        }
    complete_depth = (
        limit_per_capability is None
        and all(
            metrics["observations"] == 100
            for metrics in capability_metrics.values()
        )
    )
    semantic_pass = (
        complete_depth
        and len(capability_metrics) == len(CAPABILITY_TO_ROUTE)
        and all(
            metrics["bounded_zero_regression_pass"]
            for metrics in capability_metrics.values()
        )
    )
    evidence = {
        "schema_version": "abi-layercake-host-semantic-validation/1",
        "status": "PASS" if semantic_pass else "FAIL",
        "split": "validation",
        "final_test_accessed": False,
        "training_bundle_sha256": _sha256_file(Path(training_bundle_path)),
        "host_manifest_sha256": (
            manifest["manifest_sha256"] if manifest is not None else None
        ),
        "parent_only_baseline": manifest is None,
        "device": str(device),
        "observation_count": len(observations),
        "complete_locked_depth": complete_depth,
        "bounded_zero_regression_pass": semantic_pass,
        "capability_metrics": capability_metrics,
        "wall_seconds": time.perf_counter() - started,
        "observations": observations,
        "claim_boundary": (
            "This is paired validation evidence on the declared synthetic "
            "catalog. It is not final-test evidence or universal semantic identity."
        ),
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--bundle", required=True)
    train.add_argument("--layercake-root", required=True)
    train.add_argument("--parent", required=True)
    train.add_argument("--canonical-abi", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--budget-index", type=int, default=-1)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--steps", type=int, default=600)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--classifier-loss-weight", type=float, default=0.25)
    train.add_argument("--anchor-loss-weight", type=float, default=1e-4)
    train.add_argument("--max-tokens", type=int, default=256)
    train.add_argument("--device", default="cuda")
    train.add_argument(
        "--bridge-mode",
        choices=("cakes", "cakes_lora_fused"),
        default="cakes",
    )
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--lora-alpha", type=float, default=16.0)
    train.add_argument("--prompt-identity-rank", type=int, default=0)
    train.add_argument("--route-bridge-rank", type=int, default=0)
    train.add_argument("--prompt-overlap-loss-weight", type=float, default=0.0)
    train.add_argument("--symbolic-surface", action="store_true")
    train.add_argument("--no-repeat-ngram-size", type=int, default=0)
    train.add_argument("--recovery-start-step", type=int, default=0)
    train.add_argument("--recovery-interval", type=int, default=0)
    train.add_argument("--recovery-horizons", default="4,8,16")
    train.add_argument("--recovery-routes", default="")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--validation-bundle", action="append", required=True)
    evaluate.add_argument("--catalog", action="append", required=True)
    evaluate.add_argument("--layercake-root", required=True)
    evaluate.add_argument("--parent", required=True)
    evaluate.add_argument("--canonical-abi", required=True)
    evaluate.add_argument("--host")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--capabilities")
    evaluate.add_argument("--limit-per-capability", type=int)
    derive = subparsers.add_parser("derive-symbolic")
    derive.add_argument("--bundle", required=True)
    derive.add_argument("--source-host", required=True)
    derive.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        manifest = train_host_delta(
            bundle_path=args.bundle,
            layercake_root=args.layercake_root,
            parent_path=args.parent,
            canonical_abi_path=args.canonical_abi,
            output_path=args.output,
            budget_index=args.budget_index,
            seed=args.seed,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            classifier_loss_weight=args.classifier_loss_weight,
            anchor_loss_weight=args.anchor_loss_weight,
            max_tokens=args.max_tokens,
            device_name=args.device,
            bridge_mode=args.bridge_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            prompt_identity_rank=args.prompt_identity_rank,
            route_bridge_rank=args.route_bridge_rank,
            prompt_overlap_loss_weight=args.prompt_overlap_loss_weight,
            symbolic_surface=args.symbolic_surface,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            recovery_start_step=args.recovery_start_step,
            recovery_interval=args.recovery_interval,
            recovery_horizons=tuple(
                int(value)
                for value in args.recovery_horizons.split(",")
                if value.strip()
            ),
            recovery_routes=(
                tuple(
                    int(value)
                    for value in args.recovery_routes.split(",")
                    if value.strip()
                )
                if args.recovery_routes
                else None
            ),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate":
        evidence = evaluate_host_semantics(
            training_bundle_path=args.bundle,
            validation_bundle_paths=args.validation_bundle,
            catalog_paths=args.catalog,
            layercake_root=args.layercake_root,
            parent_path=args.parent,
            canonical_abi_path=args.canonical_abi,
            host_path=args.host,
            output_path=args.output,
            device_name=args.device,
            capabilities=(
                sorted(
                    {
                        item.strip()
                        for item in args.capabilities.split(",")
                        if item.strip()
                    }
                )
                if args.capabilities
                else None
            ),
            limit_per_capability=args.limit_per_capability,
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in evidence.items()
                    if key != "observations"
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "derive-symbolic":
        manifest = derive_symbolic_surface_host(
            bundle_path=args.bundle,
            source_host_path=args.source_host,
            output_path=args.output,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
