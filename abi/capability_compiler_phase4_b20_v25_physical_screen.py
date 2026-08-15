"""Train and physically screen the sole oracle-inconclusive B20 seed on LayerCake v25."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import psutil
from safetensors.torch import load_file, save_file
import torch

from . import capability_compiler_phase4_abi_lineage as lineage
from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    set_determinism,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _write_immutable
from .capability_compiler_phase3_guarded_screen import artifact_markers
from .capability_compiler_phase3_targeted_recovery_bridge import _batch_with_prefixes
from .capability_compiler_phase3_weak_residual import _state_hash
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_b20_host_compatibility_audit import host_can_change
from .capability_compiler_phase4_clarification_route import (
    ACTIVE_PARAMETERS,
    CLARIFICATION_ROUTE,
    INSTALLED_PARAMETERS,
    LEGACY_ROUTES,
    NEW_TRAINABLE_PARAMETERS,
    ClarificationRouteResidual,
    _attach,
    _schedule,
    _set_routes,
)
from .capability_compiler_phase4_v19_frontier_rescreen import (
    _merged_evaluation,
    _quality_gates,
    _rows,
)
from .layercake_host import _equal_record_prompt_overlap_ce


FORMAT = "abi-capability-compiler-phase4-b20-v25-physical-screen/1"
SEED = 155921
BUDGET = "B20"
RESIDUAL_CAPABILITIES = (
    "abstention",
    "coherence",
    "fluent_realization",
    "tone_control",
    "clarification",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_B20_SEED155921_V25_PHYSICAL_SCREEN"
        or protocol.get("run") != {"budget": BUDGET, "seed": SEED}
        or protocol.get("training_device") != "cuda"
        or protocol.get("evaluation_device") != "cuda"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
    ):
        raise Phase3Error("B20 v25 physical-screen governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B20 v25 physical-screen binding changed: {relative}")
    lineage_protocol = _json(root / protocol["lineage_protocol"])
    return protocol, sha256_file(path), lineage_protocol


def _selected_examples(
    root: Path,
    protocol: Mapping[str, Any],
    lineage_protocol: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    manifest = _json(root / lineage_protocol["budget_manifest"])
    selected, accounting = lineage._selected_rows(root, lineage_protocol, manifest, BUDGET)
    rows = [row for row in selected["phase1_ir"] if row["capability"] == "clarification"]
    examples = lineage._examples_subset(
        rows,
        tokenizer,
        system="A0",
        seed=SEED,
        max_tokens=int(protocol["training"]["max_tokens"]),
    )
    return examples, accounting, rows


def _route_for_capability(capability: str) -> int:
    return RESIDUAL_CAPABILITIES.index(capability) if capability in RESIDUAL_CAPABILITIES else -1


def preservation_gates(
    historical: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, bool]:
    changed = [row for row in rows if str(row["output"]) != str(historical[str(row["probe_id"])]["output"])]
    immutable = [row for row in rows if not host_can_change(historical[str(row["probe_id"])])]
    return {
        "changes_bounded_to_declared_host_scope": all(
            host_can_change(historical[str(row["probe_id"])]) for row in changed
        ),
        "all_immutable_outputs_exact": all(
            str(row["output"]) == str(historical[str(row["probe_id"])]["output"])
            for row in immutable
        ),
        "historical_passing_changed_rows_remain_passing": all(
            not bool(historical[str(row["probe_id"])]["functional_pass_v1"])
            or bool(row["functional_pass_v1"])
            for row in changed
        ),
        "all_rows_present": len(rows) == len(historical) == 1400,
    }


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("registered CUDA device unavailable")
    run_dir = root / protocol["lineage_dir"]
    inherited_path = run_dir / "v526" / "control_bridge.safetensors"
    inherited = load_file(str(inherited_path), device="cpu")
    residual = ClarificationRouteResidual(inherited, SEED)
    hidden = torch.randn(1, 3, 768)
    with torch.no_grad():
        initial = residual.delta(hidden, torch.tensor([CLARIFICATION_ROUTE]))
    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    _, tokenizer, _ = lineage._load_candidate(root, v440, run_dir / "v463", torch.device("cpu"))
    examples, accounting, rows = _selected_examples(root, protocol, lineage_protocol, tokenizer)
    schedule = _schedule(examples, SEED, int(protocol["training"]["epochs"]))
    counts = Counter(str(row["record_id"]) for row in schedule)
    acquisition = {
        hashlib.sha256(str(row["normalized_generation_prompt"]).encode()).hexdigest()
        for row in rows
    }
    development = {
        hashlib.sha256(str(row["prompt"]).encode()).hexdigest()
        for row in development_probes(root / protocol["development_catalog"])
        if row["canonical_capability"] == "clarification"
    }
    gates = {
        "exact_100_b20_clarification_records": len(rows) == len(examples) == 100,
        "exact_ten_exposures_per_record": set(counts.values()) == {10},
        "exact_1000_training_observations": len(schedule) == 1000,
        "development_prompt_hash_disjoint": acquisition.isdisjoint(development),
        "new_route_initial_delta_zero": torch.equal(initial, torch.zeros_like(initial)),
        "only_24576_parameters_trainable": sum(value.numel() for value in residual.parameters())
        == NEW_TRAINABLE_PARAMETERS,
        "b20_information_exact": int(accounting["unique_source_attempts"]) == 2028
        and int(accounting["authoritative_teacher_output_tokens"]) == 62417
        and int(accounting["record_memberships"]) == 2056,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "format": "abi-capability-compiler-phase4-b20-v25-physical-screen-preflight/1",
        "status": "PASS_B20_V25_PHYSICAL_SCREEN_PREFLIGHT" if all(gates.values()) else "FAIL_B20_V25_PHYSICAL_SCREEN_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "gates": gates,
        "selected_clarification_records": len(rows),
        "training_observations": len(schedule),
        "information": accounting,
        "inherited_checkpoint_sha256": sha256_file(inherited_path),
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable B20 fifth-route output exists or CUDA unavailable")
    set_determinism(SEED)
    device = torch.device("cuda")
    run_dir = root / protocol["lineage_dir"]
    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    model, tokenizer, _ = lineage._load_candidate(root, v440, run_dir / "v463", device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parent_before = _state_hash(model.state_dict())
    inherited_path = run_dir / "v526" / "control_bridge.safetensors"
    inherited = load_file(str(inherited_path), device="cpu")
    residual = ClarificationRouteResidual(inherited, SEED).to(device)
    handles = _attach(model, residual)
    examples, accounting, clarification_rows = _selected_examples(
        root, protocol, lineage_protocol, tokenizer
    )
    cfg = protocol["training"]
    schedule = _schedule(examples, SEED, int(cfg["epochs"]))
    if len(schedule) != int(cfg["steps"]):
        raise Phase3Error("B20 fifth-route schedule changed")
    optimizer = torch.optim.AdamW(
        residual.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(cfg["weight_decay"]),
    )
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    curves = []
    sequence = hashlib.sha256()
    response_tokens = 0
    started = time.perf_counter()
    model.eval()
    residual.train()
    for step, row in enumerate(schedule, 1):
        ids, labels, attention, prompt_lengths, task_routes = _batch_with_prefixes(
            [row], int(tokenizer.eos_token_id), device
        )
        _set_routes(model, torch.tensor([CLARIFICATION_ROUTE], dtype=torch.long, device=device))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                task_routes=task_routes,
                use_cache=False,
            )
            loss = _equal_record_prompt_overlap_ce(
                result["logits"],
                labels,
                ids,
                prompt_lengths,
                overlap_weight=float(cfg["prompt_overlap_weight"]),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), float(cfg["gradient_clip_norm"]))
        optimizer.step()
        response_tokens += int(row["response_tokens"])
        sequence.update((str(row["record_id"]) + "\n").encode())
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(cfg["curve_interval"]) == 0:
            curve = {"step": step, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - started}
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    for handle in handles:
        handle.remove()
    if _state_hash(model.state_dict()) != parent_before:
        raise Phase3Error("frozen B20 parent changed")
    state = residual.package_state()
    if not torch.equal(state["down"][:LEGACY_ROUTES], inherited["down"]) or not torch.equal(
        state["up"][:LEGACY_ROUTES], inherited["up"]
    ):
        raise Phase3Error("inherited B20 routes changed")
    output.mkdir(parents=True)
    checkpoint = output / "clarification_route.safetensors"
    save_file(state, str(checkpoint), metadata={"format": FORMAT, "budget": BUDGET, "seed": str(SEED)})
    metadata = {
        "format": FORMAT,
        "status": "TRAINED_B20_SEED155921_FIFTH_CLARIFICATION_ROUTE",
        "protocol_sha256": protocol_sha,
        "budget": BUDGET,
        "seed": SEED,
        "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
        "parent": {"checkpoint_sha256": sha256_file(run_dir / "v463" / "model.safetensors"), "mutated": False},
        "router": {"checkpoint_sha256": sha256_file(run_dir / "router" / "router.safetensors"), "mutated": False},
        "inherited_residual": {"checkpoint_sha256": sha256_file(inherited_path), "routes_mutated": 0},
        "architecture": {
            "routes": 5,
            "legacy_routes": 4,
            "new_trainable_parameters": NEW_TRAINABLE_PARAMETERS,
            "installed_parameters": INSTALLED_PARAMETERS,
            "active_parameters_on_clarification": ACTIVE_PARAMETERS,
            "active_routes_per_token": 1,
            "rank": 16,
        },
        "training": {
            "steps": len(schedule),
            "epochs": int(cfg["epochs"]),
            "selected_records": len(clarification_rows),
            "teacher_response_tokens_in_loss": response_tokens,
            "record_sequence_sha256": sequence.hexdigest(),
            "wall_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": int(peak_rss),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "curves": curves,
        },
        "imported_information": {
            "unique_source_attempts": int(accounting["unique_source_attempts"]),
            "authoritative_teacher_output_tokens": int(accounting["authoritative_teacher_output_tokens"]),
            "record_memberships": int(accounting["record_memberships"]),
            "new_teacher_outputs": 0,
            "stored_logits": 0,
            "stored_hidden_activations": 0,
            "source_parameters_copied": 0,
        },
        "teacher_present": False,
        "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


def _api(layercake_root: Path) -> dict[str, Any]:
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.cake.manifest import CakeManifest
    from layercake.cake.package import build_package, load_package, tensor_specs
    from layercake.cake.signing import key_id
    from layercake_extensions.route_isolated_clarification_core_v25 import (
        ALLOCATION_BOUNDED_ADOPTION_FEATURE,
        ARCHITECTURE_V25_FORMAT,
        CLARIFICATION_ROUTE_ISOLATION_FEATURE,
        RESIDUAL_CAPABILITIES_V25,
        ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256,
        ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION,
        ClarificationRouteAllocationBoundedCoreHost,
    )
    from layercake_extensions.route_isolated_format_literal_core_v22 import (
        FORMAT_LITERAL_DECLARATION,
        FORMAT_LITERAL_FEATURE,
        FORMAT_LITERAL_MODE,
        extract_exact_format_literals,
        render_exact_format_literals,
    )
    from layercake_extensions.route_isolated_lexical_guard_core_v21 import (
        EXACT_LEXICAL_BOUNDARY,
        EXACT_LEXICAL_GUARD_FEATURE,
    )
    from layercake_extensions.route_isolated_prompt_span_core_v19 import PROMPT_SPAN_FEATURE
    from layercake_extensions.route_isolated_runtime_residency_core_v23 import SINGLE_PARSE_ACTIVATION_FEATURE
    from layercake_extensions.route_isolated_shallow_sparse_core import CAPABILITY_TO_TASK_ROUTE
    from layercake_extensions.route_isolated_universal_guard_core_v20 import GUARD_PREDICATE, UNIVERSAL_GUARD_FEATURE
    return locals()


def _architecture(root: Path, protocol: Mapping[str, Any], api: Mapping[str, Any]) -> dict[str, Any]:
    parent = _json(root / protocol["model_metadata"])
    tokenizer = _json(root / protocol["model_tokenizer"])
    tokenizer_raw = json.dumps(tokenizer, sort_keys=True, separators=(",", ":")).encode()
    router_tokenizer = _json(root / protocol["router_tokenizer"])
    router = _json(root / protocol["router_config"])
    markers = artifact_markers(root / protocol["guard_artifact"])
    return {
        "format": api["ARCHITECTURE_V25_FORMAT"],
        "model": parent["architecture"],
        "model_tokenizer": {
            "format": "declarative-tokenizers-json/1",
            "tokenizers_json": tokenizer,
            "sha256": hashlib.sha256(tokenizer_raw).hexdigest(),
            "eos_token_id": 50256,
        },
        "router": {
            "vocabulary": int(router["vocabulary"]),
            "character_hash_buckets": int(router["character_hash_buckets"]),
            "character_ngram_minimum": int(router["character_ngram_minimum"]),
            "character_ngram_maximum": int(router["character_ngram_maximum"]),
            "hash_seed": int(router["hash_seed"]),
            "classes": 15,
        },
        "router_tokenizer": router_tokenizer,
        "residual": {"width": 768, "rank": 16, "routes": 5, "reuse": "before_each_transformer_block"},
        "capabilities": list(protocol["capabilities"]),
        "capability_to_task_route": api["CAPABILITY_TO_TASK_ROUTE"],
        "weak_capabilities": list(api["RESIDUAL_CAPABILITIES_V25"]),
        "guard": {
            "predicate": api["GUARD_PREDICATE"],
            "scope": "all_capabilities",
            "boundary": api["EXACT_LEXICAL_BOUNDARY"],
            "stop_before_collapsing_token": True,
            "abstention_markers": list(markers),
            "abstention_clause": "I cannot determine that from the information given.",
        },
        "format_literal": api["FORMAT_LITERAL_DECLARATION"],
    }


def _package(
    root: Path,
    protocol: Mapping[str, Any],
    candidate: Path,
    path: Path,
    api: Mapping[str, Any],
    private: Ed25519PrivateKey,
    public_pem: bytes,
) -> dict[str, Any]:
    states = {
        "model": load_file(str(root / protocol["components"]["model"]), device="cpu"),
        "router": load_file(str(root / protocol["components"]["router"]), device="cpu"),
        "residual": load_file(str(candidate / "clarification_route.safetensors"), device="cpu"),
    }
    counts = {name: sum(value.numel() for value in state.values()) for name, state in states.items()}
    expected = {"model": 61655050, "router": 1058040, "residual": 124416}
    if counts != expected:
        raise Phase3Error(f"B20 v25 component inventory changed: {counts}")
    tensors = {
        f"{namespace}.{name}": value
        for namespace, state in states.items()
        for name, value in state.items()
    }
    if Counter(name.split(".", 1)[0] for name in tensors) != Counter({"model": 82, "router": 3, "residual": 4}):
        raise Phase3Error("B20 v25 tensor namespace inventory changed")
    signer = api["key_id"](public_pem)
    features = [
        "byte_input",
        "safe_tensors",
        "persistent_incremental_state",
        "physical_route_isolation",
        "declarative_runtime_guard",
        "strict_utf8_boundary",
        api["PROMPT_SPAN_FEATURE"],
        api["UNIVERSAL_GUARD_FEATURE"],
        api["EXACT_LEXICAL_GUARD_FEATURE"],
        api["FORMAT_LITERAL_FEATURE"],
        api["SINGLE_PARSE_ACTIVATION_FEATURE"],
        api["ALLOCATION_BOUNDED_ADOPTION_FEATURE"],
        api["CLARIFICATION_ROUTE_ISOLATION_FEATURE"],
    ]
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id="abi-phase4-b20-seed155921-v25-english-core",
        name="ABI Phase 4 B20 seed 155921 v25 English core",
        description="Frozen B20 lineage with one clarification-only fifth route",
        version="0.25.0-b20-screen",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=api["ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION"],
        abi_hash=api["ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256"],
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=_architecture(root, protocol, api),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": features},
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](tensors),
        package_hash="",
        training_data_provenance={
            "phase4_budget": BUDGET,
            "phase4_seed": SEED,
            "lineage_result_sha256": protocol["bindings"][protocol["lineage_result"]],
            "fifth_route_checkpoint_sha256": sha256_file(candidate / "clarification_route.safetensors"),
            "teacher_at_inference": False,
            "source_transformer_blocks": 0,
            "receiver_training_steps": 0,
        },
        evaluation_evidence={"authorization": protocol["authorization"], "status": "B20_V25_DEVELOPMENT_PHYSICAL_SCREEN"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=("english-core",),
        permissions=("local-inference",),
    )
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    api["build_package"](path, manifest, tensors, private_key=private_pem)
    loaded = api["load_package"](path, trust_store={signer: public_pem}, require_signature=True)
    gates = {
        "signature_valid": loaded.signed,
        "tensor_values_exact": set(loaded.tensors) == set(tensors)
        and all(torch.equal(loaded.tensors[name], tensors[name]) for name in tensors),
        "interface_v25": loaded.manifest.abi_version == api["ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION"]
        and loaded.manifest.abi_hash == api["ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256"],
        "component_counts_exact": counts == expected,
        "receiver_learning_zero": True,
        "teacher_absent": True,
    }
    if not all(gates.values()):
        raise Phase3Error(f"B20 v25 package verification failed: {gates}")
    return {
        "archive_sha256": loaded.archive_hash,
        "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        "package_hash": loaded.manifest.package_hash,
        "archive_bytes": path.stat().st_size,
        "component_parameters": counts,
        "total_parameters": sum(counts.values()),
        "tensor_count": len(tensors),
        "signer": signer,
        "gates": gates,
    }


@torch.inference_mode()
def _generate(host: Any, prompt: str, maximum: int, capability: str):
    if capability in {"coherence", "format_control"}:
        value = host.generate(prompt, maximum_tokens=maximum).decode("utf-8")
        pointer = dict(host.last_pointer_execution or {})
        pointer.pop("wall_seconds", None)
        format_record = dict(host.last_format_execution or {})
        format_record.pop("wall_seconds", None)
        return value, False, _route_for_capability(capability), pointer, format_record
    state = host.prefill(prompt)
    for _ in range(maximum):
        if host.decode_step(state) is None:
            break
    return (
        host.realize(state).decode("utf-8"),
        bool(state["terminated_by_guard"]),
        int(state["weak_route"]),
        {},
        {},
    )


@torch.inference_mode()
def evaluate(root: Path, protocol_path: Path, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, _ = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable B20 v25 evaluation exists or CUDA unavailable")
    metadata = _json(candidate / "metadata.json")
    checkpoint = candidate / "clarification_route.safetensors"
    if metadata["protocol_sha256"] != protocol_sha or metadata["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise Phase3Error("B20 fifth-route candidate lineage changed")
    api = _api((root / protocol["layercake_root"]).resolve())
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(protocol["research_signing_seed_hex"]))
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    probes_list = development_probes(root / protocol["development_catalog"])
    probes = {str(row["probe_id"]): row for row in probes_list}
    teacher = {str(row["probe_id"]): row for row in _rows(root / protocol["teacher_reference"])}
    historical_rows = _rows(root / protocol["historical_outputs"])
    historical = {str(row["probe_id"]): row for row in historical_rows}
    if len(historical) != 1400:
        raise Phase3Error("B20 historical matrix changed")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="abi-b20-v25-") as raw:
        temporary = Path(raw)
        package = _package(root, protocol, candidate, temporary / "candidate.cake", api, private, public_pem)
        host = api["ClarificationRouteAllocationBoundedCoreHost"](
            temporary / "registry", trust_store={package["signer"]: public_pem}, device="cuda"
        )
        active = host.activate(temporary / "candidate.cake")
        rows = []
        started = time.perf_counter()
        for index, probe in enumerate(probes_list, 1):
            probe_id = str(probe["probe_id"])
            capability = str(probe["canonical_capability"])
            prior = historical[probe_id]
            value, terminated, physical_route, pointer, format_record = _generate(
                host, str(probe["prompt"]), int(probe["max_new_tokens"]), capability
            )
            routed = host.route(str(probe["prompt"]))
            row = {
                **prior,
                "output": value,
                "original_output": value,
                "output_token_ids": [int(token) for token in host.model_tokenizer.encode(value)],
                "automatic_capability_route": routed,
                "capability_route_correct": routed == capability,
                "physical_residual_route": physical_route,
                "active_residual_routes": 0 if physical_route < 0 else 1,
                "fifth_clarification_route_active": physical_route == CLARIFICATION_ROUTE,
                "strong_parent_output_exact": value == str(prior["output"]),
                "strong_parent_prefix_preserved": str(prior["output"]).startswith(value),
                "historical_functional_pass_v1": bool(prior["functional_pass_v1"]),
                "historical_repetition_collapse_v2": bool(prior["repetition_collapse_v2"]),
                "guard_terminated": terminated,
                "canonical_historical_prefix_preserved": str(prior["output"]).startswith(value),
                "abstention_clause_prefixed": capability == "abstention"
                and value.startswith("I cannot determine that from the information given."),
                "functional_pass_v1": evaluate_functional(value, probe["evaluator"]),
                "functional_pass_v2": evaluate_functional_v2(value, probe["evaluator"], capability),
                "repetition_collapse_v2": repetition_collapse_v2(value),
                "v25_pointer": pointer,
                "v25_format": format_record,
                "output_changed_from_b20_history": value != str(prior["output"]),
            }
            rows.append(row)
            if index % 200 == 0:
                print(json.dumps({"evaluated": index}), flush=True)
        verified = host.verify()
        residual_state = host.residual.state_dict()
        inherited = load_file(str(root / protocol["components"]["inherited_residual"]), device="cpu")
        inherited_exact = (
            torch.equal(residual_state["norm.weight"].cpu(), inherited["norm.weight"])
            and torch.equal(residual_state["norm.bias"].cpu(), inherited["norm.bias"])
            and torch.equal(residual_state["down"][:4].cpu(), inherited["down"])
            and torch.equal(residual_state["up"][:4].cpu(), inherited["up"])
        )
        del host
        gc.collect()
        torch.cuda.empty_cache()
    raw_path = output / "development_outputs.jsonl"
    _write_immutable(raw_path, b"".join(canonical_json_bytes(row) for row in rows))
    evaluation = _merged_evaluation(rows)
    quality, relative = _quality_gates(protocol, evaluation, rows, probes, teacher, SEED + 9_500_000)
    quality.pop("strong_parent_exact")
    quality.pop("training_absent")
    preserve = preservation_gates(historical, rows)
    route_gates = {
        "all_physical_routes_exact": all(int(row["physical_residual_route"]) == _route_for_capability(str(row["capability"])) for row in rows),
        "clarification_route_four_on_all_100": sum(row["capability"] == "clarification" and row["physical_residual_route"] == 4 for row in rows) == 100,
        "one_active_route_maximum": all(int(row["active_residual_routes"]) in {0, 1} for row in rows),
        "inherited_four_routes_exact": inherited_exact,
        "router_exact": evaluation["router_correct"] == 1400,
    }
    product_gates = {
        "signed_package_identity": active["archive_hash"] == package["archive_sha256"]
        and active["payload_hash"] == package["tensor_payload_hash"],
        "package_verified": verified["status"] == "PASS",
        "one_authenticated_parse": active["authenticated_package_parses"] == 1,
        "strict_storage_adoption": active["strict_assigned_tensor_count"] == active["authenticated_tensor_count"] == 89
        and active["meta_tensors_after_adoption"] == 0,
        "receiver_learning_zero": active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0,
        "interface_v25": package["gates"]["interface_v25"],
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    machine = all(quality.values()) and all(preserve.values()) and all(route_gates.values()) and all(product_gates.values())
    result = {
        "format": "abi-capability-compiler-phase4-b20-v25-physical-screen-result/1",
        "status": "PASS_B20_SEED155921_V25_PHYSICAL_PRODUCT_SCREEN" if machine else "FAIL_B20_SEED155921_V25_PHYSICAL_PRODUCT_SCREEN",
        "protocol_sha256": protocol_sha,
        "budget": BUDGET,
        "seed": SEED,
        "checkpoint_sha256": sha256_file(checkpoint),
        "functional_passes_v1": evaluation["functional_passes_v1"],
        "observations": evaluation["observations"],
        "per_capability": evaluation["per_capability"],
        "repetition_collapses_v2": evaluation["repetition_collapses_v2"],
        "guard_terminations": evaluation["guard_terminations"],
        "router_correct": evaluation["router_correct"],
        "teacher_comparison_v1": relative,
        "quality_gates": quality,
        "preservation_gates": preserve,
        "route_gates": route_gates,
        "product_gates": product_gates,
        "changed_rows": sum(bool(row["output_changed_from_b20_history"]) for row in rows),
        "changed_by_capability": {
            capability: sum(row["capability"] == capability and row["output_changed_from_b20_history"] for row in rows)
            for capability in CAPABILITIES
        },
        "package": {key: value for key, value in package.items() if key != "signer"},
        "activation": active,
        "evaluation_wall_seconds": time.perf_counter() - started,
        "raw_outputs_sha256": sha256_file(raw_path),
        "hardware": {
            "machine": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "teacher_present_at_inference": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "stable_minimum_established": False,
        "claim_boundary": "One exact signed B20 seed155921 V25 development product screen. The other B20 seeds, all-seed minimum, B40 product runtime, matched baselines, final test, Phase 4, and ABI superiority remain separate gates.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train")
    parser.add_argument("--evaluate")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = root / args.protocol
    if args.preflight:
        result = preflight(root, protocol_path)
    elif args.train:
        result = train(root, protocol_path, root / args.train)
    elif args.evaluate and args.output:
        result = evaluate(root, protocol_path, root / args.evaluate, root / args.output)
    else:
        raise Phase3Error("select preflight, train, or evaluate")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith(("PASS", "TRAINED")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
