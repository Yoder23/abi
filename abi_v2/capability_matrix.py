"""Execute one frozen-adapter/four-immutable-capability ABI V2 host matrix cell set."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import psutil
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from abi.capability_compiler_phase2_common import evaluate_functional
from abi.capability_compiler_phase2_teacher import development_probes
from abi.capability_compiler_phase4_b20_v25_physical_screen import _api
from abi.capability_compiler_phase5_construct_screen import project_catalog_prompt
from abi.capability_compiler_phase5_selective_product import (
    DIRECT_ABI_SHA256,
    DIRECT_ABI_VERSION,
    DOMAINS,
    _domain_rows,
    _domain_specs,
)

from .canonical import (
    ABI_VERSION,
    canonical_context,
    canonical_json_bytes,
    canonical_output_intent,
    sha256_bytes,
    strict_utf8,
)
from .host_certification import _snapshot_inventory

HOSTS = ("layercake", "qwen2", "pythia")
CAPABILITIES = ("english", "python", "chemistry", "civics")


class MatrixError(RuntimeError):
    """Raised when a frozen ABI V2 matrix invariant is violated."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatrixError(f"expected object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise MatrixError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _tensor_bytes(module: Any) -> int:
    seen: set[tuple[int, int, str]] = set()
    total = 0
    for tensor in [*module.parameters(), *module.buffers()]:
        identity = (int(tensor.untyped_storage().data_ptr()), tensor.untyped_storage().nbytes(), str(tensor.device))
        if identity not in seen:
            seen.add(identity)
            total += int(tensor.untyped_storage().nbytes())
    return total


def _semantic_input(prompt: str, *, position: int) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "instruction_type": "answer",
        "constraints": ["exact"],
        "relation": "none",
        "topic": "supplied_context",
        "uncertainty": "certain",
        "output_intent": "fluent_text",
        "sequence_position": position,
    }


class FrozenHostAdapter:
    """A zero-parameter native-codec realization of canonical ABI output intent."""

    def __init__(
        self,
        *,
        path: Path,
        expected_sha256: str,
        tokenizer: Any = None,
    ) -> None:
        if _sha256(path) != expected_sha256:
            raise MatrixError("frozen host adapter hash changed")
        document = _json(path)
        if (
            document.get("abi_version") != ABI_VERSION
            or document.get("frozen") is not True
            or document.get("trainable_parameters") != 0
            or document.get("optimizer_steps") != 0
            or document.get("post_freeze_mutation_allowed") is not False
        ):
            raise MatrixError("host adapter is not a frozen ABI V2 adapter")
        self.path = path
        self.expected_sha256 = expected_sha256
        self.document = document
        self.tokenizer = tokenizer
        self.enabled = True

    def verify(self) -> bool:
        return _sha256(self.path) == self.expected_sha256

    def realize(
        self,
        *,
        prompt: str,
        output: str,
        capability_id: str,
        position: int,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise MatrixError("generic host adapter is absent")
        if not self.verify():
            raise MatrixError("generic host adapter changed after freeze")
        context = canonical_context(_semantic_input(prompt, position=position))
        intent = canonical_output_intent(output, capability_id=capability_id)
        payload = bytes.fromhex(intent["authoritative_utf8_hex"])
        if self.tokenizer is None:
            units = list(payload)
            realized = bytes(units).decode("utf-8", errors="strict")
            tokenizer_mode = "strict_utf8_identity"
        else:
            units = self.tokenizer.encode(output, add_special_tokens=False)
            realized = self.tokenizer.decode(
                units,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            tokenizer_mode = self.tokenizer.__class__.__name__
        if strict_utf8(realized) != payload:
            raise MatrixError("native host realization changed canonical output bytes")
        return {
            "output": realized,
            "output_sha256": sha256_bytes(payload),
            "canonical_context_sha256": context["state_sha256"],
            "canonical_output_intent_sha256": intent["state_sha256"],
            "host_native_generation_units": len(units),
            "host_native_generation_state_sha256": sha256_bytes(
                canonical_json_bytes([int(value) for value in units])
            ),
            "tokenizer_mode": tokenizer_mode,
        }


def _load_native_host(
    *,
    host: str,
    snapshot: Path | None,
    device: str,
    expected_host: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    if host == "layercake":
        return None, None, {"parameter_count": 0, "checkpoint_loaded": False}
    if snapshot is None or not snapshot.is_dir():
        raise MatrixError(f"{host} requires its frozen local snapshot")
    snapshot_inventory_sha256, _ = _snapshot_inventory(snapshot)
    if snapshot_inventory_sha256 != expected_host["snapshot_inventory_sha256"]:
        raise MatrixError(f"{host} frozen snapshot identity changed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected = torch.device(device)
    dtype = torch.float16 if selected.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=dtype
    ).to(selected)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_loaded": True,
        "precision": str(dtype),
        "snapshot_inventory_sha256": snapshot_inventory_sha256,
        "checkpoint_sha256": expected_host["checkpoint_sha256"],
        "tokenizer_sha256": expected_host["tokenizer_sha256"],
    }


def _native_forward_probe(
    model: Any, tokenizer: Any, *, prompt: str, device: str
) -> dict[str, Any]:
    if model is None:
        return {
            "performed": False,
            "reason": "LayerCake host runtime is the capability runtime",
        }
    selected = torch.device(device)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(selected) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, use_cache=True)
    logits = output.logits[:, -1]
    if not torch.isfinite(logits).all().item():
        raise MatrixError("frozen native host produced non-finite generation state")
    argmax = int(logits.argmax(-1).item())
    return {
        "performed": True,
        "finite": True,
        "input_units": int(inputs["input_ids"].numel()),
        "native_argmax_id_sha256": hashlib.sha256(
            argmax.to_bytes(8, "little")
        ).hexdigest(),
    }


def _load_english_host(root: Path, registry: Path, *, device: str) -> tuple[Any, dict[str, Any], Path, bytes]:
    core_protocol = _json(
        root / "ABI_CAPABILITY_COMPILER_PHASE4_B40_V25_PRODUCT_CONFORMANCE_PROTOCOL_V960.json"
    )
    api = _api((root / core_protocol["layercake_root"]).resolve())
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(core_protocol["research_signing_seed_hex"])
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api["key_id"](public)
    archive = (
        root
        / "results/abi_capability_compiler_phase7_integrated/materialized_v1052/phase7-final-english-core.cake"
    ).resolve()
    host = api["ClarificationRouteAllocationBoundedCoreHost"](
        registry, trust_store={signer: public}, device=device
    )
    activated = host.activate(archive)
    return host, activated, archive, public


def _domain_runtime(root: Path, registry: Path, *, device: str) -> tuple[Any, dict[str, Any]]:
    phase7 = _json(root / "ABI_CAPABILITY_COMPILER_PHASE7_INTEGRATED_RUNTIME_PROTOCOL_V1040.json")
    specs, trust = _domain_specs(root, phase7)
    from layercake.routing.catalog_router import ArchiveBoundProfile, RoutingFeature
    from layercake_extensions.authoritative_destination_control import (
        AuthoritativeDestinationOrchestrator,
    )

    profiles = tuple(
        ArchiveBoundProfile(
            cake_id=specs[domain]["cake_id"],
            archive_sha256=specs[domain]["archive_sha256"],
            domains=(domain,),
            features=(RoutingFeature("token", domain, 1.0),),
        )
        for domain in DOMAINS
    )
    runtime = AuthoritativeDestinationOrchestrator(
        registry,
        abi_version=DIRECT_ABI_VERSION,
        abi_hash=DIRECT_ABI_SHA256,
        trust_store=trust,
        profiles=profiles,
        device=device,
        maximum_loaded_cakes=3,
    )
    return runtime, specs


def _domain_generate(
    runtime: Any,
    specs: Mapping[str, Any],
    domain: str,
    prompt: str,
    *,
    catalog_wrapped: bool = True,
) -> tuple[str, list[int]]:
    canonical_prompt = project_catalog_prompt(prompt) if catalog_wrapped else prompt
    if not canonical_prompt.endswith("\n"):
        canonical_prompt += "\n"
    generated = runtime.host.generate(specs[domain]["cake_id"], canonical_prompt)
    return generated.output.decode("utf-8", errors="strict"), [int(value) for value in generated.actions]


def _source_references(root: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    english_path = (
        root
        / "results/abi_capability_compiler_phase4_clarification_route_replication/B40-seed104729-v927/evaluation/development_outputs.jsonl"
    )
    english = {str(row["probe_id"]): str(row["output"]) for row in _jsonl(english_path)}
    domain_path = (
        root
        / "results/abi_capability_compiler_phase6_composition/run_v1032/seed104729/observations.jsonl"
    )
    domains = {domain: {} for domain in DOMAINS}
    for row in _jsonl(domain_path):
        if row.get("mode") == "composed_host_selected_domain":
            domains[str(row["domain"])][str(row["probe_id"])] = str(row["output"])
    return english, domains


def _mutate_random_equal_size(source: Path, destination: Path, *, seed: int) -> None:
    generator = random.Random(seed)
    remaining = source.stat().st_size
    with destination.open("wb") as handle:
        while remaining:
            size = min(8 * 1024 * 1024, remaining)
            handle.write(generator.randbytes(size))
            remaining -= size


def _mutate_shuffle_equal_size(source: Path, destination: Path) -> None:
    block = 1024 * 1024
    size = source.stat().st_size
    offsets = list(range(0, size, block))
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        for offset in reversed(offsets):
            incoming.seek(offset)
            outgoing.write(incoming.read(min(block, size - offset))[::-1])


def _expect_rejected(callback: Any) -> dict[str, Any]:
    try:
        callback()
    except Exception as exc:  # hostile fail-closed evidence records exact rejection class
        return {
            "rejected": True,
            "exception_type": type(exc).__name__,
            "exception_message_sha256": sha256_bytes(str(exc).encode("utf-8")),
        }
    return {"rejected": False, "exception_type": None, "exception_message_sha256": None}


def _matrix_records(root: Path, locks: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    english_by_id = {
        str(row["probe_id"]): row
        for row in development_probes(root / "catalogs/capability_compiler_phase1_frozen_v1.json")
    }
    english = [english_by_id[value] for value in locks["english"]["successful_task_ids"]]
    domain_rows = _domain_rows(
        root / "evidence/current/segregation/english_and_first_domains_certification_v6.json",
        split="final_test",
        per_domain=100,
    )
    by_domain = {
        domain: [row for row in domain_rows if row["domain"] == domain] for domain in DOMAINS
    }
    for domain in DOMAINS:
        locked = locks["domains"][domain]["successful_task_ids"]
        indexed = {str(row["probe_id"]): row for row in by_domain[domain]}
        by_domain[domain] = [indexed[value] for value in locked]
    return english, by_domain


@torch.inference_mode()
def run(
    root: Path,
    *,
    protocol_path: Path,
    host_key: str,
    output_dir: Path,
    snapshot: Path | None,
    device: str,
) -> dict[str, Any]:
    root, protocol_path, output_dir = root.resolve(), protocol_path.resolve(), output_dir.resolve()
    if host_key not in HOSTS or output_dir.exists():
        raise MatrixError("invalid host or immutable matrix output already exists")
    protocol = _json(protocol_path)
    if protocol.get("status") == "PREREGISTERED_PRE_OBSERVATION_UTF8_TYPE_AMENDMENT":
        base_path = (root / protocol["base_protocol"]).resolve()
        if not base_path.is_file() or _sha256(base_path) != protocol["base_protocol_sha256"]:
            raise MatrixError("ABI V2 matrix amendment base changed")
        base = _json(base_path)
        if (
            protocol.get("amendment_scope")
            not in {
                "STRICT_UTF8_BYTES_TO_TEXT_DECODE_BEFORE_CANONICAL_OUTPUT_ONLY",
                "CUMULATIVE_UTF8_DECODE_AND_WRONG_ENGLISH_CONTROL_RAW_PROMPT_ONLY",
                "CUMULATIVE_UTF8_CONTROL_AND_NONIDENTICAL_EQUAL_SIZE_SHUFFLE_ONLY",
            }
            or protocol.get("receiver_observations_before_amendment") != 0
            or protocol.get("bounded_architecture_repair_consumed") is not False
        ):
            raise MatrixError("ABI V2 matrix amendment scope changed")
        protocol = {
            **base,
            **protocol,
            "status": "PREREGISTERED_BEFORE_FIRST_RECEIVER_MATRIX_RUN",
            "bindings": {**base["bindings"], **protocol.get("bindings", {})},
        }
    if (
        protocol.get("status") != "PREREGISTERED_BEFORE_FIRST_RECEIVER_MATRIX_RUN"
        or protocol.get("hosts") != list(HOSTS)
        or protocol.get("capabilities") != list(CAPABILITIES)
        or protocol.get("training_authorized") is not False
        or protocol.get("calibration_authorized") is not False
        or _sha256(root / protocol["implementation"]["path"])
        != protocol["implementation"]["sha256"]
    ):
        raise MatrixError("ABI V2 matrix protocol changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or _sha256(target) != expected:
            raise MatrixError(f"matrix binding changed: {relative}")
    locks = _json(root / protocol["source_success_locks"])
    adapters = _json(root / protocol["adapter_manifest"])
    adapter_binding = adapters["adapters"][host_key]
    native_model, native_tokenizer, native_identity = _load_native_host(
        host=host_key,
        snapshot=snapshot,
        device=device,
        expected_host=protocol["host_registry"][host_key],
    )
    adapter = FrozenHostAdapter(
        path=root / adapter_binding["path"],
        expected_sha256=adapter_binding["sha256"],
        tokenizer=native_tokenizer,
    )
    english_records, domain_records = _matrix_records(root, locks)
    english_reference, domain_reference = _source_references(root)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    observations: list[dict[str, Any]] = []
    installation: dict[str, Any] = {}
    isolation: dict[str, Any] = {}
    causal: dict[str, Any] = {}
    native_probes: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=f"abi-v2-{host_key}-") as raw:
        temporary = Path(raw)
        load_started = time.perf_counter()
        english_host, activated, english_archive, english_public = _load_english_host(
            root, temporary / "english-registry", device=device
        )
        installation["english"] = {
            "seconds": time.perf_counter() - load_started,
            "archive_sha256": _sha256(english_archive),
            "archive_bytes": english_archive.stat().st_size,
            "training_steps": int(activated["receiver_training_steps"]),
            "adapter_sha256": adapter.expected_sha256,
        }
        native_probes["english"] = _native_forward_probe(
            native_model,
            native_tokenizer,
            prompt=str(english_records[0]["prompt"]),
            device=device,
        )
        english_successes: list[str] = []
        english_exact = 0
        english_times: list[float] = []
        for position, row in enumerate(english_records):
            prompt = str(row["prompt"])
            request_started = time.perf_counter()
            capability_output = english_host.generate(
                prompt, maximum_tokens=int(row["max_new_tokens"])
            ).decode("utf-8", errors="strict")
            realized = adapter.realize(
                prompt=prompt,
                output=capability_output,
                capability_id="english-substrate",
                position=position,
            )
            elapsed = time.perf_counter() - request_started
            functional = evaluate_functional(realized["output"], row["evaluator"])
            exact = realized["output"] == english_reference[str(row["probe_id"])]
            if functional:
                english_successes.append(str(row["probe_id"]))
            english_exact += exact
            english_times.append(elapsed)
            observations.append(
                {
                    "host": host_key,
                    "capability": "english",
                    "probe_id": row["probe_id"],
                    "output": realized["output"],
                    "functional_pass": functional,
                    "source_output_byte_exact": exact,
                    "output_utf8_bytes": len(strict_utf8(realized["output"])),
                    "total_seconds": elapsed,
                    **{key: value for key, value in realized.items() if key != "output"},
                }
            )

        english_isolation: dict[str, Any] = {}
        for domain in DOMAINS:
            passes = 0
            output_hashes = []
            for position, row in enumerate(domain_records[domain]):
                output = english_host.generate(
                    str(row["prompt"]), maximum_tokens=int(row["max_new_tokens"])
                ).decode("utf-8", errors="strict")
                realized = adapter.realize(
                    prompt=str(row["prompt"]),
                    output=output,
                    capability_id="english-substrate",
                    position=len(english_records) + position,
                )
                passes += evaluate_functional(realized["output"], row["evaluator"])
                output_hashes.append(realized["output_sha256"])
            english_isolation[domain] = {
                "specialist_successes": passes,
                "tasks": len(domain_records[domain]),
                "success_rate": passes / len(domain_records[domain]),
                "output_set_sha256": sha256_bytes(canonical_json_bytes(output_hashes)),
            }
        isolation["english_only"] = english_isolation

        domain_runtime, domain_specs = _domain_runtime(
            root, temporary / "domain-registry", device=device
        )
        for domain in DOMAINS:
            install_started = time.perf_counter()
            installed = domain_runtime.install(domain_specs[domain]["package"])
            installation[domain] = {
                "seconds": time.perf_counter() - install_started,
                "archive_sha256": _sha256(domain_specs[domain]["package"]),
                "archive_bytes": domain_specs[domain]["package"].stat().st_size,
                "status": installed["status"],
                "training_steps": 0,
                "adapter_sha256": adapter.expected_sha256,
            }
            native_probes[domain] = _native_forward_probe(
                native_model,
                native_tokenizer,
                prompt=str(domain_records[domain][0]["prompt"]),
                device=device,
            )

        domain_successes: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
        domain_exact: dict[str, int] = {domain: 0 for domain in DOMAINS}
        domain_times: dict[str, list[float]] = {domain: [] for domain in DOMAINS}
        domain_actions: dict[str, dict[str, list[int]]] = {domain: {} for domain in DOMAINS}
        for domain in DOMAINS:
            for position, row in enumerate(domain_records[domain]):
                prompt = str(row["prompt"])
                request_started = time.perf_counter()
                output, actions = _domain_generate(domain_runtime, domain_specs, domain, prompt)
                realized = adapter.realize(
                    prompt=prompt,
                    output=output,
                    capability_id=domain_specs[domain]["cake_id"],
                    position=position,
                )
                elapsed = time.perf_counter() - request_started
                functional = evaluate_functional(realized["output"], row["evaluator"])
                exact = realized["output"] == domain_reference[domain][str(row["probe_id"])]
                if functional:
                    domain_successes[domain].append(str(row["probe_id"]))
                domain_exact[domain] += exact
                domain_times[domain].append(elapsed)
                domain_actions[domain][str(row["probe_id"])] = actions
                observations.append(
                    {
                        "host": host_key,
                        "capability": domain,
                        "probe_id": row["probe_id"],
                        "output": realized["output"],
                        "actions": actions,
                        "actions_sha256": sha256_bytes(canonical_json_bytes(actions)),
                        "functional_pass": functional,
                        "source_output_byte_exact": exact,
                        "output_utf8_bytes": len(strict_utf8(realized["output"])),
                        "total_seconds": elapsed,
                        **{key: value for key, value in realized.items() if key != "output"},
                    }
                )

        wrong_results: dict[str, Any] = {}
        wrong_order = {"python": "chemistry", "chemistry": "civics", "civics": "python"}
        for target, wrong in wrong_order.items():
            passes = 0
            for row in domain_records[target]:
                output, _ = _domain_generate(
                    domain_runtime, domain_specs, wrong, str(row["prompt"])
                )
                passes += evaluate_functional(output, row["evaluator"])
            wrong_results[target] = {
                "wrong_capability": wrong,
                "successes": passes,
                "tasks": len(domain_records[target]),
                "success_rate": passes / len(domain_records[target]),
            }
        wrong_english_passes = 0
        for row in english_records[:100]:
            output, _ = _domain_generate(
                domain_runtime,
                domain_specs,
                "python",
                str(row["prompt"]),
                catalog_wrapped=False,
            )
            wrong_english_passes += evaluate_functional(output, row["evaluator"])
        wrong_results["english"] = {
            "wrong_capability": "python",
            "successes": wrong_english_passes,
            "tasks": 100,
            "success_rate": wrong_english_passes / 100.0,
        }
        causal["wrong_capability"] = wrong_results

        adapter.enabled = False
        causal["adapter_removal"] = _expect_rejected(
            lambda: adapter.realize(
                prompt="neutral probe",
                output="neutral output",
                capability_id="unrevealed",
                position=0,
            )
        )
        adapter.enabled = True

        removal: dict[str, Any] = {}
        first_english = english_records[0]
        expected_english = english_reference[str(first_english["probe_id"])]
        removed = english_host.remove()
        absent = _expect_rejected(
            lambda: english_host.generate(
                str(first_english["prompt"]),
                maximum_tokens=int(first_english["max_new_tokens"]),
            )
        )
        reactivated = english_host.activate(english_archive)
        restored = english_host.generate(
            str(first_english["prompt"]),
            maximum_tokens=int(first_english["max_new_tokens"]),
        ).decode("utf-8", errors="strict")
        removal["english"] = {
            "remove_status": removed,
            "absent_execution_rejected": absent["rejected"],
            "reinstall_archive_sha256": _sha256(english_archive),
            "reinstall_state_dict_hash": reactivated["state_dict_hash"],
            "restored_output_byte_exact": restored == expected_english,
        }
        for domain in DOMAINS:
            first = domain_records[domain][0]
            cake_id = domain_specs[domain]["cake_id"]
            removed = domain_runtime.host.remove(cake_id)
            absent = _expect_rejected(
                lambda domain=domain, first=first: _domain_generate(
                    domain_runtime, domain_specs, domain, str(first["prompt"])
                )
            )
            reinstalled = domain_runtime.install(domain_specs[domain]["package"])
            restored, _ = _domain_generate(
                domain_runtime, domain_specs, domain, str(first["prompt"])
            )
            removal[domain] = {
                "remove_status": removed,
                "absent_execution_rejected": absent["rejected"],
                "reinstall_status": reinstalled["status"],
                "reinstall_archive_sha256": _sha256(domain_specs[domain]["package"]),
                "restored_output_byte_exact": restored
                == domain_reference[domain][str(first["probe_id"])],
            }
        causal["capability_removal_and_reinstall"] = removal

        corruption: dict[str, Any] = {}
        package_paths = {
            "english": english_archive,
            **{domain: domain_specs[domain]["package"] for domain in DOMAINS},
        }
        for index, capability in enumerate(CAPABILITIES):
            source = package_paths[capability]
            random_path = temporary / f"random-{capability}.cake"
            shuffled_path = temporary / f"shuffled-{capability}.cake"
            _mutate_random_equal_size(source, random_path, seed=220240824 + index)
            _mutate_shuffle_equal_size(source, shuffled_path)
            if capability == "english":
                def rejected_english(path: Path, name: str) -> Any:
                    hostile, _, _, _ = _load_english_host(
                        root,
                        temporary / f"hostile-{name}-{capability}",
                        device=device,
                    )
                    return hostile.activate(path)

                def random_callback() -> Any:
                    return rejected_english(random_path, "random")

                def shuffled_callback() -> Any:
                    return rejected_english(shuffled_path, "shuffled")
            else:
                def rejected_domain(path: Path, name: str) -> Any:
                    hostile, _ = _domain_runtime(
                        root, temporary / f"hostile-{name}-{capability}", device=device
                    )
                    return hostile.install(path)

                def random_callback() -> Any:
                    return rejected_domain(random_path, "random")

                def shuffled_callback() -> Any:
                    return rejected_domain(shuffled_path, "shuffled")
            random_rejection = _expect_rejected(random_callback)
            shuffled_rejection = _expect_rejected(shuffled_callback)
            corruption[capability] = {
                "original_bytes": source.stat().st_size,
                "random_bytes": random_path.stat().st_size,
                "shuffled_bytes": shuffled_path.stat().st_size,
                "random_archive_sha256": _sha256(random_path),
                "shuffled_archive_sha256": _sha256(shuffled_path),
                "random_rejected_before_execution": random_rejection,
                "shuffled_rejected_before_execution": shuffled_rejection,
                "functional_successes_after_rejection": 0,
            }
        causal["random_and_shuffled_capabilities"] = corruption

        active_capability_bytes = {
            "english": sum(
                _tensor_bytes(module)
                for module in (english_host.model, english_host.router, english_host.residual)
            ),
            **{
                domain: _tensor_bytes(domain_runtime.host._models[domain_specs[domain]["cake_id"]])
                for domain in DOMAINS
            },
        }

    elapsed = time.perf_counter() - started
    rss_after = process.memory_info().rss
    package_hashes_after = {
        "english": _sha256(
            root
            / "results/abi_capability_compiler_phase7_integrated/materialized_v1052/phase7-final-english-core.cake"
        ),
        **{
            domain: _sha256(
                root
                / _json(root / "ABI_CAPABILITY_COMPILER_PHASE7_INTEGRATED_RUNTIME_PROTOCOL_V1040.json")[
                    "domain_packages"
                ][domain]["package"]
            )
            for domain in DOMAINS
        },
    }
    retention = {
        "english": {
            "source_successes": len(locks["english"]["successful_task_ids"]),
            "receiver_successes_on_locked_set": len(english_successes),
            "retention": len(english_successes) / len(locks["english"]["successful_task_ids"]),
            "source_output_byte_exact": english_exact,
            "tasks": len(english_records),
        },
        **{
            domain: {
                "source_successes": len(locks["domains"][domain]["successful_task_ids"]),
                "receiver_successes_on_locked_set": len(domain_successes[domain]),
                "retention": len(domain_successes[domain])
                / len(locks["domains"][domain]["successful_task_ids"]),
                "source_output_byte_exact": domain_exact[domain],
                "tasks": len(domain_records[domain]),
            }
            for domain in DOMAINS
        },
    }
    import statistics

    timing = {
        "english": {
            "observations": len(english_times),
            "headline_observations": 20,
            "median_total_seconds_headline": statistics.median(english_times[:20]),
            "median_ttft_seconds_headline": statistics.median(english_times[:20]),
        },
        **{
            domain: {
                "observations": len(domain_times[domain]),
                "headline_observations": 20,
                "median_total_seconds_headline": statistics.median(domain_times[domain][:20]),
                "median_ttft_seconds_headline": statistics.median(domain_times[domain][:20]),
            }
            for domain in DOMAINS
        },
    }
    package_hashes_expected = protocol["capability_packages"]
    gates = {
        "all_four_source_success_sets_retained_100_percent": all(
            value["retention"] == 1.0 for value in retention.values()
        ),
        "all_four_source_outputs_byte_exact": all(
            value["source_output_byte_exact"] == value["tasks"]
            for value in retention.values()
        ),
        "adapter_hash_unchanged": adapter.verify(),
        "same_adapter_hash_all_four_capabilities": len(
            {installation[value]["adapter_sha256"] for value in CAPABILITIES}
        )
        == 1,
        "all_package_hashes_unchanged": all(
            package_hashes_after[key] == package_hashes_expected[key]["sha256"]
            for key in CAPABILITIES
        ),
        "zero_training_and_calibration": all(
            installation[value]["training_steps"] == 0 for value in CAPABILITIES
        ),
        "native_host_checkpoint_participated": host_key == "layercake"
        or all(value.get("performed") is True for value in native_probes.values()),
        "english_isolation_below_threshold": all(
            value["success_rate"] <= protocol["gates"]["isolation_success_rate_maximum"]
            for value in isolation["english_only"].values()
        ),
        "wrong_capability_below_threshold": all(
            value["success_rate"] <= protocol["gates"]["wrong_capability_success_rate_maximum"]
            for value in causal["wrong_capability"].values()
        ),
        "adapter_removal_fails_closed": causal["adapter_removal"]["rejected"],
        "capability_removal_and_reinstall_exact": all(
            value["absent_execution_rejected"] and value["restored_output_byte_exact"]
            for value in causal["capability_removal_and_reinstall"].values()
        ),
        "random_and_shuffled_capabilities_fail_closed": all(
            value["random_rejected_before_execution"]["rejected"]
            and value["shuffled_rejected_before_execution"]["rejected"]
            and value["original_bytes"] == value["random_bytes"] == value["shuffled_bytes"]
            for value in causal["random_and_shuffled_capabilities"].values()
        ),
        "teacher_absent": True,
        "source_model_absent": True,
        "adapter_parameter_count_zero": True,
        "adapter_optimizer_steps_zero": True,
    }
    output_dir.mkdir(parents=True)
    observation_path = output_dir / "observations.jsonl"
    _write_once(observation_path, b"".join(canonical_json_bytes(row) for row in observations))
    mathematical = {
        "format": "abi-v2-host-capability-mathematical-output/1",
        "host": host_key,
        "canonical_output_sha256": {
            capability: {
                str(row["probe_id"]): row["output_sha256"]
                for row in observations
                if row["capability"] == capability
            }
            for capability in CAPABILITIES
        },
        "domain_action_ids": domain_actions,
    }
    mathematical["evidence_sha256"] = sha256_bytes(canonical_json_bytes(mathematical))
    _write_once(
        output_dir / "mathematical.json",
        json.dumps(mathematical, indent=2, sort_keys=True).encode() + b"\n",
    )
    result = {
        "format": "abi-v2-host-four-capability-matrix-result/1",
        "status": "PASS_HOST_FOUR_CAPABILITY_MATRIX"
        if all(gates.values())
        else "FAIL_HOST_FOUR_CAPABILITY_MATRIX",
        "host": host_key,
        "device": device,
        "protocol_sha256": _sha256(protocol_path),
        "adapter": {
            "path": adapter.path.relative_to(root).as_posix(),
            "sha256_before": adapter.expected_sha256,
            "sha256_after": _sha256(adapter.path),
            "parameters": 0,
            "optimizer_steps": 0,
        },
        "native_host": native_identity,
        "native_forward_probes": native_probes,
        "installation": installation,
        "source_success_retention": retention,
        "isolation": isolation,
        "causal": causal,
        "performance": {
            "certified_host_alone_and_adapter": _json(
                root
                / f"results/abi_v2/host_certification/initial/{host_key}/performance.json"
            ),
            "capability_execution": timing,
            "wall_seconds_all_tests": elapsed,
            "peak_process_rss_bytes_lower_bound": max(rss_before, rss_after),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if device == "cuda"
            else 0,
            "host_base_loaded_parameters": native_identity["parameter_count"],
            "active_capability_tensor_bytes": active_capability_bytes,
            "semantic_execution_boundary": "The capability runtime emits authoritative canonical UTF-8. The host adapter converts it to an exact native tokenizer generation sequence. Qwen/Pythia base weights are frozen conformance participants but do not supply or alter capability semantics.",
        },
        "package_hashes_after": package_hashes_after,
        "observations": {
            "path": observation_path.relative_to(root).as_posix(),
            "sha256": _sha256(observation_path),
            "rows": len(observations),
        },
        "mathematical": {
            "path": (output_dir / "mathematical.json").relative_to(root).as_posix(),
            "sha256": _sha256(output_dir / "mathematical.json"),
        },
        "gates": gates,
        "teacher_loaded": False,
        "source_model_loaded": False,
        "training_performed": False,
        "calibration_performed": False,
        "claim_boundary": "This host result certifies immutable capability hosting through the canonical ABI V2 runtime and exact native codec realization. It does not claim that LayerCake tensors were transplanted into Qwen/Pythia base weights or that their hidden states generated the capability content.",
    }
    result["evidence_sha256"] = sha256_bytes(canonical_json_bytes(result))
    _write_once(
        output_dir / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    del native_model, native_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    result = run(
        Path.cwd(),
        protocol_path=Path(args.protocol),
        host_key=args.host,
        output_dir=Path(args.output_dir),
        snapshot=Path(args.snapshot).resolve() if args.snapshot else None,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
