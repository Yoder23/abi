"""Execute live frozen-release host-causality interventions.

This module never reads prior matrix outputs.  Every condition invokes the
installed immutable capability anew.  Expected-answer ledgers are deliberately
absent from the execution path; the verifier compares new condition outputs to
the new real-host outputs from the same run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .canonical import canonical_json_bytes, sha256_bytes, strict_utf8
from .capability_matrix import (
    CAPABILITIES,
    DOMAINS,
    FrozenHostAdapter,
    _domain_generate,
    _domain_runtime,
    _json,
    _load_english_host,
    _load_native_host,
    _matrix_records,
    _sha256,
)
from .execution_sources import execution_source_manifest

CONDITIONS = (
    "real_host",
    "neutral_host",
    "zero_state",
    "random_state",
    "shuffled_state",
    "host_removed",
    "adapter_removed",
    "capability_removed",
)
SAMPLE_SEED = 2026082501


class LiveCausalityError(RuntimeError):
    """Raised when a live causal execution cannot be completed exactly."""


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise LiveCausalityError(f"immutable causal evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _selected(
    english: list[dict[str, Any]],
    domains: Mapping[str, list[dict[str, Any]]],
    count: int,
) -> dict[str, list[dict[str, Any]]]:
    def choose(rows: list[dict[str, Any]], capability: str) -> list[dict[str, Any]]:
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{SAMPLE_SEED}:{capability}:{row['probe_id']}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < count:
            raise LiveCausalityError(f"insufficient causal rows for {capability}")
        return ranked[:count]

    return {
        "english": choose(english, "english"),
        **{domain: choose(list(domains[domain]), domain) for domain in DOMAINS},
    }


def _model_state(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    device: str,
) -> list[float]:
    if model is None:
        return []
    selected = torch.device(device)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {name: value.to(selected) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, use_cache=False)
    logits = output.logits[0, -1, :32].detach().float().cpu()
    if not torch.isfinite(logits).all().item():
        raise LiveCausalityError("real host state contained non-finite values")
    return [float(value) for value in logits.tolist()]


def _parameter_state(module: Any) -> list[float]:
    for parameter in module.parameters():
        values = parameter.detach().reshape(-1)[:32].float().cpu()
        if values.numel():
            return [float(value) for value in values.tolist()]
    return []


def _condition_state(
    real: list[float],
    *,
    condition: str,
    key: str,
) -> list[float]:
    if condition == "real_host":
        return list(real)
    if condition == "neutral_host":
        return [0.5] * max(1, len(real))
    if condition == "zero_state":
        return [0.0] * max(1, len(real))
    if condition == "random_state":
        generator = random.Random(f"{SAMPLE_SEED}:{key}")
        return [generator.uniform(-1.0, 1.0) for _ in range(max(1, len(real)))]
    if condition == "shuffled_state":
        values = list(real) if real else [float(index) for index in range(32)]
        random.Random(f"{SAMPLE_SEED}:shuffle:{key}").shuffle(values)
        return values
    if condition == "host_removed":
        return []
    raise LiveCausalityError(f"state requested for non-state condition: {condition}")


def _generate(
    *,
    capability: str,
    row: Mapping[str, Any],
    english_host: Any,
    domain_runtime: Any,
    domain_specs: Mapping[str, Any],
) -> tuple[str, list[int]]:
    prompt = str(row["prompt"])
    if capability == "english":
        output = english_host.generate(
            prompt, maximum_tokens=int(row["max_new_tokens"])
        ).decode("utf-8", errors="strict")
        return output, []
    return _domain_generate(domain_runtime, domain_specs, capability, prompt)


def _remove(
    capability: str,
    *,
    english_host: Any,
    domain_runtime: Any,
    domain_specs: Mapping[str, Any],
) -> Any:
    if capability == "english":
        return english_host.remove()
    return domain_runtime.host.remove(domain_specs[capability]["cake_id"])


def _reinstall(
    capability: str,
    *,
    english_host: Any,
    english_archive: Path,
    domain_runtime: Any,
    domain_specs: Mapping[str, Any],
) -> Any:
    if capability == "english":
        return english_host.activate(english_archive)
    return domain_runtime.install(domain_specs[capability]["package"])


def _failure(callback: Any) -> dict[str, Any]:
    try:
        callback()
    except Exception as exc:  # raw causal receipt records the actual failure class
        return {
            "exception_type": type(exc).__name__,
            "exception_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8")
            ).hexdigest(),
        }
    return {"exception_type": None, "exception_message_sha256": None}


class AppliedHostStateAdapter:
    """Consume an intervention state before exact frozen-adapter realization."""

    def __init__(self, adapter: FrozenHostAdapter) -> None:
        self.adapter = adapter

    def realize(
        self,
        *,
        prompt: str,
        output: str,
        capability_id: str,
        position: int,
        host_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        required = {
            "condition",
            "intervention_sha256",
            "state_vector",
        }
        if set(host_state) != required:
            raise LiveCausalityError("host intervention state schema changed")
        vector = host_state["state_vector"]
        if not isinstance(vector, list) or any(
            not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value))
            for value in vector
        ):
            raise LiveCausalityError("host intervention state is not finite")
        trace = sha256_bytes(canonical_json_bytes(dict(host_state)))
        realized = self.adapter.realize(
            prompt=prompt,
            output=output,
            capability_id=capability_id,
            position=position,
        )
        return {**realized, "applied_host_state_sha256": trace}


def _native_parameter_intervention(model: Any, *, condition: str, host_key: str) -> dict[str, Any]:
    if model is None:
        value = {
            "kind": "structural_native_host_absence",
            "condition": condition,
            "parameter_name": None,
            "values_before": [],
            "values_after": [],
        }
        value["intervention_sha256"] = sha256_bytes(canonical_json_bytes(value))
        return value
    selected_name = ""
    selected_parameter = None
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.ndim == 1 and parameter.numel() >= 32:
            selected_name = name
            selected_parameter = parameter
            break
    if selected_parameter is None:
        raise LiveCausalityError("native host has no eligible causal intervention tensor")
    flat = selected_parameter.detach().reshape(-1)
    count = min(4096, int(flat.numel()))
    before = [float(value) for value in flat[:count].float().cpu().tolist()]
    after = list(before)
    seed = f"{SAMPLE_SEED}:{host_key}:{condition}:{selected_name}"
    if condition == "neutral_host":
        after = [0.5] * count
    elif condition == "zero_state":
        after = [0.0] * count
    elif condition == "random_state":
        generator = random.Random(seed)
        # Binary fractions survive float16/float32 assignment exactly, allowing
        # the verifier to reconstruct the requested live mutation byte-for-byte.
        after = [generator.randrange(-32, 33) / 1024.0 for _ in range(count)]
    elif condition == "shuffled_state":
        random.Random(seed).shuffle(after)
    elif condition not in {"real_host", "adapter_removed", "capability_removed"}:
        raise LiveCausalityError(f"unsupported native intervention: {condition}")
    if condition in {"neutral_host", "zero_state", "random_state", "shuffled_state"}:
        replacement = torch.tensor(
            after, dtype=selected_parameter.dtype, device=selected_parameter.device
        )
        with torch.no_grad():
            flat[:count].copy_(replacement)
        observed = [float(value) for value in flat[:count].float().cpu().tolist()]
    else:
        observed = before
    value = {
        "kind": "live_native_parameter_intervention",
        "condition": condition,
        "parameter_name": selected_name,
        "parameter_shape": [int(value) for value in selected_parameter.shape],
        "elements_intervened": count,
        "seed": seed,
        "values_before": before,
        "values_after": observed,
        "before_sha256": sha256_bytes(canonical_json_bytes(before)),
        "after_sha256": sha256_bytes(canonical_json_bytes(observed)),
    }
    value["intervention_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


def _state_for_prompt(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    condition: str,
    host_key: str,
    intervention: Mapping[str, Any],
) -> dict[str, Any]:
    if model is not None:
        vector = _model_state(model, tokenizer, prompt=prompt, device=device)
    elif condition == "real_host":
        vector = [1.0]
    else:
        vector = _condition_state(
            [1.0],
            condition=condition if condition in CONDITIONS[:6] else "real_host",
            key=f"{host_key}:{condition}:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}",
        )
    return {
        "condition": condition,
        "intervention_sha256": intervention["intervention_sha256"],
        "state_vector": vector,
    }


@torch.inference_mode()
def run_condition(
    root: Path,
    *,
    host_key: str,
    condition: str,
    output_dir: Path,
    snapshot: Path | None,
    device: str,
    samples_per_capability: int = 32,
) -> dict[str, Any]:
    """Run exactly one intervention in a fresh operating-system process."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise LiveCausalityError(f"immutable causal output exists: {output_dir}")
    if condition not in CONDITIONS:
        raise LiveCausalityError(f"unsupported causal condition: {condition}")
    protocol_path = root / "abi_v2/matrix_protocol_amendment3.json"
    protocol = _json(protocol_path)
    base = _json(root / protocol["base_protocol"])
    merged = {**base, **protocol}
    locks = _json(root / merged["source_success_locks"])
    adapter_manifest = _json(root / merged["adapter_manifest"])
    adapter_binding = adapter_manifest["adapters"][host_key]
    snapshot_argument = "present" if snapshot is not None else "absent"
    if condition == "host_removed":
        if snapshot is not None:
            raise LiveCausalityError("host-removed worker received a checkpoint path")
        native_model = native_tokenizer = None
        native_identity = {
            "parameter_count": 0,
            "runtime_mode": "physically_removed_no_snapshot_no_object",
            "host_registry_identity": merged["host_registry"][host_key],
        }
    else:
        native_model, native_tokenizer, native_identity = _load_native_host(
            host=host_key,
            snapshot=snapshot,
            device=device,
            expected_host=merged["host_registry"][host_key],
        )
        native_identity = {
            key: value
            for key, value in native_identity.items()
            if key != "checkpoint_loaded"
        }
        native_identity["runtime_mode"] = (
            "capability_native_layercake_host"
            if host_key == "layercake"
            else "live_transformer_checkpoint"
        )
    intervention = _native_parameter_intervention(
        native_model, condition=condition, host_key=host_key
    )
    adapter = FrozenHostAdapter(
        path=root / adapter_binding["path"],
        expected_sha256=adapter_binding["sha256"],
        tokenizer=native_tokenizer,
    )
    applied_adapter = AppliedHostStateAdapter(adapter)
    english_records, domain_records = _matrix_records(root, locks)
    selected = _selected(english_records, domain_records, samples_per_capability)
    observations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"abi-live-causal-{host_key}-") as raw:
        temporary = Path(raw)
        english_host, _, english_archive, _ = _load_english_host(
            root, temporary / "english-registry", device=device
        )
        domain_runtime, domain_specs = _domain_runtime(
            root, temporary / "domain-registry", device=device
        )
        for domain in DOMAINS:
            domain_runtime.install(domain_specs[domain]["package"])

        if condition in CONDITIONS[:6]:
            for capability in CAPABILITIES:
                for position, row in enumerate(selected[capability]):
                    prompt = str(row["prompt"])
                    host_state = _state_for_prompt(
                        model=native_model,
                        tokenizer=native_tokenizer,
                        prompt=prompt,
                        device=device,
                        condition=condition,
                        host_key=host_key,
                        intervention=intervention,
                    )
                    request_started = time.perf_counter_ns()
                    output, actions = _generate(
                        capability=capability,
                        row=row,
                        english_host=english_host,
                        domain_runtime=domain_runtime,
                        domain_specs=domain_specs,
                    )
                    realized = applied_adapter.realize(
                        prompt=prompt,
                        output=output,
                        capability_id=(
                            "english-substrate"
                            if capability == "english"
                            else domain_specs[capability]["cake_id"]
                        ),
                        position=position,
                        host_state=host_state,
                    )
                    observations.append(
                        {
                            "host": host_key,
                            "condition": condition,
                            "capability": capability,
                            "probe_id": row["probe_id"],
                            "prompt_sha256": hashlib.sha256(
                                prompt.encode("utf-8")
                            ).hexdigest(),
                            "host_state": host_state,
                            "host_state_sha256": sha256_bytes(
                                canonical_json_bytes(host_state)
                            ),
                            "applied_host_state_sha256": realized[
                                "applied_host_state_sha256"
                            ],
                            "capability_output": output,
                            "capability_output_sha256": hashlib.sha256(
                                strict_utf8(output)
                            ).hexdigest(),
                            "realized_output": realized["output"],
                            "realized_output_sha256": realized["output_sha256"],
                            "actions": actions,
                            "actions_sha256": sha256_bytes(canonical_json_bytes(actions)),
                            "elapsed_ns": time.perf_counter_ns() - request_started,
                            "exception_type": None,
                        }
                    )

        elif condition == "adapter_removed":
            adapter.enabled = False
            for capability in CAPABILITIES:
                for position, row in enumerate(selected[capability]):
                    prompt = str(row["prompt"])
                    host_state = _state_for_prompt(
                        model=native_model,
                        tokenizer=native_tokenizer,
                        prompt=prompt,
                        device=device,
                        condition=condition,
                        host_key=host_key,
                        intervention=intervention,
                    )
                    output, actions = _generate(
                        capability=capability,
                        row=row,
                        english_host=english_host,
                        domain_runtime=domain_runtime,
                        domain_specs=domain_specs,
                    )
                    failure = _failure(
                        lambda output=output, capability=capability, row=row, position=position, host_state=host_state: applied_adapter.realize(
                            prompt=str(row["prompt"]),
                            output=output,
                            capability_id=(
                                "english-substrate"
                                if capability == "english"
                                else domain_specs[capability]["cake_id"]
                            ),
                            position=position,
                            host_state=host_state,
                        )
                    )
                    observations.append(
                        {
                            "host": host_key,
                            "condition": condition,
                            "capability": capability,
                            "probe_id": row["probe_id"],
                            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                            "host_state": host_state,
                            "host_state_sha256": sha256_bytes(
                                canonical_json_bytes(host_state)
                            ),
                            "capability_output": output,
                            "capability_output_sha256": hashlib.sha256(
                                strict_utf8(output)
                            ).hexdigest(),
                            "actions": actions,
                            "realized_output": None,
                            **failure,
                        }
                    )
            adapter.enabled = True
        else:
            for capability in CAPABILITIES:
                _remove(
                    capability,
                    english_host=english_host,
                    domain_runtime=domain_runtime,
                    domain_specs=domain_specs,
                )
                for row in selected[capability]:
                    prompt = str(row["prompt"])
                    host_state = _state_for_prompt(
                        model=native_model,
                        tokenizer=native_tokenizer,
                        prompt=prompt,
                        device=device,
                        condition=condition,
                        host_key=host_key,
                        intervention=intervention,
                    )
                    failure = _failure(
                        lambda capability=capability, row=row: _generate(
                            capability=capability,
                            row=row,
                            english_host=english_host,
                            domain_runtime=domain_runtime,
                            domain_specs=domain_specs,
                        )
                    )
                    observations.append(
                        {
                            "host": host_key,
                            "condition": condition,
                            "capability": capability,
                            "probe_id": row["probe_id"],
                            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                            "host_state": host_state,
                            "host_state_sha256": sha256_bytes(
                                canonical_json_bytes(host_state)
                            ),
                            "capability_output": None,
                            "realized_output": None,
                            **failure,
                        }
                    )

    output_dir.mkdir(parents=True)
    raw_path = output_dir / "observations.jsonl"
    _write_once(raw_path, b"".join(canonical_json_bytes(row) for row in observations))
    receipt: dict[str, Any] = {
        "format": "abi-v2-live-host-condition/3",
        "host": host_key,
        "condition": condition,
        "device": device,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "process_started_ns": time.time_ns(),
        "snapshot_argument": snapshot_argument,
        "native_host": native_identity,
        "intervention": intervention,
        "samples_per_capability": samples_per_capability,
        "observations_rows": len(observations),
        "observations_sha256": _sha256(raw_path),
        "execution_source_sha256": _sha256(Path(__file__).resolve()),
        "transitive_execution_sources": execution_source_manifest(root),
        "adapter_sha256": adapter_binding["sha256"],
        "capability_sha256": {
            "english": _sha256(english_archive),
            **{domain: _sha256(domain_specs[domain]["package"]) for domain in DOMAINS},
        },
    }
    receipt["evidence_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    _write_once(
        output_dir / "receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    del native_model, native_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return receipt


def run(
    root: Path,
    *,
    host_key: str,
    output_dir: Path,
    snapshot: Path | None,
    device: str,
    samples_per_capability: int = 32,
) -> dict[str, Any]:
    """Orchestrate eight fresh intervention processes and bind their raw rows."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise LiveCausalityError(f"immutable causal output exists: {output_dir}")
    started = time.perf_counter()
    observations: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"abi-live-causal-parent-{host_key}-") as raw:
        staging = Path(raw)
        for condition in CONDITIONS:
            condition_output = staging / condition
            command = [
                sys.executable,
                "-B",
                "-m",
                "abi_v2.live_causality",
                "--condition-worker",
                "--host",
                host_key,
                "--condition",
                condition,
                "--output-dir",
                str(condition_output),
                "--device",
                device,
                "--samples-per-capability",
                str(samples_per_capability),
            ]
            if condition != "host_removed" and snapshot is not None:
                command.extend(("--snapshot", str(snapshot.resolve())))
            environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise LiveCausalityError(
                    f"fresh {host_key}/{condition} process failed ({completed.returncode}): "
                    f"{completed.stderr[-4000:]}"
                )
            receipt = _json(condition_output / "receipt.json")
            rows = [
                json.loads(line)
                for line in (condition_output / "observations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            if receipt.get("observations_sha256") != _sha256(
                condition_output / "observations.jsonl"
            ):
                raise LiveCausalityError(f"condition output binding changed: {condition}")
            receipts.append(receipt)
            observations.extend(rows)
        output_dir.mkdir(parents=True)
        condition_dir = output_dir / "conditions"
        condition_dir.mkdir()
        for condition in CONDITIONS:
            shutil.copy2(staging / condition / "receipt.json", condition_dir / f"{condition}.json")

    raw_path = output_dir / "observations.jsonl"
    _write_once(raw_path, b"".join(canonical_json_bytes(row) for row in observations))
    selected_probe_ids = {
        capability: [
            str(row["probe_id"])
            for row in observations
            if row["condition"] == "real_host" and row["capability"] == capability
        ]
        for capability in CAPABILITIES
    }
    manifest: dict[str, Any] = {
        "format": "abi-v2-live-host-causality-run/3",
        "host": host_key,
        "device": device,
        "conditions": list(CONDITIONS),
        "sample_seed": SAMPLE_SEED,
        "samples_per_capability": samples_per_capability,
        "selected_probe_ids": selected_probe_ids,
        "condition_processes": [
            {
                "condition": receipt["condition"],
                "process_id": receipt["process_id"],
                "receipt_sha256": _sha256(
                    output_dir / "conditions" / f"{receipt['condition']}.json"
                ),
                "observations_sha256": receipt["observations_sha256"],
            }
            for receipt in receipts
        ],
        "observations_path": "observations.jsonl",
        "observations_rows": len(observations),
        "observations_sha256": _sha256(raw_path),
        "wall_seconds": time.perf_counter() - started,
        "execution_source_sha256": _sha256(Path(__file__).resolve()),
        "transitive_execution_sources": execution_source_manifest(root),
    }
    manifest["evidence_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    _write_once(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=("layercake", "qwen2", "pythia"))
    parser.add_argument("--condition-worker", action="store_true")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--samples-per-capability", type=int, default=32)
    args = parser.parse_args(argv)
    callback = run_condition if args.condition_worker else run
    if args.condition_worker and args.condition is None:
        raise LiveCausalityError("condition worker requires --condition")
    keywords = {
        "host_key": args.host,
        "output_dir": Path(args.output_dir),
        "snapshot": Path(args.snapshot).resolve() if args.snapshot else None,
        "device": args.device,
        "samples_per_capability": args.samples_per_capability,
    }
    if args.condition_worker:
        keywords["condition"] = args.condition
    value = callback(Path.cwd(), **keywords)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
