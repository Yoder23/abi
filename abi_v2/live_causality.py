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
import random
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


@torch.inference_mode()
def run(
    root: Path,
    *,
    host_key: str,
    output_dir: Path,
    snapshot: Path | None,
    device: str,
    samples_per_capability: int = 32,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise LiveCausalityError(f"immutable causal output exists: {output_dir}")
    protocol_path = root / "abi_v2/matrix_protocol_amendment3.json"
    protocol = _json(protocol_path)
    base = _json(root / protocol["base_protocol"])
    merged = {**base, **protocol}
    locks = _json(root / merged["source_success_locks"])
    adapter_manifest = _json(root / merged["adapter_manifest"])
    adapter_binding = adapter_manifest["adapters"][host_key]
    native_model, native_tokenizer, native_identity = _load_native_host(
        host=host_key,
        snapshot=snapshot,
        device=device,
        expected_host=merged["host_registry"][host_key],
    )
    adapter = FrozenHostAdapter(
        path=root / adapter_binding["path"],
        expected_sha256=adapter_binding["sha256"],
        tokenizer=native_tokenizer,
    )
    neutral_adapter = FrozenHostAdapter(
        path=root / adapter_binding["path"],
        expected_sha256=adapter_binding["sha256"],
        tokenizer=None,
    )
    english_records, domain_records = _matrix_records(root, locks)
    selected = _selected(english_records, domain_records, samples_per_capability)
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()
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

        real_states: dict[tuple[str, str], list[float]] = {}
        for capability in CAPABILITIES:
            for row in selected[capability]:
                key = (capability, str(row["probe_id"]))
                if native_model is not None:
                    real_states[key] = _model_state(
                        native_model,
                        native_tokenizer,
                        prompt=str(row["prompt"]),
                        device=device,
                    )
                elif capability == "english":
                    real_states[key] = _parameter_state(english_host.model)
                else:
                    real_states[key] = _parameter_state(
                        domain_runtime.host._load_selected(
                            domain_specs[capability]["cake_id"]
                        )
                    )

        for condition in CONDITIONS[:6]:
            active_adapter = neutral_adapter if condition == "neutral_host" else adapter
            for capability in CAPABILITIES:
                for position, row in enumerate(selected[capability]):
                    prompt = str(row["prompt"])
                    key = f"{host_key}:{capability}:{row['probe_id']}:{condition}"
                    state = _condition_state(
                        real_states[(capability, str(row["probe_id"]))],
                        condition=condition,
                        key=key,
                    )
                    request_started = time.perf_counter_ns()
                    output, actions = _generate(
                        capability=capability,
                        row=row,
                        english_host=english_host,
                        domain_runtime=domain_runtime,
                        domain_specs=domain_specs,
                    )
                    realized = active_adapter.realize(
                        prompt=prompt,
                        output=output,
                        capability_id=(
                            "english-substrate"
                            if capability == "english"
                            else domain_specs[capability]["cake_id"]
                        ),
                        position=position,
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
                            "state_vector": state,
                            "state_sha256": sha256_bytes(canonical_json_bytes(state)),
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

        adapter.enabled = False
        for capability in CAPABILITIES:
            for position, row in enumerate(selected[capability]):
                output, actions = _generate(
                    capability=capability,
                    row=row,
                    english_host=english_host,
                    domain_runtime=domain_runtime,
                    domain_specs=domain_specs,
                )
                failure = _failure(
                    lambda output=output, capability=capability, row=row, position=position: adapter.realize(
                        prompt=str(row["prompt"]),
                        output=output,
                        capability_id=(
                            "english-substrate"
                            if capability == "english"
                            else domain_specs[capability]["cake_id"]
                        ),
                        position=position,
                    )
                )
                observations.append(
                    {
                        "host": host_key,
                        "condition": "adapter_removed",
                        "capability": capability,
                        "probe_id": row["probe_id"],
                        "prompt_sha256": hashlib.sha256(
                            str(row["prompt"]).encode("utf-8")
                        ).hexdigest(),
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

        for capability in CAPABILITIES:
            _remove(
                capability,
                english_host=english_host,
                domain_runtime=domain_runtime,
                domain_specs=domain_specs,
            )
            for row in selected[capability]:
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
                        "condition": "capability_removed",
                        "capability": capability,
                        "probe_id": row["probe_id"],
                        "prompt_sha256": hashlib.sha256(
                            str(row["prompt"]).encode("utf-8")
                        ).hexdigest(),
                        "capability_output": None,
                        "realized_output": None,
                        **failure,
                    }
                )
            _reinstall(
                capability,
                english_host=english_host,
                english_archive=english_archive,
                domain_runtime=domain_runtime,
                domain_specs=domain_specs,
            )

    output_dir.mkdir(parents=True)
    raw_path = output_dir / "observations.jsonl"
    _write_once(
        raw_path,
        b"".join(canonical_json_bytes(row) for row in observations),
    )
    manifest: dict[str, Any] = {
        "format": "abi-v2-live-host-causality-run/1",
        "host": host_key,
        "device": device,
        "conditions": list(CONDITIONS),
        "sample_seed": SAMPLE_SEED,
        "samples_per_capability": samples_per_capability,
        "selected_probe_ids": {
            capability: [str(row["probe_id"]) for row in selected[capability]]
            for capability in CAPABILITIES
        },
        "native_host": native_identity,
        "adapter_sha256_before": adapter_binding["sha256"],
        "adapter_sha256_after": _sha256(root / adapter_binding["path"]),
        "capability_sha256": {
            "english": _sha256(english_archive),
            **{
                domain: _sha256(domain_specs[domain]["package"])
                for domain in DOMAINS
            },
        },
        "observations_path": "observations.jsonl",
        "observations_rows": len(observations),
        "observations_sha256": _sha256(raw_path),
        "wall_seconds": time.perf_counter() - started,
        "execution_source_sha256": _sha256(Path(__file__).resolve()),
    }
    manifest["evidence_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    _write_once(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    del native_model, native_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=("layercake", "qwen2", "pythia"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--samples-per-capability", type=int, default=32)
    args = parser.parse_args(argv)
    value = run(
        Path.cwd(),
        host_key=args.host,
        output_dir=Path(args.output_dir),
        snapshot=Path(args.snapshot).resolve() if args.snapshot else None,
        device=args.device,
        samples_per_capability=args.samples_per_capability,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
