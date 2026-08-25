"""Run fresh raw capability-isolation executions without expected answers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from .canonical import canonical_json_bytes, sha256_bytes, strict_utf8
from .capability_matrix import (
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


class LiveIsolationError(RuntimeError):
    """Raised when fresh isolation evidence cannot be completed."""


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise LiveIsolationError(f"immutable isolation evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


@torch.inference_mode()
def run(
    root: Path,
    *,
    host_key: str,
    output_dir: Path,
    snapshot: Path | None,
    device: str,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise LiveIsolationError(f"immutable isolation output exists: {output_dir}")
    amendment = _json(root / "abi_v2/matrix_protocol_amendment3.json")
    base = _json(root / amendment["base_protocol"])
    protocol = {**base, **amendment}
    locks = _json(root / protocol["source_success_locks"])
    adapters = _json(root / protocol["adapter_manifest"])["adapters"]
    native_model, native_tokenizer, native_identity = _load_native_host(
        host=host_key,
        snapshot=snapshot,
        device=device,
        expected_host=protocol["host_registry"][host_key],
    )
    adapter = FrozenHostAdapter(
        path=root / adapters[host_key]["path"],
        expected_sha256=adapters[host_key]["sha256"],
        tokenizer=native_tokenizer,
    )
    english_records, domain_records = _matrix_records(root, locks)
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"abi-live-isolation-{host_key}-") as raw:
        temporary = Path(raw)
        english_host, _, english_archive, _ = _load_english_host(
            root, temporary / "english-registry", device=device
        )
        for target in DOMAINS:
            for position, row in enumerate(domain_records[target]):
                output = english_host.generate(
                    str(row["prompt"]), maximum_tokens=int(row["max_new_tokens"])
                ).decode("utf-8", errors="strict")
                realized = adapter.realize(
                    prompt=str(row["prompt"]),
                    output=output,
                    capability_id="english-substrate",
                    position=position,
                )
                observations.append(
                    {
                        "host": host_key,
                        "mode": "english_only_specialist_target",
                        "target_capability": target,
                        "selected_capability": "english",
                        "probe_id": row["probe_id"],
                        "prompt_sha256": hashlib.sha256(
                            str(row["prompt"]).encode("utf-8")
                        ).hexdigest(),
                        "output": realized["output"],
                        "output_sha256": hashlib.sha256(
                            strict_utf8(realized["output"])
                        ).hexdigest(),
                        "actions": [],
                    }
                )

        domain_runtime, domain_specs = _domain_runtime(
            root, temporary / "domain-registry", device=device
        )
        for domain in DOMAINS:
            domain_runtime.install(domain_specs[domain]["package"])
        wrong_order = {"python": "chemistry", "chemistry": "civics", "civics": "python"}
        for target, selected in wrong_order.items():
            for position, row in enumerate(domain_records[target]):
                output, actions = _domain_generate(
                    domain_runtime, domain_specs, selected, str(row["prompt"])
                )
                realized = adapter.realize(
                    prompt=str(row["prompt"]),
                    output=output,
                    capability_id=domain_specs[selected]["cake_id"],
                    position=position,
                )
                observations.append(
                    {
                        "host": host_key,
                        "mode": "wrong_specialist_capability",
                        "target_capability": target,
                        "selected_capability": selected,
                        "probe_id": row["probe_id"],
                        "prompt_sha256": hashlib.sha256(
                            str(row["prompt"]).encode("utf-8")
                        ).hexdigest(),
                        "output": realized["output"],
                        "output_sha256": hashlib.sha256(
                            strict_utf8(realized["output"])
                        ).hexdigest(),
                        "actions": actions,
                        "actions_sha256": sha256_bytes(canonical_json_bytes(actions)),
                    }
                )
        for position, row in enumerate(english_records[:100]):
            output, actions = _domain_generate(
                domain_runtime,
                domain_specs,
                "python",
                str(row["prompt"]),
                catalog_wrapped=False,
            )
            realized = adapter.realize(
                prompt=str(row["prompt"]),
                output=output,
                capability_id=domain_specs["python"]["cake_id"],
                position=position,
            )
            observations.append(
                {
                    "host": host_key,
                    "mode": "wrong_specialist_on_english",
                    "target_capability": "english",
                    "selected_capability": "python",
                    "probe_id": row["probe_id"],
                    "prompt_sha256": hashlib.sha256(
                        str(row["prompt"]).encode("utf-8")
                    ).hexdigest(),
                    "output": realized["output"],
                    "output_sha256": hashlib.sha256(
                        strict_utf8(realized["output"])
                    ).hexdigest(),
                    "actions": actions,
                    "actions_sha256": sha256_bytes(canonical_json_bytes(actions)),
                }
            )
    output_dir.mkdir(parents=True)
    observations_path = output_dir / "observations.jsonl"
    _write_once(
        observations_path,
        b"".join(canonical_json_bytes(row) for row in observations),
    )
    manifest: dict[str, Any] = {
        "format": "abi-v2-live-capability-isolation/1",
        "host": host_key,
        "device": device,
        "native_host": native_identity,
        "adapter_sha256_before": adapters[host_key]["sha256"],
        "adapter_sha256_after": _sha256(root / adapters[host_key]["path"]),
        "english_archive_sha256": _sha256(english_archive),
        "domain_archive_sha256": {
            domain: _sha256(domain_specs[domain]["package"]) for domain in DOMAINS
        },
        "observations_path": "observations.jsonl",
        "observations_rows": len(observations),
        "observations_sha256": _sha256(observations_path),
        "execution_source_sha256": _sha256(Path(__file__).resolve()),
        "wall_seconds": time.perf_counter() - started,
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
    args = parser.parse_args(argv)
    value = run(
        Path.cwd(),
        host_key=args.host,
        output_dir=Path(args.output_dir),
        snapshot=Path(args.snapshot).resolve() if args.snapshot else None,
        device=args.device,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
