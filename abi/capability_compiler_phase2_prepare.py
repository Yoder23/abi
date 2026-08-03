"""Prepare hash-bound Phase 2 packs and authoritative Phi-3 top-k logits."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psutil
import torch

from .capability_compiler_phase1_certificate import verify_certificate
from .capability_compiler_phase2_common import (
    PHASE1_IR_SHA256,
    Phase2Error,
    canonical_json_bytes,
    load_phase1_records,
    pack_examples,
    pack_manifest,
    sha256_bytes,
    sha256_file,
    tokenize_records,
)


SOURCE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
SOURCE_REVISION = "f39ac1d28e925b323eae81227eaba4464caced4e"
SOURCE_MANIFEST_SHA256 = "3bac528a1825e77dcb35963f5c78946fb14400e1cd832e0db40c2d964360c310"
PACKING_SEED = 104729
PACKING_CONTEXT = 768
TOP_K = 64
TEMPERATURE = 2.0


def _snapshot_path() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=SOURCE_MODEL,
        revision=SOURCE_REVISION,
        local_files_only=True,
    )).resolve()


def _verified_snapshot(root: Path) -> Path:
    """Resolve the frozen source and re-hash every declared source file."""

    snapshot = _snapshot_path()
    protocol_path = root / "ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = protocol["source"]
    if (
        source["model"] != SOURCE_MODEL
        or source["revision"] != SOURCE_REVISION
        or source["source_manifest_sha256"] != SOURCE_MANIFEST_SHA256
        or snapshot.name != SOURCE_REVISION
    ):
        raise Phase2Error("Phase 1 source identity changed")
    for category in ("weight_files", "tokenizer_files"):
        for row in source[category]:
            path = snapshot / row["relative_path"]
            if not path.is_file():
                raise Phase2Error(f"source file is missing: {row['relative_path']}")
            if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                raise Phase2Error(f"source file changed: {row['relative_path']}")
    return snapshot


def _tokenizer(snapshot: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.vocab_size != 32_064:
        raise Phase2Error("source tokenizer vocabulary changed")
    return tokenizer


def prepare_packs(*, root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise Phase2Error(f"immutable pack manifest already exists: {output}")
    verify_certificate(root / "ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json")
    ir = root / "results/abi_capability_compiler_phase1/final/normalized_acquisition_ir_v1.abicir"
    if sha256_file(ir) != PHASE1_IR_SHA256:
        raise Phase2Error("certified IR changed")
    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    records = load_phase1_records(ir)
    examples = tokenize_records(records, tokenizer)
    packs = pack_examples(examples, max_tokens=PACKING_CONTEXT, seed=PACKING_SEED)
    manifest = pack_manifest(packs)
    manifest.update({
        "status": "PASS",
        "phase1_ir_sha256": PHASE1_IR_SHA256,
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "packing_seed": PACKING_SEED,
        "packing_context": PACKING_CONTEXT,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "candidate_training_performed": False,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(manifest))
    return manifest


def reconstruct_packs(*, root: Path, manifest_path: Path) -> tuple[list[Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_phase1_records(root / "results/abi_capability_compiler_phase1/final/normalized_acquisition_ir_v1.abicir")
    tokenizer = _tokenizer(_verified_snapshot(root))
    packs = pack_examples(tokenize_records(records, tokenizer), max_tokens=int(manifest["packing_context"]), seed=int(manifest["packing_seed"]))
    rebuilt = pack_manifest(packs)
    if rebuilt["content_sha256"] != manifest["content_sha256"]:
        raise Phase2Error("pack reconstruction changed")
    return packs, manifest


def extract_topk(*, root: Path, manifest_path: Path, output_dir: Path, summary_path: Path) -> dict[str, Any]:
    if summary_path.exists():
        raise Phase2Error(f"immutable top-k summary already exists: {summary_path}")
    packs, manifest = reconstruct_packs(root=root, manifest_path=manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _verified_snapshot(root)
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    source_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if source_parameter_count != 3_821_079_552:
        raise Phase2Error("source parameter count changed")
    inference_seconds = 0.0
    generated_values = 0
    response_positions = 0
    evidence_rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    for index, pack in enumerate(packs):
        target = output_dir / f"{index:05d}-{pack.pack_id}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as cached:
                if str(cached["pack_id"].item()) != pack.pack_id:
                    raise Phase2Error("stale top-k cache pack identity")
                positions = cached["positions"]
                indices = cached["indices"]
                values = cached["values"]
        else:
            inputs = torch.tensor([pack.input_ids], dtype=torch.long, device="cuda")
            positions_tensor = torch.tensor(pack.response_positions, dtype=torch.long, device="cuda")
            batch_started = time.perf_counter()
            with torch.inference_mode():
                logits = model(inputs).logits[0, positions_tensor]
                values_tensor, indices_tensor = torch.topk(logits, k=TOP_K, dim=-1, sorted=True)
            inference_seconds += time.perf_counter() - batch_started
            positions = positions_tensor.cpu().numpy().astype(np.int32, copy=False)
            indices = indices_tensor.cpu().numpy().astype(np.int32, copy=False)
            values = values_tensor.cpu().numpy().astype(np.float16, copy=False)
            buffer = target.with_suffix(".tmp.npz")
            np.savez(
                buffer,
                pack_id=np.asarray(pack.pack_id),
                positions=positions,
                indices=indices,
                values=values,
            )
            os.replace(buffer, target)
        if indices.shape != values.shape or indices.shape != (len(pack.response_positions), TOP_K):
            raise Phase2Error("top-k cache shape changed")
        response_positions += int(indices.shape[0])
        generated_values += int(indices.size)
        evidence_rows.append({
            "pack_id": pack.pack_id,
            "path": target.relative_to(root).as_posix(),
            "sha256": sha256_file(target),
            "positions": int(indices.shape[0]),
            "topk": int(indices.shape[1]),
            "index_bytes": int(indices.nbytes),
            "value_bytes": int(values.nbytes),
        })
    summary = {
        "format": "abi-capability-compiler-phase2-topk-cache/1",
        "status": "PASS",
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_parameter_count": source_parameter_count,
        "precision": "float16",
        "attention_implementation": "eager",
        "device": torch.cuda.get_device_name(0),
        "topk": TOP_K,
        "temperature": TEMPERATURE,
        "pack_manifest_sha256": sha256_file(manifest_path),
        "pack_content_sha256": manifest["content_sha256"],
        "pack_count": len(packs),
        "response_positions": response_positions,
        "stored_logit_values": generated_values,
        "stored_logit_value_bytes": generated_values * 2,
        "stored_logit_index_bytes": generated_values * 4,
        "source_load_seconds": load_seconds,
        "source_inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_bytes": int(psutil.Process().memory_info().rss),
        "hardware": {
            "machine": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "files": evidence_rows,
        "files_content_sha256": sha256_bytes(canonical_json_bytes(evidence_rows)),
        "candidate_training_performed": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes(canonical_json_bytes(summary))
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-packs")
    prepare.add_argument("--output", required=True)
    topk = sub.add_parser("extract-topk")
    topk.add_argument("--manifest", required=True)
    topk.add_argument("--output-dir", required=True)
    topk.add_argument("--summary", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    if args.command == "prepare-packs":
        result = prepare_packs(root=root, output=Path(args.output).resolve())
    else:
        result = extract_topk(
            root=root,
            manifest_path=Path(args.manifest).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            summary_path=Path(args.summary).resolve(),
        )
    print(json.dumps({key: result[key] for key in result if key in {"status", "pack_count", "record_count", "response_positions", "stored_logit_values"}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
