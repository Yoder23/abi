"""Held-out parent-relative general-English validation for an ABI host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .layercake_host import (
    LayerCakeHostError,
    _canonical_json_bytes,
    _sha256_file,
    load_host_model,
)
from .layercake_host_preservation import _batch, _load_general_rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evaluate_general_preservation(
    *,
    curriculum_path: str | Path,
    layercake_root: str | Path,
    parent_path: str | Path,
    canonical_abi_path: str | Path,
    host_path: str | Path,
    output_path: str | Path,
    device_name: str = "cuda",
    batch_size: int = 2,
    max_tokens: int = 192,
) -> dict[str, Any]:
    """Require held-out next-token agreement with the sealed LayerCake parent."""

    curriculum_path = Path(curriculum_path).resolve()
    layercake_root = Path(layercake_root).resolve()
    parent_path = Path(parent_path).resolve()
    canonical_abi_path = Path(canonical_abi_path).resolve()
    host_path = Path(host_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostError(
            f"general validation evidence is immutable: {output_path}"
        )
    if batch_size <= 0 or max_tokens <= 0:
        raise LayerCakeHostError(
            "general validation sizes must be positive"
        )
    rows = _load_general_rows(
        curriculum_path, split="instruction_validation"
    )
    if len(rows) != 420:
        raise LayerCakeHostError(
            "locked general instruction-validation depth changed"
        )
    candidate, tokenizer, manifest, device = load_host_model(
        layercake_root=layercake_root,
        parent_path=parent_path,
        canonical_abi_path=canonical_abi_path,
        host_path=host_path,
        device_name=device_name,
    )
    parent, _, parent_manifest, _ = load_host_model(
        layercake_root=layercake_root,
        parent_path=parent_path,
        canonical_abi_path=canonical_abi_path,
        host_path=None,
        device_name=device_name,
    )
    if manifest is None or parent_manifest is not None:
        raise LayerCakeHostError("general validation host boundary failed")
    candidate.eval()
    parent.eval()
    observations = []
    total_compared = 0
    total_agreed = 0
    candidate_correct = 0
    parent_correct = 0
    route_equal_rows = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            selected = rows[offset : offset + batch_size]
            ids, attention, prompt_lengths = _batch(
                tokenizer,
                selected,
                max_tokens=max_tokens,
                device=device,
            )
            candidate_result = candidate(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                use_cache=False,
            )
            parent_result = parent(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                use_cache=False,
            )
            candidate_top = candidate_result["logits"][
                :, :-1
            ].argmax(dim=-1)
            parent_top = parent_result["logits"][
                :, :-1
            ].argmax(dim=-1)
            targets = ids[:, 1:]
            positions = torch.arange(
                targets.shape[1], device=device
            )[None, :] + 1
            response_mask = (
                positions >= prompt_lengths[:, None]
            ) & attention[:, 1:].bool()
            for index, row in enumerate(selected):
                mask = response_mask[index]
                compared = int(mask.sum().item())
                agreed = int(
                    (
                        candidate_top[index][mask]
                        == parent_top[index][mask]
                    ).sum().item()
                )
                candidate_hits = int(
                    (
                        candidate_top[index][mask]
                        == targets[index][mask]
                    ).sum().item()
                )
                parent_hits = int(
                    (
                        parent_top[index][mask]
                        == targets[index][mask]
                    ).sum().item()
                )
                route_equal = (
                    int(candidate_result["task_routes"][index].item())
                    == int(parent_result["task_routes"][index].item())
                )
                total_compared += compared
                total_agreed += agreed
                candidate_correct += candidate_hits
                parent_correct += parent_hits
                route_equal_rows += int(route_equal)
                observations.append(
                    {
                        "row_id": row["id"],
                        "prompt_sha256": row["prompt_sha256"],
                        "response_sha256": row["response_sha256"],
                        "compared_response_tokens": compared,
                        "parent_top1_agreed_tokens": agreed,
                        "parent_top1_agreement": (
                            agreed / max(1, compared)
                        ),
                        "candidate_target_top1_tokens": candidate_hits,
                        "parent_target_top1_tokens": parent_hits,
                        "route_equal": route_equal,
                    }
                )
            if len(observations) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "evaluated": len(observations),
                            "total": len(rows),
                        }
                    ),
                    flush=True,
                )
    agreement = total_agreed / max(1, total_compared)
    checks = {
        "locked_420_row_depth": len(observations) == 420,
        "parent_top1_agreement_at_least_095": agreement >= 0.95,
        "validation_only": True,
        "final_test_unopened": True,
    }
    evidence = {
        "format": "abi-layercake-general-preservation-validation/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "split": "instruction_validation",
        "final_test_accessed": False,
        "curriculum_sha256": _sha256_file(curriculum_path),
        "host_manifest_sha256": manifest["manifest_sha256"],
        "parent_checkpoint_sha256": manifest["parent_layercake"][
            "checkpoint_sha256"
        ],
        "device": str(device),
        "observation_count": len(observations),
        "compared_response_tokens": total_compared,
        "parent_top1_agreed_tokens": total_agreed,
        "parent_top1_agreement": agreement,
        "candidate_target_top1_accuracy": (
            candidate_correct / max(1, total_compared)
        ),
        "parent_target_top1_accuracy": (
            parent_correct / max(1, total_compared)
        ),
        "route_agreement": route_equal_rows / len(observations),
        "checks": checks,
        "wall_seconds": time.perf_counter() - started,
        "observations": observations,
        "claim_boundary": (
            "This is held-out teacher-forced agreement with the sealed "
            "LayerCake parent. It is not autonomous-generation or final-test "
            "evidence."
        ),
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--canonical-abi", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=192)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_general_preservation(
        curriculum_path=args.curriculum,
        layercake_root=args.layercake_root,
        parent_path=args.parent,
        canonical_abi_path=args.canonical_abi,
        host_path=args.host,
        output_path=args.output,
        device_name=args.device,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "evidence_sha256",
                    "observation_count",
                    "compared_response_tokens",
                    "parent_top1_agreement",
                    "candidate_target_top1_accuracy",
                    "parent_target_top1_accuracy",
                    "route_agreement",
                    "checks",
                    "wall_seconds",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
