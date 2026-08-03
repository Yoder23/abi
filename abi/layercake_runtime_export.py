"""Freeze one acquired LayerCake checkpoint with its measured runtime policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .layercake_host import _canonical_json_bytes, _sha256_file


def _manifest_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def export_runtime_candidate(
    *,
    source_path: str | Path,
    output_path: str | Path,
    lexical_repetition_truncation_threshold: int,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise RuntimeError(f"runtime artifact is immutable: {output}")
    if lexical_repetition_truncation_threshold <= 0:
        raise ValueError("a positive measured lexical threshold is required")
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = source / "model.safetensors"
    if (
        metadata.get("format")
        not in {
            "abi-layercake-full-english-core-acquisition/1",
            "abi-layercake-component-graft/1",
        }
        or metadata.get("checkpoint", {}).get("sha256")
        != _sha256_file(checkpoint)
    ):
        raise RuntimeError("source acquisition artifact identity changed")

    output.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name != "metadata.json":
            shutil.copy2(path, output / path.name)
    copied_checkpoint = output / "model.safetensors"
    if _sha256_file(copied_checkpoint) != _sha256_file(checkpoint):
        raise RuntimeError("runtime export changed the model checkpoint")

    exported = dict(metadata)
    exported["status"] = "FROZEN_RUNTIME_CANDIDATE_NOT_FINAL_CERTIFIED"
    exported["decoding"] = {
        "algorithm": "greedy",
        "no_repeat_ngram_size": 0,
        "allow_prompt_ngrams": False,
        "lexical_repetition_truncation_threshold": int(
            lexical_repetition_truncation_threshold
        ),
        "prompt_identity_mixture": False,
    }
    exported["runtime_export"] = {
        "source_path_at_export": str(source),
        "source_metadata_sha256": _sha256_file(metadata_path),
        "source_manifest_sha256": metadata["manifest_sha256"],
        "checkpoint_byte_identical": True,
        "checkpoint_sha256": _sha256_file(copied_checkpoint),
        "teacher_present_at_inference": False,
        "task_specific_postprocessors": 0,
        "exported_files": sorted(
            path.name for path in output.iterdir() if path.is_file()
        ),
    }
    exported["claim_boundary"] = (
        "This artifact freezes the validation-selected general lexical "
        "repetition termination policy with the byte-identical acquired "
        "LayerCake checkpoint. Final quality, human fluency, and runtime "
        "certification remain pending."
    )
    exported["manifest_sha256"] = _manifest_sha(exported)
    (output / "metadata.json").write_text(
        json.dumps(exported, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return exported


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--lexical-repetition-truncation-threshold",
        type=int,
        required=True,
    )
    args = parser.parse_args(argv)
    metadata = export_runtime_candidate(
        source_path=args.source,
        output_path=args.output,
        lexical_repetition_truncation_threshold=(
            args.lexical_repetition_truncation_threshold
        ),
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "manifest_sha256": metadata["manifest_sha256"],
                "decoding": metadata["decoding"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
