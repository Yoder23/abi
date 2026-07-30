"""Bind sequence-preserving ONNX Runtime session settings to an artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

from .layercake_host_runtime import (
    RUNTIME_FORMAT,
    LayerCakeHostRuntimeError,
    _canonical_sha,
    _sha256_file,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def derive_prepacked_session_artifact(
    *,
    source_artifact: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Copy immutable runtime components and enable ORT prepacking."""

    source_artifact = Path(source_artifact).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LayerCakeHostRuntimeError(
            f"prepacked artifact is immutable: {output_path}"
        )
    metadata = json.loads(
        (source_artifact / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("format") != RUNTIME_FORMAT:
        raise LayerCakeHostRuntimeError(
            "prepacking source is not a native host"
        )
    components = {
        "graph": (
            source_artifact / metadata["runtime"]["graph"],
            metadata["runtime"]["graph_sha256"],
        ),
        "tokenizer": (
            source_artifact / metadata["tokenizer"]["path"],
            metadata["tokenizer"]["sha256"],
        ),
        "symbolic": (
            source_artifact / metadata["symbolic_surface"]["path"],
            metadata["symbolic_surface"]["sha256"],
        ),
    }
    output_vocabulary = metadata["runtime"].get("output_vocabulary")
    if output_vocabulary is not None:
        components["vocabulary"] = (
            source_artifact / output_vocabulary["path"],
            output_vocabulary["sha256"],
        )
    for name, (path, expected) in components.items():
        if _sha256_file(path) != expected:
            raise LayerCakeHostRuntimeError(
                f"prepacking source changed: {name}"
            )

    output_path.mkdir(parents=True, exist_ok=False)
    for path, _ in components.values():
        shutil.copyfile(path, output_path / path.name)
    derived = json.loads(json.dumps(metadata))
    derived["status"] = "EXPORTED_NOT_YET_CERTIFIED"
    derived["runtime"]["session_prepacking_enabled"] = True
    derived["runtime"]["session_prepacking"] = {
        "kind": "onnxruntime_constant_weight_prepacking",
        "source_artifact_metadata_sha256": _sha256_file(
            source_artifact / "metadata.json"
        ),
        "weights_changed": False,
        "graph_changed": False,
        "decoding_changed": False,
    }
    derived["evidence_sha256"] = _canonical_sha(derived)
    _write_json(output_path / "metadata.json", derived)
    return derived


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = derive_prepacked_session_artifact(
        source_artifact=args.source_artifact,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
