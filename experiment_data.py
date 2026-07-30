#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared corpus loaders for the scale-extension ABI experiments."""

from __future__ import annotations

import pathlib

from wikitext_cache import load_wikitext_split


def hf_local_path(model_id: str) -> str:
    """Return the local Hugging Face snapshot path for a cached model id."""
    local_models = pathlib.Path(__file__).parent / "hf_models"
    for name in (model_id.replace("/", "--"), model_id.split("/")[-1]):
        candidate = local_models / name
        if (candidate / "config.json").exists():
            return str(candidate)

    cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    snapshots = model_dir / "snapshots"

    ref_main = model_dir / "refs" / "main"
    if ref_main.exists():
        revision = ref_main.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if candidate.exists():
            return str(candidate)

    if snapshots.exists():
        candidates = sorted(
            [p for p in snapshots.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    return model_id


def load_local_python_text(
    root: pathlib.Path,
    max_tokens: int,
    exclude_globs: tuple[str, ...] = (),
) -> tuple[str, dict]:
    """Load a deterministic local Python corpus from repository source files."""
    max_chars = max_tokens * 4
    parts: list[str] = []
    chars = 0
    skipped = 0
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root)
        if any(rel.match(pattern) for pattern in exclude_globs):
            skipped += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        parts.append(text)
        chars += len(text)
        if chars >= max_chars:
            break

    if not parts:
        raise RuntimeError(f"No Python source files found under {root}")

    return "\n\n".join(parts), {
        "files": len(parts),
        "chars": chars,
        "exclude_globs": list(exclude_globs),
        "skipped": skipped,
    }


def load_wikitext_text_and_sentences(
    split: str = "validation",
    min_chars: int = 20,
) -> tuple[str, list[str], dict]:
    """Load cached WikiText-2 text plus alignment paragraphs."""
    records = [
        r for r in load_wikitext_split("wikitext-2-raw-v1", split)
        if r.get("text", "").strip()
    ]
    text = "\n".join(r["text"] for r in records)
    sentences = [r["text"].strip() for r in records if len(r["text"].strip()) >= min_chars]

    if not sentences:
        raise RuntimeError(f"No WikiText alignment sentences found for split={split!r}")

    return text, sentences, {
        "split": split,
        "records": len(records),
        "sentences": len(sentences),
        "chars": len(text),
    }
