#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local WikiText cache loader used by the ABI proof scripts."""

import pathlib

from datasets import Dataset, load_dataset


def load_wikitext_split(config: str = "wikitext-2-raw-v1", split: str = "train"):
    """
    Load WikiText from the local Hugging Face Arrow cache when available.

    On this Windows setup, load_dataset(..., offline) can spend minutes resolving
    a cached builder.  Dataset.from_file() loads the already-present Arrow split
    directly and keeps proof reruns reproducible without changing corpus content.
    """
    cache_root = (pathlib.Path.home() / ".cache" / "huggingface" /
                  "datasets" / "wikitext" / config)
    candidates = sorted(
        cache_root.glob(f"**/wikitext-{split}.arrow"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"  [data] using cached {config}/{split}: {candidates[0]}", flush=True)
        return Dataset.from_file(str(candidates[0]))
    return load_dataset("wikitext", config, split=split)
