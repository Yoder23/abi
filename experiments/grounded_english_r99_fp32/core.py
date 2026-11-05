"""Separate FP32 namespace with unchanged R99 source, targets and gates."""

from pathlib import Path

from experiments.grounded_english_r99 import core as original

ROOT, PARENT, HOST = original.ROOT, original.PARENT, original.HOST
PARENT_SHA, HOST_COMMIT = original.PARENT_SHA, original.HOST_COMMIT
SOURCE, REVISION = original.SOURCE, original.REVISION
TASKS, ARMS, SEED = original.TASKS, original.ARMS, original.SEED
STEPS, BATCH, CONTEXT, MAX_NEW = original.STEPS, original.BATCH, original.CONTEXT, original.MAX_NEW
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/grounded_english_r99_fp32"
previous = original.previous  # C1 raw archive for the unchanged R99 source verifier
sha, canonical, write_once = original.sha, original.canonical, original.write_once
read, rows, append = original.read, original.rows, original.append
model_prompt, target = original.model_prompt, original.target
visible_response, screen = original.visible_response, original.screen
aggregate, source_gate, normalize = original.aggregate, original.source_gate, original.normalize


def verify_binding():
    manifest = read(OUT / "freeze.json")
    if not manifest.get("files"):
        raise RuntimeError("Missing FP32 bindings")
    for relative, digest in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"R99 FP32 input changed or missing: {relative}")
    original.verify_binding()
    return manifest
