"""Physical pivot-root wrapper for the R28 quorum compiler."""

from __future__ import annotations

import hashlib, json, os, shutil, subprocess, tempfile, uuid
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file
from experiments.preexisting_representation_r15b.public_qualification import canonical_json_bytes


class R28IsolationError(RuntimeError): pass
def _write(path, value):
    if path.exists(): raise R28IsolationError(f"immutable output exists: {path}")
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
def _wsl(path):
    value = path.resolve(); drive = value.drive.rstrip(":").casefold()
    if len(drive) != 1: raise R28IsolationError(f"cannot map into WSL: {value}")
    return f"/mnt/{drive}{value.as_posix().split(':', 1)[1]}"
def build_capsule(root, bundle, capsule):
    if capsule.exists(): raise R28IsolationError("R28 capsule exists")
    capsule.mkdir(parents=True)
    for name, source in {"isolated_worker.py": root / "experiments/robust_labeling_r28/isolated_worker.py", "source_bundle.json": bundle}.items():
        if not source.is_file(): raise R28IsolationError(f"missing R28 capsule input: {source}")
        shutil.copyfile(source, capsule / name)
    spec = {"format": "abi-r28-generic-quorum-compiler/1", "views_per_fact": 3, "quorum": 2, "label_choices": 0, "oracle_fields": 0, "accepted_label_aliases": 0, "fact_ids": 0, "success_ids": 0}; spec["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(spec)).hexdigest(); _write(capsule / "spec.json", spec)
    files = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(capsule.iterdir(), key=lambda item: item.name)]
    manifest = {"format": "abi-r28-isolated-quorum-capsule/1", "files": files, "reveal_files_included": 0, "oracle_fields_included": 0, "accepted_label_aliases_included": 0, "fact_ids_included": 0, "success_ids_included": 0, "network": False}; manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(); _write(capsule / "manifest.json", manifest); return manifest
def run_isolated(root, bundle, destination, distribution="Ubuntu"):
    root, destination = root.resolve(), destination.resolve()
    if os.name != "nt" or destination.exists(): raise R28IsolationError("registered Windows/WSL destination contract changed")
    with tempfile.TemporaryDirectory(prefix="abi-r28-stage-") as raw:
        capsule = Path(raw) / "capsule"; manifest = build_capsule(root, bundle.resolve(), capsule); linux_capsule = f"/tmp/abi-r28-extraction-{uuid.uuid4().hex}"; linux_runner = f"/tmp/abi-r28-runner-{uuid.uuid4().hex}.sh"; sandbox_root = f"/tmp/abi-r28-root-{uuid.uuid4().hex}"
        command = "set -euo pipefail; " + f"cp -aL '{_wsl(capsule)}' '{linux_capsule}'; cp -L '{_wsl(root / 'experiments/robust_labeling_r28/extraction_pivot_runner.sh')}' '{linux_runner}'; chmod 500 '{linux_runner}'; ABI_CAPSULE_PATH='{linux_capsule}' ABI_SANDBOX_ROOT='{sandbox_root}' unshare --mount --pid --fork --ipc --uts --net --propagation private '{linux_runner}'; mkdir -p '{_wsl(destination)}'; cp -a '{linux_capsule}/output/.' '{_wsl(destination)}/'; cp -a '{linux_capsule}/manifest.json' '{_wsl(destination)}/manifest.json'"
        completed = subprocess.run(["wsl.exe", "-d", distribution, "-u", "root", "--", "bash", "-lc", command], cwd=root, capture_output=True, text=True, check=False)
        if completed.returncode != 0: raise R28IsolationError(f"physical R28 extraction failed ({completed.returncode}): {completed.stderr[-4000:]}")
        result_path = destination / "result.json"
        if not result_path.is_file(): raise R28IsolationError("R28 extraction result missing")
        result = json.loads(result_path.read_text(encoding="utf-8")); launcher = {"format": "abi-r28-isolated-extraction-launcher/1", "distribution": distribution, "worker_exit_code": completed.returncode, "sandbox_policy": "linux-pivot-root-no-network/1", "manifest_evidence_sha256": manifest["evidence_sha256"], "result_sha256": sha256_file(result_path), "mountinfo_sha256": sha256_file(destination / "mountinfo.txt")}; launcher["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(launcher)).hexdigest(); _write(destination / "launcher.json", launcher); return {"result": result, "launcher": launcher}

