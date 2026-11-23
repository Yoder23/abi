"""R116. Execute from the ABI repository root with python -m ...run."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.trained_grounding_r108 import run as old

from . import design as d

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/progress_grounding_r116"
HERE = Path(__file__).resolve().parent
MODULE = "experiments.progress_grounding_r116.run"
PREREG = "696726356abf21cb9142e7b6adb55f6954b25487"
BASE = "99308a6f231ad26b2f7cde0d6bffa9c2b72bc00b"
HOST = "46a8c27eff00bd09280e630b3479fe6d8671a712"
TAG = "abi-freeze-r116-2026-11-23"
MANIFEST = ROOT / "results/review_2026_11_20/r115_manifest.json"
MH = "4de0dcc3105b9cf2a9d885d489dd0d7cd582125fdbb20c4a8e8ee1c69f1d1f9a"
rawread, sha = old.c.read, old.c.sha


def need(ok, message):
    if not ok:
        raise ValueError(message)


def git(*args, root=ROOT):
    return subprocess.check_output(["git", *args], cwd=root).decode().strip()


def tag():
    return git("rev-parse", TAG + "^{commit}")


def blob(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(dump(value))


def guard():
    need(shutil.disk_usage(ROOT).free >= 10 * 2**30, "disk floor")
    need(sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) < 2**30, "evidence cap")


def folder(arm):
    if arm == "source":
        return old.snapshot()
    if arm == "abi_plan":
        return ROOT / "results/grounded_english_r99_fp32/abi_plan"
    return old.c.checkpoint(arm)


def identities():
    need(sha(MANIFEST) == MH, "changed R115")
    m = rawread(MANIFEST)
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in (
        HERE/"PROTOCOL.md", HERE/"run.py", HERE/"design.py", ROOT/"tests/test_progress_grounding_r116.py")}
    for name in code:
        need(blob(tag(), ROOT/name) == (ROOT/name).read_bytes(), "changed frozen R116 code")
    need(blob(PREREG, HERE/"PROTOCOL.md") == (HERE/"PROTOCOL.md").read_bytes(), "changed preregistration")
    need(git("rev-parse", "HEAD", root=old.c.HOST) == HOST, "wrong host lineage")
    for name, h in m["host_inputs"].items():
        need(sha(ROOT/name) == h, "changed host dependency")
    for name in ("abi/layercake_core_loader.py", "experiments/trained_grounding_r108/core.py",
                 "experiments/trained_grounding_r108/run.py"):
        need(sha(ROOT/name) == m["files"][name]["sha256"], "changed loader or runtime")
    need(old.inventory(folder("source")) == m["source_files"], "changed teacher bytes")
    versions = {n: importlib.metadata.version(n) for n in m["packages"]}
    need(versions == m["packages"], "changed runtime versions")
    weights, provenance = {}, {}
    for arm in d.ARMS[1:]:
        path = folder(arm)
        files = {}
        for n in old.c.FILES:
            p = path/n
            h = sha(p)
            original = blob(BASE, p)
            if original.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                need(original.decode().splitlines()[1:] == [f"oid sha256:{h}", f"size {p.stat().st_size}"], "LFS mismatch")
            else:
                need(hashlib.sha256(original).hexdigest() == h, "Git payload mismatch")
            files[n] = h
        weights[arm] = files
        if arm != "parent":
            p = path/"training_result.json"
            need(blob(BASE, p) == p.read_bytes(), "changed training receipt")
            t = rawread(p)
            need(t["checkpoint_sha256"] == files["model.safetensors"] and
                 t["training_log_sha256"] == sha(path/"training.jsonl"), "training not bound to checkpoint")
            provenance[arm] = {k: t[k] for k in ("sample_order_sha256", "source_rows_sha256", "loss_tokens",
                                                  "trainable_parameters", "successful_steps", "training_rows")}
    for k in ("sample_order_sha256", "source_rows_sha256", "successful_steps", "training_rows"):
        need(len({v[k] for v in provenance.values()}) == 1, "unmatched historical controls")
    return dict(code=code, commit=tag(), prereg=PREREG, host=HOST, parent_manifest=MH,
                source=m["source_files"], recipients=weights, historical_training=provenance, versions=versions)


def packet():
    return {"identities": identities(), "cases": d.cases(tag())}


def check_packet():
    p = OUT/"packet.json"
    need(blob(git("rev-parse", "abi-inputs-r116-2026-11-23^{commit}"), p) == p.read_bytes(), "unpublished inputs")
    need(rawread(p) == packet(), "stale frozen packet")


def tokenizer(arm):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(folder(arm), local_files_only=True, trust_remote_code=False)


def render(row, arm, tok):
    if arm == "source":
        return tok.apply_chat_template([{"role": "user", "content": row["body"]}],
                                       tokenize=False, add_generation_prompt=True)
    return row["body"] + "\nResponse:\n"


def choices(tok):
    opts = [tok.encode(v, add_special_tokens=False) for v in ("Yes", "No")]
    need(all(len(ids) == 1 and tok.decode(ids) == v for ids, v in zip(opts, ("Yes", "No")))
         and opts[0] != opts[1], "ambiguous answer tokens")
    return opts[0][0], opts[1][0]


def worker(arm, rep):
    import psutil
    import torch
    from transformers import AutoModelForCausalLM

    from abi.layercake_core_loader import load_layercake_core
    start = time.monotonic()
    proc = psutil.Process()
    cpu = sum(proc.cpu_times()[:2])
    guard()
    check_packet()
    if arm != "source":
        need(verify_arm("source")["qualified_diagnostic_only"], "source did not qualify")
    run = OUT/arm/rep
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(116001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tok = tokenizer(arm)
    if arm == "source":
        model = AutoModelForCausalLM.from_pretrained(folder(arm), dtype=torch.bfloat16,
                  local_files_only=True, trust_remote_code=False).to("cuda")
    else:
        model, t, _ = load_layercake_core(folder(arm), layercake_root=old.c.HOST, device="cuda")
        need(t.get_vocab() == tok.get_vocab(), "recipient tokenizer mismatch")
    need(all(p.dtype == (torch.bfloat16 if arm == "source" else torch.float32)
             for p in model.parameters()), "precision mismatch")
    model.eval()
    y, n = choices(tok)
    route = torch.tensor([old.c.ROUTES.index("summarization")], device="cuda")
    commit = tag()
    need(commit == rawread(OUT/"launch.json")["commit"], "changed launch")
    with torch.inference_mode(), (run/"raw.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        for p in d.cases(commit):
            guard()
            need(time.monotonic()-start < 900, "worker cap")
            prompt = render(p, arm, tok)
            ids = tok.encode(prompt, add_special_tokens=False)
            need(len(ids) < 1024, "context")
            x = torch.tensor([ids], device="cuda")
            logits = (model(x, use_cache=False).logits if arm == "source" else
                      model(x, task_routes=route, use_cache=False)["logits"])[0, -1].float()
            need(torch.isfinite(logits).all().item(), "nonfinite forward")
            i = int(logits.argmax().item())
            f.write(dump(dict(id=p["id"], prompt=prompt, ids=ids, yes_id=y, no_id=n, top_id=i,
                         width=logits.numel(), yes=logits[y].item(), no=logits[n].item(), top=logits[i].item())))
            f.flush()
    write(run/"receipt.json", dict(pid=os.getpid(), arm=arm, repeat=rep, packet=sha(OUT/"packet.json"),
          raw=sha(run/"raw.jsonl"), seconds=time.monotonic()-start, cpu_seconds=sum(proc.cpu_times()[:2])-cpu,
          peak_cuda=torch.cuda.max_memory_allocated(), device=torch.cuda.get_device_name(),
          parameter_count=sum(p.numel() for p in model.parameters()), calls=288))


def check_rows(rows, pop, arm, tok):
    y, n = choices(tok)
    for r, p in zip(rows, pop):
        prompt = render(p, arm, tok)
        need(r["prompt"] == prompt and r["ids"] == tok.encode(prompt, add_special_tokens=False)
             and (r["yes_id"], r["no_id"]) == (y, n), "changed prompt or tokenizer")
        need(type(r["top_id"]) is int and type(r["width"]) is int and r["width"] >= len(tok)
             and 0 <= r["top_id"] < r["width"] and math.isfinite(r["top"])
             and r["top"] >= max(r["yes"], r["no"])
             and all(r["top_id"] != i or r["top"] == r[k] for i, k in ((y, "yes"), (n, "no"))),
             "inconsistent unrestricted logits")
    return d.summarize(rows, pop)


def verify_arm(arm):
    pop, tok = d.cases(tag()), tokenizer(arm)
    hashes, pids = [], []
    for rep in d.REPEATS:
        cell = OUT/arm/rep
        p = cell/"raw.jsonl"
        rec = rawread(cell/"receipt.json")
        op = rawread(OUT/f"{arm}-{rep}.json")
        hashes.append(sha(p))
        pids.append(rec["pid"])
        need((rec["arm"], rec["repeat"], rec["raw"], rec["packet"], rec["pid"], op["code"]) ==
             (arm, rep, hashes[-1], sha(OUT/"packet.json"), op["pid"], 0)
             and 0 < rec["seconds"] <= op["seconds"] <= 901 and rec["calls"] == 288
             and rec["peak_cuda"] > 0 and rec["cpu_seconds"] > 0 and rec["device"], "invalid execution receipt")
        vals = old.c.rows(p)
        summary = check_rows(vals, pop, arm, tok)
    need(len(set(hashes)) == 1 and len(set(pids)) == 2, "not exact fresh replay")
    return {**summary, "input_tokens_per_worker": sum(len(v["ids"]) for v in vals)}


def outcome():
    check_packet()
    need(not (OUT/"stop.json").exists(), "interrupted")
    launch = rawread(OUT/"launch.json")
    need(launch["commit"] == tag() and launch["packet"] == sha(OUT/"packet.json") and
         launch["public_commit"] == git("rev-parse", "abi-inputs-r116-2026-11-23^{commit}"), "launch mismatch")
    source = verify_arm("source")
    scores = {"source": source}
    contrasts = {}
    if source["qualified_diagnostic_only"]:
        scores.update({arm: verify_arm(arm) for arm in d.ARMS[1:]})
        for arm in ("parent", "distillation", "lora"):
            contrasts[arm] = d.contrast(scores["abi_plan"]["clusters"], scores[arm]["clusters"])
        verdict = "MATCHED_RECIPIENT_DIAGNOSTIC_COMPLETE"
    else:
        need(not any((OUT/a).exists() or any(OUT.glob(a+"-*.json")) for a in d.ARMS[1:]), "recipient after failed source")
        verdict = "SOURCE_PROBE_NOT_QUALIFIED"
    costs = {a: [rawread(OUT/a/r/"receipt.json") for r in d.REPEATS] for a in scores}
    files = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob("*"))
             if p.is_file() and p.name != "decision.json"}
    return dict(verdict=verdict, scores=scores, abi_minus_controls=contrasts, costs=costs, evidence=files,
                calls=len(scores)*576, new_training=0, english_artifact=None, native_host_acceptance="NOT_TESTED")


def execute():
    start = time.monotonic()
    guard()
    check_packet()
    head = git("rev-parse", "HEAD")
    need(git("ls-remote", "origin", "refs/heads/master").split()[0] == head
         and git("rev-parse", "abi-inputs-r116-2026-11-23^{commit}") == head, "publish input packet first")
    write(OUT/"launch.json", dict(commit=tag(), packet=sha(OUT/"packet.json"), public_commit=head))
    try:
        for arm in d.ARMS:
            for rep in d.REPEATS:
                guard()
                t = time.monotonic()
                proc = subprocess.Popen([sys.executable, "-m", MODULE, "worker", arm, rep],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ,
                       "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"})
                try:
                    out, err = proc.communicate(timeout=min(900, 10800-(t-start)))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, err = proc.communicate()
                    err += "\nWorker deadline exceeded"
                write(OUT/f"{arm}-{rep}.json", dict(pid=proc.pid, code=proc.returncode, seconds=time.monotonic()-t,
                                                      stdout=out, stderr=err))
                need(proc.returncode == 0, f"{arm}/{rep} stopped")
                print(arm, rep, "complete", flush=True)
            if not verify_arm(arm)["qualified_diagnostic_only"] and arm == "source":
                break
        result = outcome()
        need(time.monotonic()-start <= 10800, "controller budget")
        write(OUT/"decision.json", result)
        print(result["verdict"])
    except BaseException as e:
        write(OUT/"stop.json", dict(error=repr(e), seconds=time.monotonic()-start))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("materialize", "run", "worker", "verify"))
    parser.add_argument("arm", nargs="?", choices=d.ARMS)
    parser.add_argument("rep", nargs="?", choices=d.REPEATS)
    args = parser.parse_args()
    if args.command == "materialize":
        guard()
        write(OUT/"packet.json", packet())
    elif args.command == "run":
        execute()
    elif args.command == "worker":
        need(args.arm and args.rep, "missing worker identity")
        worker(args.arm, args.rep)
    else:
        need(outcome() == rawread(OUT/"decision.json"), "decision differs from recomputation")
        print("R116 strict recomputation passed")
