"""Offline stdlib audit of sealed R116 source logs, not a new inference run."""
import argparse
import collections
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = "results/progress_grounding_r116/"
DECISION = "d42f8c8b71bf8fde4891641ea2374a30289073e046ff15f6eb0a4e2fffdee279"
PACKET = "532829b576c99da7c2cf31e579f8a0bf91e9d168ab47d7bee59b5c663ec58aa1"
H = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(files):
    dec = files[RAW+"decision.json"]
    require(H(dec) == DECISION and H(files[RAW+"packet.json"]) == PACKET, "changed release bindings")
    result = json.loads(dec)
    required = {RAW+k for k in result["evidence"]} | {RAW+"decision.json"}
    require(set(files) == required, "missing/extra research files")
    for k, h in result["evidence"].items():
        require(H(files[RAW+k]) == h, "changed research evidence: "+k)
    a, b = (files[RAW+f"source/{r}/raw.jsonl"] for r in ("initial", "replay"))
    require(a == b, "nonidentical replay")
    first, replay = (json.loads(files[RAW+f"source/{r}/receipt.json"]) for r in ("initial", "replay"))
    require(first["pid"] != replay["pid"] and first["raw"] == replay["raw"] == H(a), "replay receipt")
    inputs = json.loads(files[RAW+"packet.json"])["cases"]
    rows = [json.loads(s) for s in a.splitlines()]
    require(len(rows) == len(inputs) == len(set(x["body"] for x in inputs)) == 288, "population")
    by = collections.defaultdict(dict)
    errors = []
    withheld = false_support = missed = top_agrees = 0
    for p, r in zip(inputs, rows):
        require(p["id"] == r["id"] and p["body"] in r["prompt"] and r["ids"], "conditioning")
        # Independent label derivation from the actual rendered notes and the claim subject.
        notes, claim = p["body"].split("Notes:\n")[1].split("\nClaim: ")
        target = next(x.strip(".") for x in claim.split() if len(x.strip(".")) == 12
                      and all(c in "0123456789ABCDEF" for c in x.strip(".")))
        actual = ("has begun", "is working", "is done", "is complete", "are back at work", "restarted",
                  "reached the destination", "in hand", "finished examining", "took place",
                  "released its notice", "gone out", "was credited", "has the funds", "crew began", "clocked in")
        sentences = [line for line in notes.splitlines() if line.startswith("Project "+target+":")]
        require(len(sentences) == int(p["world"] != 2), "world not bound to subject")
        yes = bool(sentences and any(x in sentences[0] for x in actual))
        require(yes == (p["gold"] == "Yes"), "construction oracle disagrees")
        hi, lo = r["yes"], r["no"]
        require(all(type(x) is float and math.isfinite(x) for x in (hi, lo)) and hi != lo, "invalid logits or tie")
        pred = hi > lo
        if not (pred == yes):
            errors.append(p["id"])
        top_agrees += r["top_id"] == (r["yes_id"] if pred else r["no_id"])
        if p["world"] == 2:
            withheld += pred
        else:
            by[p["family"]].setdefault(p["pair"], []).append(pred == yes)
            false_support += pred and not yes
            missed += yes and not pred
    require(len(by) == 8 and all(len(g) == 6 and all(len(v) == 4 for v in g.values()) for g in by.values()), "factorial cells")
    per = {family: sum(all(v) for v in clusters.values()) for family, clusters in by.items()}
    qualified = sum(per.values()) >= 44 and min(per.values()) >= 5 and not withheld
    sealed = result["scores"]["source"]
    require((per, errors, false_support, withheld, qualified) == (sealed["per_family"], sealed["errors"],
            sealed["foil_false_yes"], sealed["withheld_incorrect"], sealed["qualified_diagnostic_only"]), "scoring mismatch")
    require(not qualified and result["calls"] == 576 and set(result["scores"]) == {"source"}
            and result["verdict"] == "SOURCE_PROBE_NOT_QUALIFIED", "wrong endpoint")
    return {"complete_clusters": sum(per.values()), "per_family": per, "false_support": false_support,
            "missed_actual": missed, "withheld_incorrect": withheld, "unrestricted_agrees": top_agrees,
            "verdict": result["verdict"], "raw_sha256": H(a)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--build", type=Path)
    args = parser.parse_args()
    if args.archive:
        with zipfile.ZipFile(args.archive) as z:
            names = z.namelist()
            require(len(names) == len(set(names)) and sum(i.file_size for i in z.infolist()) < 16*2**20,
                    "invalid archive")
            files = {k: z.read(k) for k in names}
    else:
        files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in (ROOT/RAW).rglob("*") if p.is_file()}
    print(json.dumps(audit(files), sort_keys=True))
    if args.build:
        require(not args.archive, "build only from local sealed logs")
        with zipfile.ZipFile(args.build, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for name in sorted(files):
                z.writestr(name, files[name])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, StopIteration) as error:
        sys.exit(f"R116 evidence rejected: {error!r}")
