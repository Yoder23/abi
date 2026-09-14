"""Fail-closed R97 shape and exact-disjointness verifier."""

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from abi.hf_extraction import load_probe_catalog

CATALOG_SHA256 = "TO_BE_SEALED"

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--catalog",required=True,type=Path); p.add_argument("--search-root",required=True,type=Path); a=p.parse_args(); catalog=a.catalog.resolve()
    if CATALOG_SHA256.startswith("TO_BE") or hashlib.sha256(catalog.read_bytes()).hexdigest()!=CATALOG_SHA256: p.error("R97 catalog unsealed or changed")
    probes=list(load_probe_catalog(catalog)["probes"]); prompts={str(row["prompt"]) for row in probes}; prior=set()
    for path in a.search_root.resolve().rglob("*.json"):
        if path.resolve()==catalog: continue
        try: value=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,UnicodeDecodeError,json.JSONDecodeError): continue
        if isinstance(value,dict) and isinstance(value.get("probes"),list): prior.update(str(row["prompt"]) for row in value["probes"] if isinstance(row,dict) and "prompt" in row)
    family={str(i):0 for i in range(4)}; placement={str(i):0 for i in range(4)}; display={str(i):0 for i in range(3)}
    for index,row in enumerate(probes):
        expected=index//350
        if row["probe_id"]!=f"r97-reasoning-f{expected}-{index%350:04d}": p.error("R97 identity changed")
        family[str(row["structural_family"])]+=1; placement[str(row["candidate_list_placement"])]+=1; display[str(row["destination_display_index"])]+=1
    if len(probes)!=1400 or len(prompts)!=1400 or prompts & prior or set(family.values())!={350} or set(placement.values())!={350} or min(display.values())<450: p.error("R97 shape/disjointness failed")
    print(json.dumps({"format":"abi-r97-catalog-verification/1","verdict":"PASS_R97_CATALOG","catalog_sha256":CATALOG_SHA256,"rows":1400,"unique_prompts":1400,"exact_prior_prompt_overlap":0,"family_counts":family,"placement_counts":placement,"display_counts":display,"candidate_loaded":False,"teacher_loaded":False},indent=2,sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
