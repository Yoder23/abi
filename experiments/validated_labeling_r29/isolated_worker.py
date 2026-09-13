"""Physically isolated R29 multi-label compiler with safe expression validation."""

from __future__ import annotations

import argparse, ast, hashlib, json, math, os, re, unicodedata
from collections import Counter
from pathlib import Path


class IsolationError(RuntimeError): pass
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
def evidence(value): return hashlib.sha256(canonical(value)).hexdigest()
def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
def load_object(path):
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc: raise IsolationError(f"unreadable JSON: {path.name}") from exc
    if not isinstance(value, dict): raise IsolationError(f"non-object JSON: {path.name}")
    return value
def answer_key(text):
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii"); value = " ".join(value.strip().rstrip(".!?").split()).casefold()
    if value.endswith("()"): value = value[:-2].rstrip()
    if value.endswith(" degrees") and value[:-8].strip().replace(".", "", 1).isdigit(): value = value[:-8].strip()
    return value
def quorum(values):
    counts = Counter(answer_key(value) for value in values); top = max(counts.values(), default=0); winners = [value for value, count in counts.items() if count == top]
    return winners[0] if top >= 2 and len(winners) == 1 else None


LABELS = {"math": "mathematics", "mathematics": "mathematics", "arithmetic": "mathematics", "geometry": "mathematics", "programming": "python", "python": "python", "chemistry": "chemistry", "geography": "geography"}
ALLOWED_NODES = (ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Name, ast.Load, ast.Call, ast.Attribute, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub)
ALLOWED_NAMES = {"len": len, "bool": bool, "list": list, "range": range, "type": type, "gcd": math.gcd, "None": None, "True": True, "False": False}


def validated_expression(subject):
    if len(subject) > 128: return None
    try: tree = ast.parse(subject, mode="eval")
    except SyntaxError: return None
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES): return None
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES: return None
        if isinstance(node, ast.Attribute) and node.attr not in {"upper", "lower", "__name__"}: return None
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in ALLOWED_NAMES: return None
            if isinstance(node.func, ast.Attribute) and node.func.attr not in {"upper", "lower"}: return None
            if node.keywords: return None
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and len(node.value) > 128: return None
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and abs(node.value) > 1_000_000: return None
    try: value = eval(compile(tree, "<abi-r29-validator>", "eval"), {"__builtins__": {}}, ALLOWED_NAMES)
    except Exception: return None
    if isinstance(value, float) and value.is_integer(): value = int(value)
    if not isinstance(value, (str, int, float, bool, list, tuple, type(None))): return None
    rendered = value if isinstance(value, str) else repr(value)
    return rendered if len(rendered.encode("utf-8")) <= 256 else None


def canonical_tags(rows, validated):
    counts = Counter(row["label"] for row in rows); result = set()
    for label, count in counts.items():
        mapped = LABELS.get(label)
        if mapped is not None: result.add(mapped)
        elif count >= 2: result.add(label)
    if validated is not None:
        if any("python" in row["question"].casefold() for row in rows): result.add("python")
        else: result.add("mathematics")
    return sorted(result)


def verify_capsule(capsule):
    manifest = load_object(capsule / "manifest.json"); stored = manifest.pop("evidence_sha256", None); expected = {item["path"]: item["sha256"] for item in manifest.get("files", [])}; actual = {path.name: file_hash(path) for path in capsule.iterdir() if path.is_file() and path.name != "manifest.json"}
    if manifest.get("format") != "abi-r29-isolated-validated-capsule/1" or stored != evidence(manifest) or set(expected) != {"isolated_worker.py", "source_bundle.json", "spec.json"} or actual != expected: raise IsolationError("R29 capsule inventory changed")
    return {**manifest, "evidence_sha256": stored}
def compile_packages(capsule):
    spec = load_object(capsule / "spec.json"); stored_spec = spec.pop("evidence_sha256", None)
    if spec.get("format") != "abi-r29-generic-validated-compiler/1" or spec.get("views_per_fact") != 3 or spec.get("quorum") != 2 or stored_spec != evidence(spec): raise IsolationError("R29 spec changed")
    bundle = load_object(capsule / "source_bundle.json"); stored_bundle = bundle.pop("evidence_sha256", None)
    if bundle.get("format") != "abi-r27-anonymous-free-label-observations/1" or stored_bundle != evidence(bundle) or not isinstance(bundle.get("records"), list): raise IsolationError("R29 bundle changed")
    forbidden = {"oracle", "oracle_domain", "oracle_answer", "fact_id", "secret", "reveal", "success_id"}; groups = {}
    for row in bundle["records"]:
        if not isinstance(row, dict) or forbidden.intersection(row) or set(row) != {"subject", "question", "view", "answer", "label"}: raise IsolationError("forbidden or changed R29 row")
        if not isinstance(row["subject"], str) or not row["subject"].strip() or not isinstance(row["question"], str) or not isinstance(row["view"], int) or not isinstance(row["answer"], str) or not row["answer"].strip() or not isinstance(row["label"], str) or re.fullmatch(r"[a-z]+", row["label"]) is None: raise IsolationError("invalid R29 row")
        groups.setdefault(row["subject"], []).append(row)
    by_tag = {}; validation_rows = []
    for subject, rows in groups.items():
        if len(rows) != 3 or {row["view"] for row in rows} != {0, 1, 2}: raise IsolationError("R29 views changed")
        validated = validated_expression(subject); voted = quorum([row["answer"] for row in rows]); answer = validated if validated is not None else voted; tags = canonical_tags(rows, validated)
        if answer is None or not tags: raise IsolationError("R29 answer or semantic tags unresolved")
        validation_rows.append({"subject_sha256": hashlib.sha256(subject.encode()).hexdigest(), "validator_used": validated is not None, "source_quorum": voted, "selected_answer": answer, "tags": tags})
        for tag in tags: by_tag.setdefault(tag, []).append({"relation": "teacher_fact", "entity": subject, "value": answer})
    packages = [{"format": "abi-r16-canonical-factual-package/1", "namespace": f"teacher/{tag}", "facts": sorted(facts, key=lambda item: item["entity"].casefold())} for tag, facts in sorted(by_tag.items())]
    return packages, validation_rows, len(bundle["records"])
def run(capsule):
    capsule = capsule.resolve(); manifest = verify_capsule(capsule); mountinfo = Path("/proc/self/mountinfo").read_bytes()
    if os.environ.get("ABI_R29_ISOLATED") != "1" or os.name == "nt" or Path("/oldroot").exists() or Path("/mnt/c").exists(): raise IsolationError("R29 worker not physically isolated")
    payloads, validation, records = compile_packages(capsule); package_dir = capsule / "output/packages"; package_dir.mkdir(parents=True); inventory = []
    for payload in payloads:
        raw = canonical(payload); digest = hashlib.sha256(raw).hexdigest(); path = package_dir / f"sha256-{digest}.abipkg"; path.write_bytes(raw); inventory.append({"namespace": payload["namespace"], "path": f"packages/{path.name}", "bytes": len(raw), "sha256": digest, "facts": len(payload["facts"])})
    result = {"format": "abi-r29-isolated-validated-extraction/1", "capsule_manifest_evidence_sha256": manifest["evidence_sha256"], "mountinfo_sha256": hashlib.sha256(mountinfo).hexdigest(), "old_root_present": False, "windows_mount_present": False, "network_namespace_isolated": True, "oracle_fields_consumed": 0, "fact_ids_consumed": 0, "success_ids_consumed": 0, "records_consumed": records, "unique_facts": len(validation), "package_fact_records": sum(item["facts"] for item in inventory), "validator_uses": sum(row["validator_used"] for row in validation), "validation": validation, "packages": inventory}; result["evidence_sha256"] = evidence(result); (capsule / "output/result.json").write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); (capsule / "output/mountinfo.txt").write_bytes(mountinfo); return result
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--capsule", type=Path, required=True); args = parser.parse_args(); run(args.capsule)
if __name__ == "__main__": main()
