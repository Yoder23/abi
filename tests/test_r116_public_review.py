import json

import pytest

from scripts import review_r116 as review


def packet():
    root = review.ROOT / review.RAW
    return {p.relative_to(review.ROOT).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_actual_rows_reproduce():
    result = review.audit(packet())
    assert (result["complete_clusters"], result["false_support"], result["missed_actual"]) == (33, 20, 2)


@pytest.mark.parametrize("mutation", ["missing", "extra", "swapped", "forged_logits", "stale_result", "pid"])
def test_corrupted_public_evidence_rejected(mutation):
    data = packet()
    initial = review.RAW + "source/initial/raw.jsonl"
    if mutation == "missing":
        del data[initial]
    elif mutation == "extra":
        data[review.RAW+"lora/initial/raw.jsonl"] = b"{}"
    elif mutation == "swapped":
        lines = data[initial].splitlines(keepends=True)
        lines[:2] = reversed(lines[:2])
        data[initial] = b"".join(lines)
    elif mutation == "forged_logits":
        lines = data[initial].splitlines(keepends=True)
        row = json.loads(lines[0])
        row["yes"], row["no"] = row["no"], row["yes"]
        lines[0] = json.dumps(row).encode() + b"\n"
        data[initial] = b"".join(lines)
    else:
        key = review.RAW+("decision.json" if mutation == "stale_result" else "source/replay/receipt.json")
        value = json.loads(data[key])
        value.update({"verdict": "PASSED"} if mutation == "stale_result" else {"pid": -1})
        data[key] = json.dumps(value).encode()
    with pytest.raises((ValueError, KeyError)):
        review.audit(data)
