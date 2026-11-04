"""Frozen R98 catalog, normalization, mechanical screens and evidence helpers."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/plan_english_r98"
PARENT = ROOT / "results/role_invariant_choice_r76/candidate_v1"
HOST = ROOT.parent / "layercake_release"
PARENT_SHA = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
HOST_COMMIT = "a24458818dc4489b4b977c49bc6ea5b8d089ef24"
SOURCE = "Qwen/Qwen2-7B-Instruct"
REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
TASKS = (
    "summarization",
    "rewriting",
    "email_drafting",
    "conversation",
    "format_control",
    "tone_control",
    "clarification",
    "abstention",
)
ARMS = ("abi_plan", "distillation", "lora")
SEED, STEPS, BATCH, CONTEXT, MAX_NEW = 98001, 384, 2, 768, 384


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode()


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append(stream, value) -> None:
    stream.write(canonical(value).decode("ascii"))
    stream.flush()


def catalog(split: str) -> list[dict]:
    if split not in {"train", "development"}:
        raise ValueError("No implicit test split is authorized")
    rng = random.Random(SEED + (0 if split == "train" else 1))
    count = 24 if split == "train" else 6
    items = []
    activities = (
        "prepare a welcome booklet",
        "organize a reading session",
        "repair a wooden bench",
        "arrange a music rehearsal",
        "review a draft invitation",
        "assemble a display",
        "move a collection of drawings",
        "schedule a practice session",
    )
    for index in range(count):
        for task_index, task in enumerate(TASKS):
            code = f"NOTE{('T' if split == 'train' else 'V')}{index:03d}{task_index}"
            owner = f"Person{('T' if split == 'train' else 'V')}{index:03d}{task_index}A"
            recipient = owner[:-1] + "B"
            day = rng.choice(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"))
            place = rng.choice(
                ("the small hall", "the upstairs room", "the garden", "the side office")
            )
            activity = rng.choice(activities)
            facts = [
                f"The project identifier is {code}.",
                f"{owner} is coordinating the project.",
                f"The task is to {activity}.",
                f"The planned day is {day}.",
                f"The location is {place}.",
                f"{recipient} needs an update.",
                "The arrangements are provisional and approval is still pending.",
            ]
            if index % 2:
                facts += ["The previous draft is obsolete; use the current notes."]
            required = [code, owner, day]
            task_requests = {
                "summarization": "Summarize the current project in two or three connected sentences. Include its identifier, coordinator and planned day, and say approval is pending.",
                "rewriting": "Rewrite these notes as a clear short paragraph. Include the identifier, coordinator, day, and provisional approval status.",
                "email_drafting": f"Draft a short polite email to {recipient} with a greeting and sign-off. Include the identifier, coordinator, day, and pending approval.",
                "conversation": "A colleague asks: 'What is happening with this project, and is it confirmed?' Answer conversationally. Include the identifier, coordinator and day; explain that approval is pending.",
                "format_control": "Write exactly three bullet points starting with '- '. Point one: identifier and coordinator. Point two: planned day. Point three: approval is pending.",
                "tone_control": "Give a warm, reassuring but accurate update in two or three sentences. Include the identifier, coordinator, planned day and pending approval. Do not promise confirmation.",
                "clarification": "The notes do not give a start time. Ask one concise question requesting that missing start time, and include the project identifier. Do not invent a time.",
                "abstention": "What is the exact approved budget? The notes do not supply a budget. State that it cannot be determined from the supplied notes and include the project identifier. Do not invent an amount.",
            }
            if task in {"clarification", "abstention"}:
                required = [code]
            rng.shuffle(facts)
            prompt = (
                "Use only the following fictional supplied notes.\n"
                + "\n".join("- " + fact for fact in facts)
                + "\n\n"
                + task_requests[task]
            )
            # Learner-facing plans have only supplied content and the user request.
            plan = {"facts": facts, "request": task_requests[task]}
            items.append(
                {
                    "id": f"r98-{split}-{task}-{index}",
                    "split": split,
                    "task": task,
                    "prompt": prompt,
                    "plan": plan,
                    "required": required,
                    "code": code,
                }
            )
    rng.shuffle(items)
    return items


def model_prompt(row: dict, arm: str) -> str:
    # The task label is provided to all arms, including the unchanged parent.
    return f"Task: {row['task']}\n{row['prompt']}\n\n" + (
        "Plan:\n" if arm == "abi_plan" else "Response:\n"
    )


def target(row: dict, answer: str, arm: str) -> str:
    if arm == "abi_plan":
        return (
            json.dumps(row["plan"], ensure_ascii=True, separators=(",", ":"))
            + "\nResponse:\n"
            + answer
        )
    return answer


def visible_response(raw: str, arm: str) -> tuple[str, bool]:
    if arm != "abi_plan":
        return raw.strip(), True
    if "\nResponse:\n" not in raw:
        return "", False
    plan, answer = raw.split("\nResponse:\n", 1)
    try:
        parsed = json.loads(plan)
        valid = isinstance(parsed, dict) and set(parsed) == {"facts", "request"}
        valid = valid and isinstance(parsed["facts"], list) and isinstance(parsed["request"], str)
    except (ValueError, TypeError):
        valid = False
    return answer.strip(), bool(valid)


def screen(row: dict, answer: str) -> dict:
    """Mechanical content/format proxy only; deliberately not a fluency judge."""
    words = re.findall(r"\b[\w']+\b", answer.lower())
    grams = [tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    collapse = any(grams.count(g) >= 4 for g in set(grams))
    supplied_ids = set(re.findall(r"\b(?:NOTE|Person)[A-Za-z0-9]+\b", row["prompt"]))
    output_ids = set(re.findall(r"\b(?:NOTE|Person)[A-Za-z0-9]+\b", answer))
    stripped = re.sub(r"\b(?:NOTE|Person)[A-Za-z0-9]+\b", "", answer)
    extra_numbers = set(re.findall(r"\b\d+(?:[.:]\d+)?\b", stripped)) - set(
        re.findall(r"\b\d+\b", row["prompt"])
    )
    coverage = all(
        re.search(r"\b" + re.escape(item) + r"\b", answer, re.I) for item in row["required"]
    )
    task = row["task"]
    if task == "clarification":
        form = answer.count("?") == 1 and bool(re.search(r"\b(time|when)\b", answer, re.I))
    elif task == "abstention":
        form = bool(
            re.search(r"\b(not|cannot|can't|unable|unspecified|unknown|no)\b", answer, re.I)
        ) and bool(re.search(r"\b(budget|amount|cost)\b", answer, re.I))
    else:
        form = bool(
            re.search(
                r"\b(pending|provisional|unconfirmed|awaiting)\b|not (?:yet )?(?:approved|confirmed)",
                answer,
                re.I,
            )
        )
        if task == "format_control":
            lines = [line for line in answer.splitlines() if line.strip()]
            form = form and len(lines) == 3 and all(line.startswith("- ") for line in lines)
        if task == "email_drafting":
            form = (
                form
                and bool(re.search(r"\b(dear|hi|hello)\b", answer, re.I))
                and bool(re.search(r"\b(regards|best|sincerely|thanks|thank)\b", answer, re.I))
            )
    # Never label this as semantic entailment or proof of zero hallucinations.
    contradiction = bool(
        re.search(
            r"(?<!not )(?<!not yet )\b(?:is|are|was|were|has been|have been) (?:fully )?(?:approved|confirmed)\b",
            answer,
            re.I,
        )
    )
    result = {
        "required_literal_coverage": bool(coverage),
        "surface_form": bool(form),
        "unknown_identifiers": sorted(output_ids - supplied_ids),
        "extra_numbers": sorted(extra_numbers),
        "explicit_approval_contradiction": contradiction,
        "collapse": collapse,
        "nonempty_bounded": 3 <= len(words) <= 180,
    }
    result["mechanical_pass"] = bool(
        coverage
        and form
        and not result["unknown_identifiers"]
        and not extra_numbers
        and not collapse
        and not contradiction
        and result["nonempty_bounded"]
    )
    return result


def aggregate(observations: list[dict], field: str = "screen") -> dict:
    return {
        "rows": len(observations),
        "passing": sum(row[field]["mechanical_pass"] for row in observations),
        "collapses": sum(row[field]["collapse"] for row in observations),
        "per_task": {
            task: {
                "rows": sum(r["task"] == task for r in observations),
                "passing": sum(
                    r["task"] == task and r[field]["mechanical_pass"] for r in observations
                ),
            }
            for task in TASKS
        },
    }


def source_gate(observations: list[dict]) -> bool:
    summary = aggregate(observations)
    return (
        len(observations) == 240
        and summary["passing"] >= 216
        and summary["collapses"] == 0
        and all(x["passing"] >= 0.75 * x["rows"] for x in summary["per_task"].values())
        and not any(r["truncated"] for r in observations)
    )


def verify_binding() -> dict:
    manifest = read(OUT / "freeze.json")
    for relative, digest in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"R98 frozen input changed or missing: {relative}")
    if sha(PARENT / "model.safetensors") != PARENT_SHA:
        raise RuntimeError("R98 parent state changed")
    return manifest
