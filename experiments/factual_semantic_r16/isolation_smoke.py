"""Run a public synthetic smoke test of the R16 physical compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    sha256_file,
    write_json_once,
)

from .isolation import run_wsl_isolated_extraction
from .package import answer, load_package


def _records(subject: str, questions: list[str], candidates: list[str], winner: int) -> list[dict]:
    scores = [-10.0] * len(candidates)
    scores[winner] = 0.0
    return [
        {
            "subject": subject,
            "question": question,
            "view": view,
            "candidates": candidates,
            "scores": scores,
        }
        for view, question in enumerate(questions)
    ]


def run(root: Path, output: Path) -> dict:
    if output.exists():
        raise R14Error(f"immutable R16 smoke output exists: {output}")
    output.mkdir(parents=True)
    chemistry = [str(value) for value in range(1, 13)]
    geography = [
        "Athens",
        "Berlin",
        "Cairo",
        "Hanoi",
        "Lima",
        "Lisbon",
        "Madrid",
        "Oslo",
        "Paris",
        "Rome",
        "Tokyo",
        "Vienna",
    ]
    records = _records(
        "helium",
        [
            "What is the atomic number of helium?",
            "Identify helium's atomic number.",
            "Which atomic number belongs to the element helium?",
        ],
        chemistry,
        chemistry.index("2"),
    ) + _records(
        "Italy",
        [
            "What is the national capital of Italy?",
            "Identify Italy's capital city.",
            "Which city is the capital of Italy?",
        ],
        geography,
        geography.index("Rome"),
    )
    bundle = {"format": "abi-r16-anonymous-source-scores/1", "records": records}
    bundle["evidence_sha256"] = evidence_hash(bundle)
    bundle_path = output / "source_bundle.json"
    write_json_once(bundle_path, bundle)
    replay = run_wsl_isolated_extraction(root, bundle_path, output / "extraction")
    packages = [
        load_package(output / "extraction" / item["path"])
        for item in replay["result"]["packages"]
    ]
    if (
        answer(packages, "State the atomic number assigned to helium.") != "2"
        or answer(packages, "Name the seat-of-government capital of Italy.") != "Rome"
    ):
        raise R14Error("R16 physical smoke package behavior changed")
    receipt = {
        "format": "abi-r16-physical-isolation-smoke/1",
        "verdict": "PASS",
        "result_sha256": sha256_file(output / "extraction/result.json"),
        "launcher_sha256": sha256_file(output / "extraction/launcher.json"),
        "packages": replay["result"]["packages"],
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(run(root, args.output), indent=2))


if __name__ == "__main__":
    main()
