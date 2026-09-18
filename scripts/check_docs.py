"""Fail-closed checks for ABI's canonical documentation surface."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOCUMENTS = (
    "README.md",
    "CURRENT_PROJECT_STATUS.md",
    "ACTIVE_MISSION.md",
    "CLAIMS.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/README.md",
    "docs/GETTING_STARTED.md",
    "docs/PROJECT_STATUS.md",
    "docs/architecture.md",
    "docs/repository-layout.md",
    "docs/research-status.md",
    "docs/PEER_REVIEW_READINESS.md",
    "docs/INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md",
    "docs/PHASE2_HUMAN_RATING_HANDOFF_V1.md",
    "docs/PHASE8_EXTERNAL_REPRODUCTION_V1.md",
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def local_link_target(document: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    path_part = unquote(target.split("#", 1)[0])
    if not path_part:
        return None
    return (document.parent / path_part).resolve()


def check() -> list[str]:
    errors: list[str] = []
    canonical_documents: list[Path] = []
    for relative in CANONICAL_DOCUMENTS:
        document = ROOT / relative
        if not document.is_file():
            errors.append(f"missing canonical document: {relative}")
            continue
        canonical_documents.append(document)

    tracked_markdown = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    link_documents = {
        *canonical_documents,
        *(ROOT / relative for relative in tracked_markdown),
    }

    for document in canonical_documents:
        text = document.read_text(encoding="utf-8")
        if not text.startswith("# "):
            errors.append(f"missing level-one heading: {document.relative_to(ROOT)}")

    for document in sorted(link_documents):
        text = document.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = local_link_target(document, match.group(1))
            if target is not None and not target.exists():
                errors.append(
                    f"broken local link in {document.relative_to(ROOT)}: {match.group(1)}"
                )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "docs/README.md",
        "docs/GETTING_STARTED.md",
        "docs/PROJECT_STATUS.md",
        "docs/repository-layout.md",
        "docs/INDEPENDENT_REVIEW_HANDOFF_2026-09-14.md",
    ):
        if required not in readme:
            errors.append(f"README discovery path is missing: {required}")

    quickstart = (ROOT / "docs" / "GETTING_STARTED.md").read_text(encoding="utf-8")
    for required_command in (
        "python -m pip install -e",
        "python -m abi --help",
        "python -m abi status --json",
        "python -m abi self-check",
        "python -m examples.segregate_capabilities",
    ):
        if required_command not in quickstart:
            errors.append(f"getting-started command is missing: {required_command}")

    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"DOCS_ERROR: {error}")
        return 1
    print(
        f"DOCS_OK: {len(CANONICAL_DOCUMENTS)} canonical documents and all tracked Markdown links checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
