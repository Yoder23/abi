"""Fresh secret-selected lexical bank for R19 held-out replication."""

from __future__ import annotations

import hashlib
from typing import Any

from experiments.linguistic_realization_r17.frames import (
    SIGNATURES,
    Lexeme,
    build_frame,
    expected,
)
from experiments.linguistic_realization_r17.frames_v2 import expected_v2, prompt_v2

LEXEME_BANK = (
    Lexeme("Adel", "Adel and Brin", "align", "aligns", "aligned", "the quartz panel"),
    Lexeme("Bela", "Bela and Cyra", "brush", "brushes", "brushed", "the indigo canvas"),
    Lexeme("Cato", "Cato and Davi", "close", "closes", "closed", "the iron cabinet"),
    Lexeme("Dara", "Dara and Enzo", "design", "designs", "designed", "the spiral emblem"),
    Lexeme("Eli", "Eli and Freya", "enter", "enters", "entered", "the stone gallery"),
    Lexeme("Fia", "Fia and Garo", "fasten", "fastens", "fastened", "the canvas cover"),
    Lexeme("Hale", "Hale and Inga", "guide", "guides", "guided", "the copper trolley"),
    Lexeme("Ira", "Ira and Juno", "handle", "handles", "handled", "the crystal prism"),
    Lexeme("Jari", "Jari and Kiva", "invite", "invites", "invited", "the visiting scholar"),
    Lexeme("Kela", "Kela and Ludo", "join", "joins", "joined", "the evening circle"),
    Lexeme("Mavi", "Mavi and Nara", "lock", "locks", "locked", "the oak chest"),
    Lexeme("Neri", "Neri and Orla", "mark", "marks", "marked", "the eastern boundary"),
    Lexeme("Omar", "Omar and Peri", "notice", "notices", "noticed", "the faint symbol"),
    Lexeme("Qira", "Qira and Rolo", "open", "opens", "opened", "the velvet pouch"),
    Lexeme("Ravi", "Ravi and Sona", "place", "places", "placed", "the porcelain tile"),
    Lexeme("Sia", "Sia and Taro", "question", "questions", "questioned", "the silent witness"),
    Lexeme("Umi", "Umi and Vali", "rotate", "rotates", "rotated", "the brass dial"),
    Lexeme("Wira", "Wira and Xeno", "sort", "sorts", "sorted", "the colored cards"),
    Lexeme("Yara", "Yara and Zeno", "touch", "touches", "touched", "the frosted window"),
    Lexeme("Zia", "Zia and Arlo", "wrap", "wraps", "wrapped", "the scarlet parcel"),
)


class R19HiddenFrameError(RuntimeError):
    """Raised when R19 secret selection is invalid."""


def _order(secret_hex: str, commitment: str, signature: str) -> list[int]:
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError as exc:
        raise R19HiddenFrameError("invalid R19 hidden secret") from exc
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != commitment:
        raise R19HiddenFrameError("R19 hidden commitment mismatch")
    return sorted(
        range(len(LEXEME_BANK)),
        key=lambda index: hashlib.sha256(
            secret + commitment.encode() + signature.encode() + index.to_bytes(2, "big")
        ).digest(),
    )


def heldout_rows(secret_hex: str, commitment: str, split: str) -> list[dict[str, Any]]:
    if split not in {"extraction", "evaluation"}:
        raise R19HiddenFrameError("unknown R19 held-out split")
    offset = 0 if split == "extraction" else 3
    repeats = 3 if split == "extraction" else 2
    rows = []
    for signature in SIGNATURES:
        order = _order(secret_hex, commitment, signature)
        for repeat in range(repeats):
            lexeme_index = order[offset + repeat]
            frame = build_frame(signature, LEXEME_BANK[lexeme_index])
            with_expected = {**frame, "expected": expected(frame)}
            identity = hashlib.sha256(
                f"{secret_hex}|{signature}|{split}|{repeat}|{lexeme_index}".encode()
            ).hexdigest()[:20]
            rows.append(
                {
                    "record_id": f"r19-heldout-{identity}",
                    "split": split,
                    **frame,
                    "prompt": prompt_v2(frame),
                    "expected": expected_v2(with_expected),
                }
            )
    return rows
