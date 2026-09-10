"""Secret-selected, public-bank lexical replication frames for R18."""

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
    Lexeme("Cora", "Cora and Jalen", "balance", "balances", "balanced", "the crimson ledger"),
    Lexeme("Dev", "Dev and Luma", "polish", "polishes", "polished", "the marble tablet"),
    Lexeme("Evan", "Evan and Suri", "arrange", "arranges", "arranged", "the linen banner"),
    Lexeme("Fara", "Fara and Quin", "review", "reviews", "reviewed", "the bronze diagram"),
    Lexeme("Galen", "Galen and Wren", "deliver", "delivers", "delivered", "the glass vessel"),
    Lexeme("Hana", "Hana and Rafi", "protect", "protects", "protected", "the woolen journal"),
    Lexeme("Isla", "Isla and Vero", "catalog", "catalogs", "cataloged", "the granite marker"),
    Lexeme("Jora", "Jora and Cian", "assemble", "assembles", "assembled", "the wooden lattice"),
    Lexeme("Kato", "Kato and Dena", "examine", "examines", "examined", "the coral pendant"),
    Lexeme("Lior", "Lior and Faye", "gather", "gathers", "gathered", "the paper lantern"),
    Lexeme("Mara", "Mara and Gino", "position", "positions", "positioned", "the velvet cushion"),
    Lexeme("Nell", "Nell and Hugo", "prepare", "prepares", "prepared", "the ceramic bowl"),
    Lexeme("Orin", "Orin and Inez", "record", "records", "recorded", "the lunar pattern"),
    Lexeme("Pema", "Pema and Joss", "secure", "secures", "secured", "the woven basket"),
    Lexeme("Rian", "Rian and Kora", "trace", "traces", "traced", "the golden outline"),
    Lexeme("Sela", "Sela and Luc", "uncover", "uncovers", "uncovered", "the hidden mosaic"),
    Lexeme("Tova", "Tova and Miro", "verify", "verifies", "verified", "the coastal map"),
    Lexeme("Ulan", "Ulan and Nia", "wash", "washes", "washed", "the cotton screen"),
    Lexeme("Vina", "Vina and Olek", "anchor", "anchors", "anchored", "the narrow bridge"),
    Lexeme("Wynn", "Wynn and Pavo", "compare", "compares", "compared", "the satin ribbon"),
)


class R18HiddenFrameError(RuntimeError):
    """Raised when the registered hidden lexical selection is invalid."""


def _order(secret_hex: str, commitment: str, signature: str) -> list[int]:
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError as exc:
        raise R18HiddenFrameError("invalid R18 hidden secret") from exc
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != commitment:
        raise R18HiddenFrameError("R18 hidden commitment mismatch")
    return sorted(
        range(len(LEXEME_BANK)),
        key=lambda index: hashlib.sha256(
            secret + commitment.encode() + signature.encode() + index.to_bytes(2, "big")
        ).digest(),
    )


def heldout_rows(secret_hex: str, commitment: str, split: str) -> list[dict[str, Any]]:
    if split not in {"extraction", "evaluation"}:
        raise R18HiddenFrameError("unknown R18 held-out split")
    offset = 0 if split == "extraction" else 3
    repeats = 3 if split == "extraction" else 2
    rows = []
    for signature_index, signature in enumerate(SIGNATURES):
        order = _order(secret_hex, commitment, signature)
        for repeat in range(repeats):
            lexeme_index = order[offset + repeat]
            frame = build_frame(signature, LEXEME_BANK[lexeme_index])
            frame_with_expected = {**frame, "expected": expected(frame)}
            identity = hashlib.sha256(
                (
                    secret_hex
                    + "|"
                    + signature
                    + "|"
                    + split
                    + "|"
                    + str(repeat)
                    + "|"
                    + str(lexeme_index)
                ).encode()
            ).hexdigest()[:20]
            rows.append(
                {
                    "record_id": f"r18-heldout-{identity}",
                    "split": split,
                    **frame,
                    "prompt": prompt_v2(frame),
                    "expected": expected_v2(frame_with_expected),
                }
            )
    return rows
