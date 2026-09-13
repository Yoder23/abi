"""Frozen selection and semantic scoring for R27."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata

from .facts import Fact, HIDDEN_FACTS


def answer_key(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = " ".join(value.strip().rstrip(".!?").split()).casefold()
    if value.endswith("()"):
        value = value[:-2].rstrip()
    if value.endswith(" degrees") and value[:-8].strip().replace(".", "", 1).isdigit():
        value = value[:-8].strip()
    return value


def valid_free_label(text: str) -> bool:
    return re.fullmatch(r"[a-z]+", text) is not None


def selected_facts(secret_hex: str, commitment: str, per_domain: int) -> list[Fact]:
    secret = bytes.fromhex(secret_hex)
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != commitment:
        raise ValueError("R27 reveal does not match its commitment")
    generator = random.Random(int.from_bytes(hashlib.sha256(b"abi-r27\0" + secret).digest(), "big"))
    chosen: list[Fact] = []
    for domain in sorted({fact.oracle_domain for fact in HIDDEN_FACTS}):
        candidates = [fact for fact in HIDDEN_FACTS if fact.oracle_domain == domain]
        if not 0 < per_domain <= len(candidates):
            raise ValueError("R27 selection count changed")
        chosen.extend(generator.sample(candidates, per_domain))
    generator.shuffle(chosen)
    return chosen

