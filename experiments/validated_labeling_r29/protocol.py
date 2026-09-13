"""R29 selection and package execution helpers."""

from __future__ import annotations

import hashlib, random
from experiments.autonomous_labeling_r27.protocol import answer_key
from experiments.robust_labeling_r28.protocol import answer_packages
from .facts import HIDDEN_FACTS


def selected_facts(secret_hex, commitment, per_domain):
    secret = bytes.fromhex(secret_hex)
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != commitment: raise ValueError("R29 reveal mismatch")
    generator = random.Random(int.from_bytes(hashlib.sha256(b"abi-r29\0" + secret).digest(), "big")); result = []
    for domain in sorted({fact.oracle_domain for fact in HIDDEN_FACTS}): result.extend(generator.sample([fact for fact in HIDDEN_FACTS if fact.oracle_domain == domain], per_domain))
    generator.shuffle(result); return result

