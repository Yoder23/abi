"""Finite structural logit mask; no prompts, facts, scores or semantic rules."""
from __future__ import annotations

import hashlib
import json
import struct

START = ('start', 0, 0, 0, 0)
SPACE = b' \t\r\n'
TERMINAL = b'.!?'


def step(state, value):
    """One UTF-8/JSON byte transition; None means inadmissible, never repair."""
    mode, count, remaining, low, high = state
    if remaining:
        if not low <= value <= high:
            return None
        return ('body', count, remaining - 1, 128 if remaining > 1 else 0, 191 if remaining > 1 else 0)
    if mode in ('start', 'open', 'after', 'comma', 'done') and value in SPACE:
        return state
    if mode == 'start':
        return ('open', 0, 0, 0, 0) if value == 91 else None
    if mode in ('open', 'comma'):
        return ('first', count, 0, 0, 0) if value == 34 else None
    if mode == 'after':
        if value == 44 and count < 3:
            return ('comma', count, 0, 0, 0)
        if value == 93 and count in (2, 3):
            return ('done', count, 0, 0, 0)
        return None
    if mode == 'terminal':
        return ('after', count + 1, 0, 0, 0) if value == 34 else None
    if mode == 'escape':
        return ('body', count, 0, 0, 0) if value in (34, 92, 47) else None
    if mode not in ('first', 'body'):
        return None
    if value < 32 or value == 34 or (mode == 'first' and value in (32, *TERMINAL)):
        return None
    if value == 92:
        return ('escape', count, 0, 0, 0)
    if value in TERMINAL:
        return ('terminal', count, 0, 0, 0)
    if value < 128:
        return ('body', count, 0, 0, 0)
    if 194 <= value <= 223:
        return ('body', count, 1, 128, 191)
    if 224 <= value <= 239:
        return ('body', count, 2, 160 if value == 224 else 128, 159 if value == 237 else 191)
    if 240 <= value <= 244:
        return ('body', count, 3, 144 if value == 240 else 128, 143 if value == 244 else 191)
    return None


def advance(state, payload):
    if not payload:
        return None
    for value in payload:
        state = step(state, value)
        if state is None:
            break
    return state


def byte_decoder():
    # Standard reversible ByteLevel alphabet; tokenizer decoder identity is checked.
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = list(values), 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): value for value, char in zip(values, chars)}


class Grammar:
    def __init__(self, token_bytes, eos):
        if type(eos) is not int or not 0 <= eos < len(token_bytes):
            raise ValueError('Invalid EOS identity')
        self.tokens = tuple(token_bytes)
        self.eos = eos
        self._allowed = {}

    @classmethod
    def from_tokenizer(cls, tok):
        config = json.loads(tok.backend_tokenizer.to_str())
        if config['decoder']['type'] != 'ByteLevel':
            raise RuntimeError('Unsupported tokenizer decoder')
        inverse = byte_decoder()
        excluded = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
        payloads = []
        for token_id in range(len(tok)):
            if token_id in excluded:
                payloads.append(None)
                continue
            value = tok.convert_ids_to_tokens(token_id)
            try:
                encoded = bytes(inverse[c] for c in value)
            except (KeyError, TypeError) as error:
                raise RuntimeError('Non-ByteLevel vocabulary token') from error
            if not encoded:
                raise RuntimeError('Empty regular token')
            payloads.append(encoded)
        return cls(payloads, tok.eos_token_id)

    def transition(self, state, token):
        if type(token) is not int or not 0 <= token < len(self.tokens):
            return None
        if token == self.eos:
            return state if state[0] == 'done' else None
        payload = self.tokens[token]
        return advance(state, payload) if payload else None

    def allowed(self, state):
        if state not in self._allowed:
            ids = tuple(i for i in range(len(self.tokens)) if self.transition(state, i) is not None)
            if not ids:
                raise RuntimeError('No admissible token; no forced fallback')
            digest = hashlib.sha256(struct.pack(f'<{len(ids)}I', *ids)).hexdigest()
            self._allowed[state] = (ids, digest)
        return self._allowed[state]

    def observation(self, state, index):
        ids, digest = self.allowed(state)
        return {'step': index, 'state': list(state), 'allowed_count': len(ids), 'allowed_sha256': digest}

    def trace(self, tokens):
        state, trace = START, []
        for index, token in enumerate(tokens):
            trace.append(self.observation(state, index))
            target = self.transition(state, token)
            if target is None or (token == self.eos and index != len(tokens) - 1):
                raise RuntimeError('Selected token violates structural grammar')
            state = target
        return trace


class StructuralMask:
    """Transformers-compatible LogitsProcessor, one greedy sequence only."""
    def __init__(self, grammar, prompt_length):
        self.grammar, self.prompt_length = grammar, prompt_length
        self.state, self.events, self.device_masks = START, [], {}

    def __call__(self, input_ids, scores):
        import torch
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise RuntimeError('Only registered single-sequence decoding')
        position = input_ids.shape[1] - self.prompt_length
        if position != len(self.events):
            raise RuntimeError('Missing/repeated decoder callback')
        if position:
            self.state = self.grammar.transition(self.state, int(input_ids[0, -1].item()))
            if self.state is None:
                raise RuntimeError('Previous generated token violates grammar')
        ids, _ = self.grammar.allowed(self.state)
        key = (self.state, str(scores.device), scores.shape[-1])
        if key not in self.device_masks:
            if scores.shape[-1] < len(self.grammar.tokens):
                raise RuntimeError('Output head smaller than tokenizer')
            mask = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
            mask[list(ids)] = True
            self.device_masks[key] = mask
        mask = self.device_masks[key]
        if not torch.isfinite(scores[0, mask]).any().item():
            raise RuntimeError('No finite admissible logit; never force completion')
        self.events.append(self.grammar.observation(self.state, position))
        return scores.masked_fill(~mask.unsqueeze(0), -float('inf'))
