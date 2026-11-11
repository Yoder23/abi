"""R104 protocol and fail-closed raw observation tests; no model calls."""

import copy

import pytest

from experiments.evaluator_control_r104 import core as c
from experiments.evaluator_control_r104.run import route_trace


class Tokenizer:
    eos_token_id = 0

    def __len__(self):
        return 256

    def encode(self, text, add_special_tokens=False):
        return [ord(v) for v in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(v) for v in ids if v)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "system: " + messages[0]["content"] + "\nuser: " + messages[1]["content"] + "\nassistant:"


def fixture():
    ref = {"id": "fixture", "task": "rewriting", "prompt": "NOTEQA is managed by PersonQA on Monday. Approval is pending.",
           "required": ["NOTEQA", "PersonQA", "Monday"]}
    tok = Tokenizer()
    prompt = c.render(ref, "parent", "native", tok)
    tokens = tok.encode(ref["prompt"]) + [0]
    route = c.ROUTE_ORDER.index(ref["task"])
    observed = {"id": ref["id"], "task": ref["task"], "arm": "parent", "interface": "native", "route": ref["task"],
                "prompt": prompt, "prompt_ids": tok.encode(prompt), "token_ids": tokens,
                "output": ref["prompt"], "screen": c.screen(ref, ref["prompt"]), "legacy_screen": c.base.screen(ref, ref["prompt"]), "output_utf8_bytes": len(ref["prompt"].encode()),
                "output_characters": len(ref["prompt"]), "stop_reason": "eos", "wall_seconds": .01,
                "execution_routes": [{"cakes": [route], "adapters": [[route]] * 6}] * len(tokens)}
    return ref, observed, tok


def test_catalog_fixed_disjoint_and_balanced():
    first, second = c.catalog("qualification"), c.catalog("comparison")
    assert first == c.catalog("qualification") and len(first) == len(second) == 48
    assert len({r["prompt"] for r in first + second}) == 96
    assert not {r["code"] for r in first} & {r["code"] for r in second}
    for population in (first, second):
        assert all(sum(r["task"] == t for r in population) == 6 for t in c.TASKS)
        assert all("within 120 words" in r["prompt"] for r in population)
        assert all(v in r["prompt"] for r in population for v in r["required"])
        assert all(v in r["prompt"] for r in population for v in r["plan"]["facts"])
    with pytest.raises(ValueError):
        c.catalog("unregistered")


def test_only_two_registered_interfaces():
    ref, _, tok = fixture()
    assert c.render(ref, "parent", "native", tok) == ref["prompt"] + "\n"
    benchmark = c.base.model_prompt(ref, "distillation")
    assert all(c.render(ref, arm, "benchmark", tok) == benchmark for arm in (*c.ARMS, "control"))
    assert c.render(ref, "control", "native", tok).endswith("assistant:")
    with pytest.raises(ValueError):
        c.render(ref, "parent", "best_of_sweep", tok)
    with pytest.raises(ValueError):
        c.checkpoint("new_training")


def qualification_fixture():
    return [{"id": f"{task}-{i}", "task": task, "screen": {"mechanical_pass": True, "collapse": False}, "stop_reason": "eos"}
            for task in c.TASKS for i in range(6)]


def test_qualification_requires_raw_task_floor_and_coverage():
    rows = qualification_fixture()
    assert c.qualified(rows)
    for index in (0, 6, 12, 18):
        rows[index]["screen"]["mechanical_pass"] = False
    assert c.qualified(rows)
    rows[24]["screen"]["mechanical_pass"] = False
    assert not c.qualified(rows)
    rows = qualification_fixture()
    rows[0]["screen"]["mechanical_pass"] = rows[1]["screen"]["mechanical_pass"] = False
    assert not c.qualified(rows)
    assert not c.qualified(qualification_fixture()[:-1])


@pytest.mark.parametrize("field", ["collapse", "truncation"])
def test_no_collapse_or_truncation_qualification(field):
    rows = qualification_fixture()
    if field == "collapse":
        rows[0]["screen"]["collapse"] = True
    else:
        rows[0]["stop_reason"] = "budget"
    assert not c.qualified(rows)


def test_paired_intervals_and_pair_identity():
    first = [{"id": str(i), "screen": {"mechanical_pass": True}} for i in range(48)]
    second = [{"id": str(i), "screen": {"mechanical_pass": False}} for i in range(48)]
    assert c.paired(first, second)["ci97_5"] == [1, 1]
    assert c.paired(first, first)["ci95"] == [0, 0]
    for invalid in (second[:-1], second[::-1], []):
        with pytest.raises(RuntimeError):
            c.paired(first, invalid)


def test_valid_raw_observation():
    ref, observed, tok = fixture()
    c.validate([observed], [ref], "parent", ("native",), tok)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "text", "tokens", "screen", "prompt", "interface", "stop", "route", "time", "bytes"])
def test_raw_hostile_mutations_fail_closed(mutation):
    ref, original, tok = fixture()
    observed = copy.deepcopy(original)
    if mutation == "missing":
        rows = []
    elif mutation == "duplicate":
        rows = [observed, observed]
    else:
        rows = [observed]
        if mutation == "text":
            observed["output"] = "different"
        elif mutation == "tokens":
            observed["token_ids"][0] = -1
        elif mutation == "screen":
            observed["screen"]["mechanical_pass"] = False
        elif mutation == "prompt":
            observed["prompt_ids"] = []
        elif mutation == "interface":
            observed["interface"] = "benchmark"
        elif mutation == "stop":
            observed["stop_reason"] = "budget"
        elif mutation == "route":
            observed["execution_routes"] = []
        elif mutation == "time":
            observed["wall_seconds"] = float("nan")
        elif mutation == "bytes":
            observed["output_utf8_bytes"] += 1
    with pytest.raises(RuntimeError):
        c.validate(rows, [ref], "parent", ("native",), tok)


def test_physical_trace_records_routes_not_trusted_pass_flags():
    from types import SimpleNamespace

    model = SimpleNamespace(last_cake_calls=(3,), transformer=SimpleNamespace(h=[SimpleNamespace(_abi_last_deep_adapter_routes=(3,)) for _ in range(6)]))
    assert route_trace(model, 3) == {"cakes": [3], "adapters": [[3]] * 6}
    model.transformer.h[0]._abi_last_deep_adapter_routes = (4,)
    with pytest.raises(RuntimeError):
        route_trace(model, 3)


def test_missing_freeze_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUT", tmp_path)
    c.write_once(tmp_path / "freeze.json", {"files": {}})
    with pytest.raises(RuntimeError):
        c.binding()


def test_evaluator_contract_fixtures():
    from experiments.evaluator_control_r104.evaluator_cases import audit

    value = audit()
    assert len(value['cases']) == 155
    assert not value['mismatches']
    assert len({r['id'] for r in value['cases']}) == 155


def test_temporal_repair_does_not_mask_other_assertions():
    safe = 'As soon as the details are confirmed, I will send an update.'
    assert c.base.screen({'prompt': '', 'required': [], 'task': 'rewriting'}, safe)['explicit_approval_contradiction']
    assert not c.approval_evidence(safe)['assertions']
    assert c.approval_evidence(safe + ' The project is approved.')['assertions']
    assert c.approval_evidence('If the details are confirmed and the project is approved, I will update you.')['assertions']


def test_stricter_word_limit_and_sentence_floor():
    row, _, _ = fixture()
    text = row['prompt'] + ' ' + ' '.join(chr(97+i//26)+chr(97+i%26) for i in range(125))
    assert c.base.screen(row, text)['nonempty_bounded']
    assert not c.screen(row, text)['nonempty_bounded']
    row['task'] = 'summarization'
    assert not c.screen(row, 'NOTEQA PersonQA Monday approval pending')['requested_sentence_count']


def test_audit_requires_exact_recomputation(tmp_path, monkeypatch):
    from experiments.evaluator_control_r104.evaluator_cases import audit, require_audit

    monkeypatch.setattr(c, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        require_audit()
    value = audit()
    value['cases'][0]['expected_pass'] = False
    c.write_once(tmp_path / 'evaluator_audit.json', value)
    with pytest.raises(RuntimeError):
        require_audit()


def test_qualification_rejects_duplicate_ids():
    values = qualification_fixture()
    values[0]['id'] = values[1]['id']
    assert not c.qualified(values)
