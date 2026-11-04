import json

import pytest

from experiments.plan_english_r98 import core as c


def example(task="summarization"):
    return next(row for row in c.catalog("development") if row["task"] == task)


def answer(row):
    code, owner, day = row["required"]
    return f"For {code}, {owner} is coordinating the project for {day}. Approval is pending."


def test_split_and_proposition_isolation():
    train, dev = c.catalog("train"), c.catalog("development")
    assert len(train) == 192 and len(dev) == 48
    assert len({r["id"] for r in train + dev}) == 240
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in dev}
    assert not {r["code"] for r in train} & {r["code"] for r in dev}
    for row in train + dev:
        assert all(fact in row["prompt"] for fact in row["plan"]["facts"])
        assert row["plan"]["request"] in row["prompt"]
        assert set(row["plan"]) == {"facts", "request"}


def test_controls_receive_same_prompt_information():
    row = example()
    for arm in c.ARMS:
        assert row["prompt"] in c.model_prompt(row, arm)
        assert "required" not in c.model_prompt(row, arm)
    assert c.model_prompt(row, "distillation") == c.model_prompt(row, "lora")


def test_plan_is_generated_not_supplied_at_inference():
    row = example()
    assert c.model_prompt(row, "abi_plan").endswith("Plan:\n")
    assert '"facts":' not in c.model_prompt(row, "abi_plan")
    text = c.target(row, answer(row), "abi_plan")
    assert c.visible_response(text, "abi_plan") == (answer(row), True)
    assert c.visible_response(answer(row), "abi_plan") == ("", False)
    assert c.visible_response("not json\nResponse:\n" + answer(row), "abi_plan")[1] is False


@pytest.mark.parametrize("mutation", ["missing", "unknown_id", "amount", "repeat", "contradiction"])
def test_machine_screen_rejects_known_failures(mutation):
    row = example()
    text = answer(row)
    assert c.screen(row, text)["mechanical_pass"]
    if mutation == "missing":
        text = text.replace(row["code"], "that project")
    elif mutation == "unknown_id":
        text += " Ask PersonZ9999A."
    elif mutation == "amount":
        text += " The cost is 9876."
    elif mutation == "repeat":
        text += " Please follow the supplied notes." * 8
    else:
        text += " The project is approved."
    assert not c.screen(row, text)["mechanical_pass"]


def test_format_is_not_just_keyword_presence():
    row = example("format_control")
    assert not c.screen(row, answer(row))["mechanical_pass"]
    code, owner, day = row["required"]
    assert c.screen(row, f"- {code}: {owner}\n- Planned for {day}\n- Approval is pending")[
        "mechanical_pass"
    ]


def test_evidence_write_is_immutable(tmp_path):
    path = tmp_path / "evidence.json"
    c.write_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        c.write_once(path, {"value": 2})
    assert c.read(path) == {"value": 1}


def test_binding_fails_closed_missing_changed_and_parent(monkeypatch, tmp_path):
    root = tmp_path
    out = root / "results"
    parent = root / "parent"
    parent.mkdir()
    (parent / "model.safetensors").write_bytes(b"parent")
    (root / "code.py").write_bytes(b"code")
    monkeypatch.setattr(c, "ROOT", root)
    monkeypatch.setattr(c, "OUT", out)
    monkeypatch.setattr(c, "PARENT", parent)
    monkeypatch.setattr(c, "PARENT_SHA", c.sha(parent / "model.safetensors"))
    c.write_once(out / "freeze.json", {"files": {"code.py": c.sha(root / "code.py")}})
    assert c.verify_binding()["files"]
    (root / "code.py").write_bytes(b"altered")
    with pytest.raises(RuntimeError, match="changed or missing"):
        c.verify_binding()
    (root / "code.py").unlink()
    with pytest.raises(RuntimeError, match="changed or missing"):
        c.verify_binding()


def test_fluency_is_never_inferred_by_mechanical_screen():
    assert "fluency" not in c.screen(example(), answer(example()))
    assert json.loads(json.dumps(c.catalog("train"))) == c.catalog("train")


def test_existing_lora_starts_function_identical():
    import torch
    from transformers.pytorch_utils import Conv1D

    from abi.layercake_host import LoRAConv1D

    torch.manual_seed(98)
    base = Conv1D(12, 8)
    x = torch.randn(2, 3, 8)
    expected = base(x).detach()
    wrapper = LoRAConv1D(base, rank=4, alpha=8)
    assert torch.equal(expected, wrapper(x).detach())
    assert not any(p.requires_grad for p in wrapper.base.parameters())
