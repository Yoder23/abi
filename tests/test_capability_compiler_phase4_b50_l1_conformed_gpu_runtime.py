from abi.capability_compiler_phase4_b50_l1_conformed_gpu_runtime import (
    FORMAT,
    _conformed_request,
)


class _Tokenizer:
    def __call__(self, value, add_special_tokens=False):
        del add_special_tokens
        return type("Tokens", (), {"input_ids": list(value.encode("utf-8"))})()


def _original(runtime, system, probe):
    del runtime, system
    output = "The answer cannot be known."
    return {
        "probe_id": probe["probe_id"],
        "capability": probe["canonical_capability"],
        "output": output,
        "output_token_ids": [1, 2],
        "retokenized_output_token_ids": [1, 2],
        "output_utf8_bytes": len(output.encode()),
        "output_characters": len(output),
        "authoritative_output_tokens": 2,
        "token_accounting": "completed_response_retokenization",
        "time_to_first_output_seconds": 0.1,
        "total_seconds": 1.0,
        "bytes_per_second": len(output.encode()),
        "characters_per_second": len(output),
        "execution": {"route_correct": True},
    }


def test_runtime_contract_format():
    assert FORMAT == "abi-capability-compiler-phase4-b50-l1-conformed-gpu-runtime/1"


def test_runtime_includes_conformance_and_retokenizes_completed_output():
    audit = {"invocations": 0, "replacements": 0, "seconds": 0.0}
    row = _conformed_request(
        _original,
        {"tokenizer": _Tokenizer()},
        "L1",
        {"probe_id": "p1", "canonical_capability": "abstention"},
        audit,
    )
    assert row["output"] == "The answer is unknown."
    assert row["total_seconds"] >= 1.0
    assert row["authoritative_output_tokens"] == len(
        "The answer is unknown.".encode("utf-8")
    )
    assert row["execution"]["conformance_rule_executed"]
    assert row["execution"]["conformance_replacements"] == 1
    assert audit["invocations"] == 1
    assert audit["replacements"] == 1


def test_runtime_does_not_change_non_abstention_output():
    audit = {"invocations": 0, "replacements": 0, "seconds": 0.0}
    row = _conformed_request(
        _original,
        {"tokenizer": _Tokenizer()},
        "L1",
        {"probe_id": "p2", "canonical_capability": "grammar"},
        audit,
    )
    assert row["output"] == "The answer cannot be known."
    assert audit["replacements"] == 0
