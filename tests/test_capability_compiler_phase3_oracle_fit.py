import json
from pathlib import Path

from abi.capability_compiler_phase3_oracle_fit import _examples


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_examples_are_development_only_and_balanced():
    protocol = {
        "development_catalog": "catalogs/capability_compiler_phase1_frozen_v1.json",
        "teacher_reference": "results/abi_capability_compiler_phase2/teacher/T0/development_outputs.jsonl",
        "capability_routes": {"0": ["grammar", "coherence", "fluent_realization"], "1": ["prompt_grounding", "instruction_following"], "2": ["conversation", "clarification", "abstention"], "3": ["supplied_text_summarization", "rewriting"], "4": ["email_drafting_from_notes", "tone_control", "format_control"], "5": ["fact_free_reasoning"]},
        "training": {"max_tokens": 1024},
    }
    from abi.capability_compiler_phase3_shared_output import load_protocol
    from abi.capability_compiler_phase3_sequence_bridge import _load_parent
    v11, _ = load_protocol(ROOT, ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_SHARED_OUTPUT_PROTOCOL_V11.json")
    _, tokenizer, _ = _load_parent(ROOT, v11, __import__("torch").device("cpu"))
    rows = _examples(ROOT, protocol, tokenizer)
    assert len(rows) == 1400
    assert all(row["record_id"].startswith("phase1-validation-") for row in rows)
