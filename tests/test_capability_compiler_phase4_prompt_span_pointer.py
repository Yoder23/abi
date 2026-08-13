import itertools

from abi.capability_compiler_phase4_prompt_span_pointer import extract_segments, render_segments


def test_extracts_only_literal_bracketed_prompt_segments() -> None:
    prompt = "Order these: [B-END] done; [B-START] begin; [B-MIDDLE] work."
    segments = extract_segments(prompt)
    assert segments == ("[B-END] done", "[B-START] begin", "[B-MIDDLE] work")
    assert len(set(itertools.permutations(segments))) == 6


def test_render_preserves_exact_prompt_spans() -> None:
    segments = ("[B-START] begin", "[B-MIDDLE] work", "[B-END] done")
    output = render_segments(segments)
    assert output == "[B-START] begin; [B-MIDDLE] work; [B-END] done."
    assert all(segment in output for segment in segments)
