def test_output_only_pruning_definition():
    prompt={b"shared",b"prompt"}; output={b"shared",b"target"}
    assert output-prompt=={b"target"}
