from abi.capability_compiler_phase3_utf8_bpe import fit
def test_bpe_tokens_concatenate_exactly():
 t=fit(["café marketing", "100% / ready"],128)
 for value in ("café marketing","100% / ready"):
  assert "".join(t.encode(value).tokens)==value
