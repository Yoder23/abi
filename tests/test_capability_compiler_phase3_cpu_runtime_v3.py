import torch
from abi.capability_compiler_phase3_cpu_runtime_v3 import _score_on_model_device
from abi.capability_compiler_phase3_sparse_router import SparseRouter


class TinyTokenizer:
    lexeme_to_id = {"x": 0}
    def split(self, text): return ["x"]


def test_score_follows_router_parameter_device():
    model = SparseRouter(1, 8, 3)
    protocol = {"representation": {"character_hash_buckets": 8, "character_ngram_minimum": 1, "character_ngram_maximum": 1, "hash_seed": 1}}
    assert _score_on_model_device(model, TinyTokenizer(), protocol, ["x"]).device.type == "cpu"
