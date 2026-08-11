from abi.capability_compiler_phase3_cpu_runtime_v2 import force_cpu_body


def test_force_cpu_changes_only_chat_options():
    source = {"options": {"temperature": 0}, "model": "m"}
    assert force_cpu_body("http://host/api/chat", source)["options"] == {"temperature": 0, "num_gpu": 0}
    assert force_cpu_body("http://host/api/generate", source) == source
