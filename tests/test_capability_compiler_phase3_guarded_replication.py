from abi.capability_compiler_phase3_guarded_replication import evaluate_replications


def test_replication_gate_requires_two_identical_passing_hosts():
    protocol = {"candidate": {"checkpoint_sha256": "c"}, "reference": {"outputs_sha256": "o"}}
    row = {"passed": True, "gates": {"quality": True}, "checkpoint_sha256": "c", "observations": 1400, "teacher_present_at_inference": False, "final_test_accessed": False}
    assert all(evaluate_replications(protocol, [row, row], ["o", "o"]).values())
    assert not evaluate_replications(protocol, [row], ["o"])["exactly_two_fresh_replications"]
