from abi.capability_compiler_phase3_final_certificate_audit import assess, physical_sparse_microcheck


def test_physical_sparse_microcheck():
    result = physical_sparse_microcheck()
    assert result["unselected_expert_cannot_affect_route"]
    assert result["selected_expert_affects_route"]
    assert result["active_experts_per_token"] == 1


def test_assess_keeps_phase2_external_to_machine_gates():
    artifact = {"status": "PASS_INDEPENDENT_HOSTILE_ARTIFACT_VERIFICATION", "artifact": {"verified_records": 1280}, "accounting_reconciled": {"stored_logits": 0, "stored_hidden_activations": 0, "copied_source_parameters": 0}}
    verifier = {"status": "PASS_INDEPENDENT_HOSTILE_RECONSTRUCTION", "A0_passes": 1393, "hostile_mutations_rejected": 5}
    replication = {"status": "PASS_THREE_PAIRED_SEED_ROUTE_ISOLATED_REPLICATION", "replication_passed": True, "hierarchical_A0_minus_control": {"x": {"lower_95": 0.1}}}
    checkpoint = "1649e110338904f69fafc0f5ff110e2c8d99f4f8366eb133442fb4938fa3c390"
    runtime = {"status": "PASS_EXACT_ROUTE_ISOLATED_CORRECTED_FULLY_CPU_RUNTIME_GATE_MATRIX", "candidate": {"checkpoint_sha256": checkpoint, "parent_throughput_retention": 1.0}, "gates": {"complete_corrected_gate_matrix": "19/19 PASS", "candidate_fully_cpu": True, "final_test_not_accessed": True}, "comparisons": {"median_throughput_ratio": 3.0, "paired_throughput_lower_95": 2.5, "median_ttft_ratio": 0.5, "peak_rss_ratio": 0.8}}
    hosts = {"status": "PASS_THREE_HOST_EXACT_ROUTE_ISOLATED_REPRODUCTION", "reference": {"checkpoint_sha256": checkpoint}, "passed": True, "gates": {"host_initializations": 3, "byte_identical_outputs": True, "zero_collapse": True, "final_test_not_accessed": True}, "teacher_present_at_inference": False}
    sparse = {"unselected_expert_cannot_affect_route": True, "selected_expert_affects_route": True, "active_experts_per_token": 1, "active_rank": 16}
    incremental = {"persistent_incremental_state": True, "canonical_interface_unchanged": True}
    machine, prerequisites = assess(artifact, verifier, replication, runtime, hosts, sparse, incremental, False)
    assert all(machine.values())
    assert not prerequisites["phase2_complete"]
