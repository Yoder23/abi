from pathlib import Path


def _text(): return (Path(__file__).parents[1]/"abi"/"capability_compiler_phase3_source_neuron_threshold_audit.py").read_text(encoding="utf-8")


def test_single_neuron_count_is_derived_from_train_contribution_energy():
    text=_text();assert "torch.searchsorted(cumulative" in text;assert "importance *= down.double().square().sum" in text;assert '"neuron_count_sweep_performed":False' in text


def test_threshold_audit_is_read_only_and_uses_selected_source_equations():
    text=_text();assert "selected_gate=" in text;assert "selected_up=" in text;assert "selected_down=" in text;assert "save_file" not in text;assert "torch.optim" not in text
