from abi.capability_compiler_phase3_routed_v16_sequence_conformance_batched import _batch_loss


def test_batched_loss_is_explicitly_callable() -> None:
    assert callable(_batch_loss)
