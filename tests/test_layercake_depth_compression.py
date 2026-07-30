import pytest

from abi.layercake_depth_compression import parse_selected_layers


def test_depth_compression_layer_selection_is_exact_and_ordered():
    assert parse_selected_layers("0,2,5") == (0, 2, 5)
    assert parse_selected_layers([1, 3, 5]) == (1, 3, 5)
    for invalid in ("", "0,2", "0,2,6", "2,1,5", "0,0,5", "a,2,5"):
        with pytest.raises(ValueError):
            parse_selected_layers(invalid)
