def test_printable_ascii_inventory_is_complete():
    values = {bytes((value,)) for value in range(0x20, 0x7F)}
    assert len(values) == 95
    assert b"%" in values and b"/" in values
    assert all(value.decode("utf-8") for value in values)
