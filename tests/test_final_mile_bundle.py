from abi.final_mile_bundle import (
    ABI_RUNTIME_COMMIT,
    LAYERCAKE_RUNTIME_COMMIT,
    _instructions,
    _runtime_environment,
)


def test_cleanroom_instructions_expose_exact_command_surface_and_boundaries():
    text = _instructions("abi_capability_compiler.whl")
    for command in ("verify", "cpu", "cuda", "quality", "portability", "report"):
        assert f"abi-reproduce {command}" in text
    assert ABI_RUNTIME_COMMIT in text
    assert LAYERCAKE_RUNTIME_COMMIT in text
    assert "HOST_INDEPENDENCE_FAILED" in text
    assert "runtime-environment.json" in text


def test_runtime_environment_discloses_reproduction_dependencies():
    environment = _runtime_environment()
    assert environment["format"] == "abi-final-mile-runtime-environment/1"
    assert environment["packages"]["torch"]
    assert environment["packages"]["cryptography"]
    assert environment["packages"]["psutil"]
