from __future__ import annotations

import hashlib
import json

import pytest

from abi.layercake_core_loader import (
    ABIEnglishCoreConfig,
    CAPABILITY_CAKE_ARCHITECTURE,
    CAPABILITY_CAKE_CANONICAL_ROUTES,
    CAPABILITY_CAKE_ORDER,
    DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
    DEEP_CAPABILITY_ADAPTER_RANK,
    PERSISTENT_CAPABILITY_PREFIX_ARCHITECTURE,
    PERSISTENT_PREFIX_LENGTH,
    PERSISTENT_PREFIX_ROUTER_BUCKETS,
    PERSISTENT_PREFIX_ROUTER_WIDTH,
    TASK_ROUTE_LAYERWISE_CONTROL_ARCHITECTURE,
    TASK_ROUTE_PROMPT_IDENTITY_ARCHITECTURE,
    TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_ARCHITECTURE,
    PROMPT_IDENTITY_RANK,
    _load_symbolic_surface_substrate,
)
from abi.layercake_host import _canonical_json_bytes


def test_versioned_core_config_accepts_only_locked_depths() -> None:
    assert ABIEnglishCoreConfig(layers=3).layers == 3
    assert ABIEnglishCoreConfig(
        layers=6,
        architecture_version=(
            "layercake-shallow-sparse-english/2-six-block-task-cakes"
        ),
    ).layers == 6
    with pytest.raises(ValueError, match="three or six"):
        ABIEnglishCoreConfig(layers=4)


def test_versioned_core_config_preserves_sparse_topology() -> None:
    with pytest.raises(ValueError, match="instruction-cake topology"):
        ABIEnglishCoreConfig(task_cakes=9)
    with pytest.raises(ValueError, match="width or head"):
        ABIEnglishCoreConfig(width=512)


def test_versioned_core_config_accepts_preregistered_rank256_sparse_extension() -> None:
    config = ABIEnglishCoreConfig(
        task_cake_rank=256,
        architecture_version=(
            "layercake-shallow-sparse-english/2-three-block-rank256-task-cakes"
        ),
    )
    assert config.layers == 3
    assert config.task_cakes == 10
    assert config.task_cake_rank == 256
    with pytest.raises(ValueError, match="architecture version"):
        ABIEnglishCoreConfig(task_cake_rank=256)


def test_versioned_core_config_accepts_capability_isolated_rank64_cakes() -> None:
    config = ABIEnglishCoreConfig(
        task_cakes=14,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=(
            CAPABILITY_CAKE_CANONICAL_ROUTES
        ),
        architecture_version=CAPABILITY_CAKE_ARCHITECTURE,
    )
    assert config.task_cakes == 14
    assert config.task_cake_rank == 64
    assert len(config.capability_cake_order) == 14
    with pytest.raises(ValueError, match="capability-cake topology"):
        ABIEnglishCoreConfig(
            task_cakes=14,
            capability_cake_order=CAPABILITY_CAKE_ORDER,
            capability_cake_canonical_routes=(0,) * 14,
            architecture_version=CAPABILITY_CAKE_ARCHITECTURE,
        )


def test_versioned_core_config_accepts_only_locked_persistent_prefix() -> None:
    config = ABIEnglishCoreConfig(
        task_cakes=14,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=CAPABILITY_CAKE_CANONICAL_ROUTES,
        capability_prefix_length=PERSISTENT_PREFIX_LENGTH,
        capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
        capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
        architecture_version=PERSISTENT_CAPABILITY_PREFIX_ARCHITECTURE,
    )
    assert config.capability_prefix_length == 8
    assert config.capability_router_buckets == 4096
    with pytest.raises(ValueError, match="persistent capability-prefix"):
        ABIEnglishCoreConfig(
            task_cakes=14,
            capability_cake_order=CAPABILITY_CAKE_ORDER,
            capability_cake_canonical_routes=(
                CAPABILITY_CAKE_CANONICAL_ROUTES
            ),
            capability_prefix_length=7,
            capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
            capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
            architecture_version=PERSISTENT_CAPABILITY_PREFIX_ARCHITECTURE,
        )


def test_versioned_core_config_accepts_only_locked_deep_adapters() -> None:
    config = ABIEnglishCoreConfig(
        task_cakes=14,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=CAPABILITY_CAKE_CANONICAL_ROUTES,
        capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
        capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
        capability_adapter_rank=DEEP_CAPABILITY_ADAPTER_RANK,
        architecture_version=DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
    )
    assert config.capability_adapter_rank == 32
    with pytest.raises(ValueError, match="deep capability-adapter"):
        ABIEnglishCoreConfig(
            task_cakes=14,
            capability_cake_order=CAPABILITY_CAKE_ORDER,
            capability_cake_canonical_routes=(
                CAPABILITY_CAKE_CANONICAL_ROUTES
            ),
            capability_router_buckets=PERSISTENT_PREFIX_ROUTER_BUCKETS,
            capability_router_width=PERSISTENT_PREFIX_ROUTER_WIDTH,
            capability_adapter_rank=16,
            architecture_version=DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
        )


def test_versioned_core_config_accepts_ten_route_layerwise_control() -> None:
    config = ABIEnglishCoreConfig(
        capability_control_width=768,
        task_route_layerwise_control=True,
        architecture_version=TASK_ROUTE_LAYERWISE_CONTROL_ARCHITECTURE,
    )
    assert config.task_cakes == 10
    assert config.task_route_layerwise_control is True
    assert config.capability_router_buckets == 0
    with pytest.raises(ValueError):
        ABIEnglishCoreConfig(
            capability_control_width=768,
            task_route_layerwise_control=True,
            capability_router_buckets=4096,
            architecture_version=TASK_ROUTE_LAYERWISE_CONTROL_ARCHITECTURE,
        )


def test_versioned_core_config_accepts_only_locked_prompt_carriage() -> None:
    config = ABIEnglishCoreConfig(
        capability_control_width=768,
        task_route_layerwise_control=True,
        prompt_identity_rank=PROMPT_IDENTITY_RANK,
        architecture_version=TASK_ROUTE_PROMPT_IDENTITY_ARCHITECTURE,
    )
    assert config.prompt_identity_rank == 32
    with pytest.raises(ValueError, match="prompt-identity carriage"):
        ABIEnglishCoreConfig(
            capability_control_width=768,
            task_route_layerwise_control=True,
            prompt_identity_rank=16,
            architecture_version=TASK_ROUTE_PROMPT_IDENTITY_ARCHITECTURE,
        )
    with pytest.raises(ValueError, match="prompt-identity carriage"):
        ABIEnglishCoreConfig(
            prompt_identity_rank=PROMPT_IDENTITY_RANK,
            architecture_version=TASK_ROUTE_PROMPT_IDENTITY_ARCHITECTURE,
        )


def test_versioned_core_config_distinguishes_selective_prompt_carriage() -> None:
    config = ABIEnglishCoreConfig(
        capability_control_width=768,
        task_route_layerwise_control=True,
        prompt_identity_rank=PROMPT_IDENTITY_RANK,
        prompt_identity_selective=True,
        architecture_version=(
            TASK_ROUTE_SELECTIVE_PROMPT_IDENTITY_ARCHITECTURE
        ),
    )
    assert config.prompt_identity_selective is True
    with pytest.raises(ValueError, match="selective prompt identity"):
        ABIEnglishCoreConfig(prompt_identity_selective=True)


def test_symbolic_substrate_loader_is_canonical_hash_bound_and_teacher_free(
    tmp_path,
) -> None:
    contract = {
        "handlers": ["natural_email_from_notes"],
        "source_teacher_text_retained": False,
    }
    payload = _canonical_json_bytes(contract)
    (tmp_path / "symbolic_surface.json").write_bytes(payload)
    metadata = {
        "symbolic_surface_substrate": {
            "format": "abi-layercake-symbolic-substrate-graft/1",
            "path": "symbolic_surface.json",
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "handlers": list(contract["handlers"]),
            "maximum_active_handlers_per_sequence": 1,
            "source_neural_parameters_copied": 0,
            "source_task_cakes_copied": 0,
            "source_classifier_parameters_copied": 0,
            "source_teacher_text_retained": False,
            "teacher_present_at_inference": False,
        }
    }
    assert _load_symbolic_surface_substrate(tmp_path, metadata) == contract
    (tmp_path / "symbolic_surface.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="stale or tampered"):
        _load_symbolic_surface_substrate(tmp_path, metadata)
