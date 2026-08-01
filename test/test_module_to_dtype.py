"""Regression tests for Module.to(device=, dtype=) vs Module.type / broken FQN buffer casts.

Replaces the removed _to_device_and_dtype helper: Module.to only casts floating /
complex tensors, preserves integer buffers, and does not create rogue root buffers
from fully-qualified named_buffers() names (the bug that broke SpookyNet extract).
"""
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from enerzyme.models.ff import build_model
from enerzyme.models.spookynet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
from enerzyme.tasks.trainer import _load_state_dict

ROOT = Path(__file__).resolve().parents[1]
L2_CKPT = Path(
    "/home/gridsan/wlluo/multiscale/MLFF/enerzyme-new/spookynet/"
    "L2-fragments/FF02-SpookyNet/model_best.pth"
)

# AL-style SpookyNet stack (NuclearEmbedding at pre_sequence.3 after Core.build).
AL_SPOOKYNET_LAYERS = [
    {"name": "RangeSeparation", "params": {"cutoff_fn": "bump"}},
    {
        "name": "ExponentialBernsteinRBF",
        "params": {
            "exp_weighting": False,
            "init_alpha": 0.944863062918464,
            "learnable_shape": True,
            "no_basis_at_infinity": False,
        },
    },
    {
        "name": "NuclearEmbedding",
        "params": {"use_electron_config": True, "zero_init": False},
    },
    {"name": "ElectronicEmbedding", "params": {"attribute": "charge", "num_residual": 1}},
    {"name": "ElectronicEmbedding", "params": {"attribute": "spin", "num_residual": 1}},
    {
        "name": "Core",
        "params": {
            "dropout_rate": 0.0,
            "num_modules": 6,
            "num_residual_local": 1,
            "num_residual_local_d": 1,
            "num_residual_local_p": 1,
            "num_residual_local_s": 1,
            "num_residual_local_x": 1,
            "num_residual_nonlocal_k": 1,
            "num_residual_nonlocal_q": 1,
            "num_residual_nonlocal_v": 1,
            "num_residual_output": 1,
            "num_residual_post": 1,
            "num_residual_pre": 1,
            "shallow_ensemble_size": 10,
            "use_irreps": True,
        },
    },
    {
        "name": "AtomicAffine",
        "params": {
            "scales": {
                "Ea": {"learnable": True, "values": 1},
                "Qa": {"learnable": True, "values": 1},
            },
            "shifts": {
                "Ea": {"learnable": True, "values": 0},
                "Qa": {"learnable": True, "values": 0},
            },
        },
    },
    {"name": "ChargeConservation"},
    {
        "name": "ElectrostaticEnergy",
        "params": {"dielectric_constant": 10.0, "flavor": "SpookyNet"},
    },
    {"name": "AtomicCharge2Dipole"},
    {"name": "EnergyReduce"},
    {
        "name": "ShallowEnsembleReduce",
        "params": {
            "reduce_mean": ["E", "Qa", "M2", "Q"],
            "train_only": True,
            "var": ["E"],
        },
    },
    {"name": "Force"},
    {
        "name": "ShallowEnsembleReduce",
        "params": {
            "eval_only": True,
            "reduce_mean": ["E", "Qa", "M2", "Q", "Fa"],
            "var": ["E", "Fa"],
        },
    },
]

AL_BUILD_PARAMS = {
    "Bohr_in_R": 0.5291772108,
    "Hartree_in_E": 1,
    "activation_fn": "swish",
    "activation_params": {"learnable": True},
    "cutoff_sr": 5.291772108,
    "dim_embedding": 128,
    "max_Za": 86,
    "num_rbf": 16,
}


class _ToySub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("embedding", torch.zeros(2, 3), persistent=False)
        self.register_buffer("idx", torch.arange(4, dtype=torch.long))
        self.register_parameter("w", nn.Parameter(torch.ones(2, 3)))


class _ToyRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sub = _ToySub()


def _broken_fqn_buffer_cast(module: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Historical buggy pattern from _to_device_and_dtype (do not use in production)."""
    for name, buf in module.named_buffers():
        if buf is not None and buf.is_floating_point():
            module._buffers[name] = buf.to(dtype)
    return module


def _rogue_fqn_buffer_keys(module: nn.Module) -> list[str]:
    return [k for k in module._buffers if "." in k]


# ---------------------------------------------------------------------------
# A. Minimal module contract
# ---------------------------------------------------------------------------


def test_module_to_preserves_int_buffers_and_nonpersistent_embedding():
    m = _ToyRoot()
    m2 = m.to(dtype=torch.float64)
    assert m2.sub.w.dtype == torch.float64
    assert m2.sub.embedding.dtype == torch.float64
    assert m2.sub.idx.dtype == torch.int64
    assert _rogue_fqn_buffer_keys(m2) == []
    assert not any(k.endswith("embedding") for k in m2.state_dict())


def test_module_type_converts_integer_buffers():
    m = _ToyRoot().type(torch.float64)
    assert m.sub.idx.is_floating_point()


def test_broken_fqn_cast_creates_rogue_root_buffers_module_to_does_not():
    buggy = _broken_fqn_buffer_cast(deepcopy(_ToyRoot()), torch.float64)
    assert "sub.embedding" in buggy._buffers

    good = _ToyRoot().to(dtype=torch.float64)
    assert _rogue_fqn_buffer_keys(good) == []
    assert "pre_sequence.3.embedding" not in good.state_dict()


# ---------------------------------------------------------------------------
# B. SpookyNet structure + load
# ---------------------------------------------------------------------------


def _build_al_spookynet() -> nn.Module:
    return build_model(
        "SpookyNet",
        layer_params=AL_SPOOKYNET_LAYERS,
        build_params=AL_BUILD_PARAMS,
        verbose=0,
    )


def test_spookynet_module_to_no_rogue_buffers_or_persistent_embedding():
    model = _build_al_spookynet()
    assert "pre_sequence.3.embedding" not in model.state_dict()
    cast = model.to(dtype=torch.float64)
    assert "pre_sequence.3.embedding" not in cast.state_dict()
    assert _rogue_fqn_buffer_keys(cast) == []
    assert cast.pre_sequence[2].n.dtype == torch.int64
    assert cast.pre_sequence[2].v.dtype == torch.int64


def test_spookynet_module_type_would_cast_rbf_integer_buffers():
    model = _build_al_spookynet().type(torch.float64)
    assert model.pre_sequence[2].n.is_floating_point()
    assert model.pre_sequence[2].v.is_floating_point()


def test_spookynet_roundtrip_strict_load_after_module_to():
    model = _build_al_spookynet()
    sd = model.state_dict()
    cast = model.to(device=torch.device("cpu"), dtype=torch.float32)
    cast.load_state_dict(sd, strict=True)


@pytest.mark.skipif(not L2_CKPT.is_file(), reason="L2 SpookyNet checkpoint not available")
def test_spookynet_l2_ema_load_after_module_to_matches_predict_order():
    model = _build_al_spookynet()
    model = model.to(device=torch.device("cpu"), dtype=torch.float32)
    _load_state_dict(
        model,
        device=torch.device("cpu"),
        pretrain_path=str(L2_CKPT),
        inference=True,
    )
    assert "pre_sequence.3.embedding" not in model.state_dict()
    assert _rogue_fqn_buffer_keys(model) == []


@pytest.mark.skipif(not L2_CKPT.is_file(), reason="L2 SpookyNet checkpoint not available")
def test_spookynet_load_then_to_equals_to_then_load_on_key_params():
    def load_then_to() -> nn.Module:
        m = _build_al_spookynet()
        _load_state_dict(
            m, device=torch.device("cpu"), pretrain_path=str(L2_CKPT), inference=True
        )
        return m.to(device=torch.device("cpu"), dtype=torch.float64)

    def to_then_load() -> nn.Module:
        m = _build_al_spookynet().to(device=torch.device("cpu"), dtype=torch.float64)
        _load_state_dict(
            m, device=torch.device("cpu"), pretrain_path=str(L2_CKPT), inference=True
        )
        return m

    a = load_then_to()
    b = to_then_load()
    # Compare NuclearEmbedding element weights (pre_sequence.3)
    torch.testing.assert_close(
        a.pre_sequence[3].element_embedding,
        b.pre_sequence[3].element_embedding,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        a.pre_sequence[3].config_linear.weight,
        b.pre_sequence[3].config_linear.weight,
        rtol=0,
        atol=0,
    )


def test_default_spookynet_module_to_still_builds():
    """Smoke: default stack also survives Module.to without rogue buffers."""
    model = build_model(
        "SpookyNet",
        layer_params=DEFAULT_LAYER_PARAMS,
        build_params=DEFAULT_BUILD_PARAMS,
        verbose=0,
    )
    cast = model.to(dtype=torch.float32)
    assert _rogue_fqn_buffer_keys(cast) == []


# ---------------------------------------------------------------------------
# C. Source guards — no helper / Module.type(dtype) regressions
# ---------------------------------------------------------------------------


def test_trainer_no_longer_defines_or_calls_to_device_and_dtype():
    text = (ROOT / "enerzyme" / "tasks" / "trainer.py").read_text()
    assert "_to_device_and_dtype" not in text


@pytest.mark.parametrize(
    "relpath",
    [
        "enerzyme/tasks/simulator.py",
        "enerzyme/tasks/server.py",
        "enerzyme/tasks/calculator.py",
    ],
)
def test_task_entrypoints_do_not_chain_module_type_dtype(relpath: str):
    text = (ROOT / relpath).read_text()
    # Module chain: .to(...).type(self.dtype) or .type(dtype)
    assert not re.search(r"\)\.type\(\s*self\.dtype\s*\)", text)
    assert not re.search(r"\)\.type\(\s*dtype\s*\)", text)
