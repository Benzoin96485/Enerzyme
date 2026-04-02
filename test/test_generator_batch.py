"""Tests for flow Generator batch fields (``flow_t``, ``Q_flow_a``, ``S_flow_a``)."""

import numpy as np
import torch

from enerzyme.tasks.batch import _decorate_batch_input, _decorate_pyg_batch_input


def _sample_batch():
    f0 = {
        "N": 2,
        "Za": np.array([6, 6], dtype=np.int64),
        "Ra": np.zeros((2, 3), dtype=np.float64),
        "Q_init_a": np.array([1.0, 1.0], dtype=np.float64),
        "S_init_a": np.array([0.5, 0.5], dtype=np.float64),
    }
    f1 = {
        "N": 2,
        "Za": np.array([1, 1], dtype=np.int64),
        "Ra": np.zeros((2, 3), dtype=np.float64),
        "Q_init_a": np.array([2.0, 2.0], dtype=np.float64),
        "S_init_a": np.array([1.0, 1.0], dtype=np.float64),
    }
    t0 = {"Qa": np.array([3.0, 3.0], dtype=np.float64), "Sa": np.array([2.5, 2.5], dtype=np.float64)}
    t1 = {"Qa": np.array([4.0, 4.0], dtype=np.float64), "Sa": np.array([3.5, 3.5], dtype=np.float64)}
    return [(f0, t0, "a"), (f1, t1, "b")]


def test_generator_disabled_no_flow_fields():
    batch = _sample_batch()
    bf, bt = _decorate_batch_input(
        batch, torch.float64, torch.device("cpu"), True, generator_config=None
    )
    assert "flow_t" not in bf
    assert "Q_flow_a" not in bf


def test_generator_dict_interpolation():
    torch.manual_seed(0)
    batch = _sample_batch()
    gen = {"enabled": True}
    bf, bt = _decorate_batch_input(
        batch, torch.float64, torch.device("cpu"), True, generator_config=gen
    )
    assert bf["flow_t"].shape == (2,)
    assert bf["Q_flow_a"].shape == (4,)
    assert bf["S_flow_a"].shape == (4,)
    t = bf["flow_t"]
    torch.testing.assert_close(
        bf["Q_flow_a"],
        (1.0 - t[[0, 0, 1, 1]]) * bf["Q_init_a"] + t[[0, 0, 1, 1]] * bt["Qa"],
    )
    torch.testing.assert_close(
        bf["S_flow_a"],
        (1.0 - t[[0, 0, 1, 1]]) * bf["S_init_a"] + t[[0, 0, 1, 1]] * bt["Sa"],
    )


def test_generator_pyg_same_as_dict_logic():
    torch.manual_seed(1)
    batch = _sample_batch()
    gen = {"enabled": True}
    bf_pyg, bt_pyg = _decorate_pyg_batch_input(
        batch, torch.float64, torch.device("cpu"), True, generator_config=gen
    )
    torch.manual_seed(1)
    bf_d, bt_d = _decorate_batch_input(
        batch, torch.float64, torch.device("cpu"), True, generator_config=gen
    )
    torch.testing.assert_close(bf_pyg.flow_t, bf_d["flow_t"])
    torch.testing.assert_close(bf_pyg.Q_flow_a, bf_d["Q_flow_a"])
    torch.testing.assert_close(bf_pyg.S_flow_a, bf_d["S_flow_a"])
