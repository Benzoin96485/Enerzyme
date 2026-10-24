"""End-to-end smoke test: Datahub(neighbor_list='cutoff') -> batch collation ->
an *unmodified* existing model (PhysNetCore) with EwaldElectrostaticEnergy swapped
in for ElectrostaticEnergy via YAML/layer_params only.

This is the key regression guard for the whole feature: DistanceLayer already had
an unwired ``offsets`` input, so once the datahub/batch pipeline supplies
``offsets``/``cell``/``pbc``, every architecture built on the standard
DistanceLayer -> RangeSeparationLayer pre-sequence should become periodic-boundary
correct with *zero* changes to that architecture's own core.py.
"""
import pickle

import numpy as np
import torch

from enerzyme.data.datahub import SingleDataHub
from enerzyme.tasks.batch import _decorate_batch_input
from enerzyme.models.ff import build_model
from enerzyme.models.physnet.core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS


def _write_pickle(path, records):
    keys = set()
    for r in records:
        keys.update(r.keys())
    data = {k: [r.get(k) for r in records] for k in keys}
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _ewald_physnet_layer_params(cutoff_real, k_cutoff):
    layer_params = []
    for layer in DEFAULT_LAYER_PARAMS:
        if layer["name"] == "ElectrostaticEnergy":
            layer_params.append({
                "name": "EwaldElectrostaticEnergy",
                "params": {"cutoff_real": cutoff_real, "k_cutoff": k_cutoff},
            })
        else:
            layer_params.append(layer)
    return layer_params


def test_pbc_physnet_ewald_forward_finite(tmp_path):
    torch.manual_seed(0)
    a = 4.0
    cutoff_sr = 3.0

    records = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        Ra = rng.uniform(0.3, a - 0.3, size=(5, 3))
        Za = np.array([1, 1, 1, 6, 6])
        records.append({
            "Ra": Ra, "Za": Za, "N": 5,
            "cell": np.eye(3) * a, "pbc": np.array([1, 1, 1]),
            "Q": 0,
        })
    data_path = tmp_path / "periodic_toy.pkl"
    _write_pickle(data_path, records)

    hub = SingleDataHub(
        dump_dir=str(tmp_path),
        data_format="pickle",
        data_path=str(data_path),
        features={"Ra": None, "Za": None, "N": None, "cell": None, "pbc": None, "Q": None},
        neighbor_list="cutoff",
        neighbor_list_cutoff=cutoff_sr,
        preload=False,
    )
    features = hub.features
    batch = [(features.loc(i), None, str(i)) for i in range(3)]
    dtype = torch.float64
    batch_features, _ = _decorate_batch_input(
        batch, dtype=dtype, device="cpu", otf_graph=False
    )

    build_params = dict(DEFAULT_BUILD_PARAMS)
    build_params.update({"dim_embedding": 8, "num_rbf": 8, "cutoff_sr": cutoff_sr})
    layer_params = _ewald_physnet_layer_params(cutoff_real=cutoff_sr, k_cutoff=6.0)
    for layer in layer_params:
        if layer["name"] == "Core":
            layer["params"] = {**layer.get("params", {}), "num_blocks": 1}

    model = build_model(architecture="PhysNet", layer_params=layer_params, build_params=build_params, verbose=0)
    model = model.to(dtype=dtype)
    output = model(batch_features)

    assert torch.isfinite(output["E"]).all()
    assert torch.isfinite(output["Fa"]).all()
    assert torch.isfinite(output["Qa"]).all()
    # 3 independent periodic graphs -> 3 energies.
    assert output["E"].shape[0] == 3
