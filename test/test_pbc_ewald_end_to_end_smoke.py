"""DataHub periodic images through both batch formats and the registered PhysNet force stack."""
import copy
import pickle

import numpy as np
import pytest
import torch

from enerzyme.data.datahub import SingleDataHub
from enerzyme.tasks.batch import _decorate_batch_input, _decorate_pyg_batch_input, _to_device
from enerzyme.models.ff import build_model, get_ff_core


def _write_pickle(path, records):
    keys = set()
    for r in records:
        keys.update(r.keys())
    data = {k: [r.get(k) for r in records] for k in keys}
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _ewald_physnet_layer_params(default_layers, cutoff_real, k_cutoff):
    layer_params = []
    for layer in copy.deepcopy(default_layers):
        if layer["name"] == "ElectrostaticEnergy":
            layer_params.append({
                "name": "EwaldElectrostaticEnergy",
                "params": {"cutoff_real": cutoff_real, "k_cutoff": k_cutoff},
            })
        else:
            layer_params.append(layer)
    return layer_params


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pbc_physnet_ewald_forward_finite(tmp_path, device, dtype):
    torch.manual_seed(0)
    a = 4.0
    cutoff_sr = 3.0

    records = []
    for seed in range(2):
        rng = np.random.default_rng(seed)
        Ra = rng.uniform(0.3, a - 0.3, size=(5, 3))
        Za = np.array([1, 1, 1, 6, 6])
        records.append({
            "Ra": Ra, "Za": Za, "N": 5,
            "cell": np.eye(3) * a, "pbc": np.array([1, 1, 1]),
            "Q": 0,
        })
    # Moving one atom to the next copy of the same triclinic cell leaves E/F unchanged.
    for record in records:
        record["cell"][1, 0] = 0.65
    shifted = copy.deepcopy(records[0])
    shifted["Ra"][0] += shifted["cell"][1]
    records.append(shifted)
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
    batch = [(features.loc(i), {}, str(i)) for i in range(3)]
    batch_features, _ = _to_device(_decorate_batch_input(batch, dtype, device, otf_graph=False), device)
    pyg_features, _ = _decorate_pyg_batch_input(batch, dtype, device, otf_graph=False)
    pyg_features = pyg_features.to(device)

    _, defaults, default_layers = get_ff_core("PhysNet")
    build_params = dict(defaults, dim_embedding=8, num_rbf=8, cutoff_sr=cutoff_sr)
    layers = _ewald_physnet_layer_params(default_layers, cutoff_real=cutoff_sr, k_cutoff=6.0)
    for layer in layers:
        if layer["name"] == "Core":
            layer["params"]["num_blocks"] = 1
        elif layer["name"] == "AtomicAffine":
            layer["params"]["shifts"]["Qa"]["values"] = {1: 0.6, 6: -0.8}

    model = build_model("PhysNet", layers, build_params, verbose=0).to(device=device, dtype=dtype)
    output = model(batch_features)
    assert output["E"].shape == (3,)
    assert output["Qa"].abs().max() > 0.1
    assert output["E_ele_a"].abs().max() > 1e-3
    assert output["Fa"].abs().max() > 1e-4
    atol = 1e-5 if dtype == torch.float32 else 1e-10
    for name in ("E", "E_ele_a", "Qa", "Fa"):
        value = output[name]
        assert value.dtype == dtype and value.device.type == device
        torch.testing.assert_close(value.reshape(3, -1)[0], value.reshape(3, -1)[2], atol=atol, rtol=atol)
    (output["E"].square().sum() + output["Fa"].square().sum()).backward()
    charge_shift = next(m.shifts["Qa"] for m in model.post_sequence if hasattr(m, "shifts"))
    assert torch.isfinite(charge_shift.grad).all() and charge_shift.grad.abs().max() > 0

    pyg_output = model(pyg_features)
    for name in ("E", "E_ele_a", "Qa", "Fa"):
        torch.testing.assert_close(pyg_output[name], output[name], atol=atol, rtol=atol)

    if dtype == torch.float64 and device == "cpu":
        # Discrete site/mode membership is determined before the force VJP.
        # Capture the real training stack and verify its mixed derivatives.
        expected = {key: param.grad.clone() for key, param in model.named_parameters() if param.grad is not None}
        model.zero_grad(set_to_none=True)
        compiled = torch.compile(model, backend="eager")(batch_features)
        (compiled["E"].square().sum() + compiled["Fa"].square().sum()).backward()
        for name in ("E", "Fa"):
            torch.testing.assert_close(compiled[name], output[name], atol=atol, rtol=atol)
        gradients = {key: param.grad for key, param in model.named_parameters() if param.grad is not None}
        assert gradients.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(gradients[key], expected[key], atol=atol, rtol=atol)
