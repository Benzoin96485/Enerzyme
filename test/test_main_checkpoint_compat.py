"""``main``-branch checkpoints of the four legacy Cores load and reproduce ``main``'s outputs.

Fixtures come from ``scripts/make_main_fixtures.py`` run on ``main``. Each
holds a trainer-format checkpoint (model, EMA, Adam state), the YAML model
section, fixed inputs, and ``main``'s outputs; one fixture per architecture
also holds a checkpoint trained with ``main``'s Muon setup. Loading goes through
``enerzyme.tasks.trainer._load_state_dict``, the path the trainer, calculator,
server and simulator share.
"""

import copy
import glob
import gzip
import io
import os

import pytest
import torch
from torch_ema import ExponentialMovingAverage

from enerzyme.models.compat.main_branch import FORMAT_KEY, KEY_RULES, convert_state_dict
from enerzyme.models.ff import build_model
from enerzyme.tasks.optimizer import get_optimizer
from enerzyme.tasks.trainer import _load_state_dict

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = sorted(glob.glob(os.path.join(HERE, "fixtures", "main_legacy", "*.pt.gz")))
TARGETS = ("E", "Fa", "Qa", "Ea", "M2", "E_ele_a", "E_disp_a", "nh_loss")
TOL = {torch.float64: 1e-10, torch.float32: 1e-6}


def _load(path):
    with gzip.open(path, "rb") as f:
        return torch.load(io.BytesIO(f.read()), weights_only=False)


def _build(payload):
    return build_model(payload["architecture"], copy.deepcopy(payload["layers"]),
                       copy.deepcopy(payload["build_params"]), verbose=0)


def _cast(batch, dtype):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone().to(dtype) if v.is_floating_point() else v.clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def _run(model, batch, dtype, train=False):
    model = model.to(dtype)
    model.train(train)
    inp = _cast(batch, dtype)
    inp["Ra"].requires_grad_(True)
    out = model(inp)
    return {k: v for k, v in out.items() if k in TARGETS and torch.is_tensor(v)}


def _loss(out):
    return (out["E"] ** 2).sum() + (out["Fa"] ** 2).sum() + (out["Qa"] ** 2).sum()


def _assert_close(out, ref, dtype, label):
    assert set(out) == set(ref), (label, sorted(out), sorted(ref))
    for key, expected in ref.items():
        got = out[key].detach().to(torch.float64)
        expected = expected.to(torch.float64)
        magnitude = float(expected.abs().max()) if expected.numel() else 0.0
        if key in ("E", "Fa", "Qa"):
            # The main compatibility contract uses absolute error in fp64 and
            # normwise relative error in fp32, including small charged outputs.
            scale = 1.0 if dtype == torch.float64 or magnitude == 0 else magnitude
        else:
            scale = max(1.0, magnitude)
        diff = float((got - expected).abs().max()) if expected.numel() else 0.0
        assert diff <= TOL[dtype] * scale, f"{label} {key}: max |diff| {diff:.3e} (scale {scale:.3e})"


@pytest.fixture(scope="module", params=FIXTURES, ids=[os.path.basename(p)[:-6] for p in FIXTURES])
def case(request, tmp_path_factory):
    payload = _load(request.param)
    path = tmp_path_factory.mktemp("ckpt") / f"{payload['name']}.pth"
    torch.save(payload["checkpoint"], path)
    return payload, str(path)


def test_fixtures_present():
    assert len(FIXTURES) >= 4, "main-branch fixtures are missing; run scripts/make_main_fixtures.py"


def test_raw_weights(case):
    payload, path = case
    for dtype, tag in ((torch.float64, "raw_fp64"), (torch.float32, "raw_fp32")):
        model = _build(payload)
        _load_state_dict(model, pretrain_path=path, strict=True)
        _assert_close(_run(model, payload["test_batch"], dtype), payload["reference"][tag], dtype,
                      f"{payload['name']} {tag}")


def test_ema_inference(case):
    payload, path = case
    model = _build(payload).float()
    _load_state_dict(model, pretrain_path=path, inference=True)
    _assert_close(_run(model, payload["test_batch"], torch.float64), payload["reference"]["ema_fp64"],
                  torch.float64, f"{payload['name']} ema")


def test_ema_resume(case):
    payload, path = case
    model = _build(payload).float()
    ema = ExponentialMovingAverage(model.parameters(), 0.9, True)
    _load_state_dict(model, pretrain_path=path, ema=ema)
    ema.copy_to(model.parameters())
    _assert_close(_run(model, payload["test_batch"], torch.float64), payload["reference"]["ema_fp64"],
                  torch.float64, f"{payload['name']} ema resume")


def _step_and_compare(payload, model, optimizer, checkpoint, reference, label, atol=lambda param: 0.0):
    optimizer.zero_grad()
    _loss(_run(model, payload["train_batch"], torch.float32, train=True)).backward()
    optimizer.step()
    _, key_map = convert_state_dict(payload["architecture"].lower(), checkpoint["model_state_dict"])
    params = dict(model.named_parameters())
    assert len(params) == len(reference)
    for old_name, expected in reference.items():
        param = params[key_map.get(old_name, old_name)]
        scale = max(1.0, float(expected.abs().max()))
        diff = float((param.detach() - expected).abs().max())
        bound = TOL[torch.float32] * scale + atol(param)
        assert diff <= bound, f"{payload['name']} {label} {old_name}: {diff:.3e}"


def test_continue_training(case):
    payload, path = case
    model = _build(payload).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-6, amsgrad=True)
    _load_state_dict(model, pretrain_path=path, optimizer=optimizer)
    assert len(optimizer.state) == len(optimizer.param_groups[0]["params"])
    _step_and_compare(payload, model, optimizer, payload["checkpoint"],
                      payload["reference"]["after_step_params"], "adam")


def test_continue_training_muon(case, tmp_path):
    """``main``'s two-group Muon state maps onto the ``head``-based Muon split."""
    payload, _ = case
    if "muon_checkpoint" not in payload:
        pytest.skip("fixture has no Muon trajectory")
    pytest.importorskip("muon")
    path = tmp_path / "muon.pth"
    torch.save(payload["muon_checkpoint"], path)
    model = _build(payload).float()
    optimizer = get_optimizer("Muon", model, {})
    _load_state_dict(model, pretrain_path=str(path), optimizer=optimizer)

    _, key_map = convert_state_dict(payload["architecture"].lower(), payload["muon_checkpoint"]["model_state_dict"])
    params = dict(model.named_parameters())
    saved = payload["muon_checkpoint"]["optimizer_state_dict"]
    assert len(optimizer.state) == len(saved["state"])
    for saved_group, group, names in zip(saved["param_groups"], optimizer.param_groups, payload["muon_param_names"]):
        assert {id(params[key_map.get(n, n)]) for n in names} == {id(p) for p in group["params"]}
        for index, name in zip(saved_group["params"], names):
            restored = optimizer.state[params[key_map.get(name, name)]]
            for key, value in saved["state"][index].items():
                same = torch.equal(restored[key], value) if torch.is_tensor(value) else restored[key] == value
                assert same, (name, key)

    # Muon orthogonalizes hidden-weight updates with five bfloat16 Newton-Schulz
    # iterations, which amplify float32-level gradient noise: on ``main``,
    # 1e-7 relative noise moves spookynet_mtase's Muon weights by 4e-4 = lr / 25.
    # The AdamW group keeps the float32 bound.
    muon = {id(p): group["lr"] / 16 for group in optimizer.param_groups if group["use_muon"] for p in group["params"]}
    _step_and_compare(payload, model, optimizer, payload["muon_checkpoint"],
                      payload["reference"]["muon_after_step_params"], "muon",
                      atol=lambda param: muon.get(id(param), 0.0))


def test_parameter_order_matches_main(case):
    payload, _ = case
    model = _build(payload)
    sd = payload["checkpoint"]["model_state_dict"]
    _, key_map = convert_state_dict(payload["architecture"].lower(), sd)
    names = {name for name, _ in model.named_parameters()}
    assert [key_map.get(k, k) for k in payload["main_param_names"]] == [
        key_map.get(k, k) for k in sd if key_map.get(k, k) in names
        and k == next(o for o in sd if key_map.get(o, o) == key_map.get(k, k))
    ]


def test_saved_checkpoints_skip_conversion(case, tmp_path):
    payload, path = case
    model = _build(payload)
    _load_state_dict(model, pretrain_path=path)
    resaved = tmp_path / "resaved.pth"
    torch.save({"model_state_dict": model.state_dict(), FORMAT_KEY: 2}, resaved)
    fresh = _build(payload)
    _load_state_dict(fresh, pretrain_path=str(resaved), strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[key]), key


def test_every_legacy_architecture_is_covered():
    covered = {_load(p)["architecture"].lower() for p in FIXTURES}
    assert covered >= set(KEY_RULES)
