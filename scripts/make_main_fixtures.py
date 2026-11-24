"""Generate golden checkpoints for the four legacy architectures on the ``main`` branch.

Run from a checkout of ``main`` (it must import ``main``'s ``enerzyme``)::

    git worktree add ../enerzyme-main-ref main
    cd ../enerzyme-main-ref
    PYTHONPATH=$PWD python <devel>/scripts/make_main_fixtures.py --out <devel>/test/fixtures/main_legacy

Every variant writes ``<name>.pt`` holding the trainer-format checkpoint
(``model_state_dict``, ``ema_state_dict``, ``optimizer_state_dict``), the
YAML-level config, the exact batched inputs, and reference outputs for the raw
and EMA weights in float64 and float32, plus the parameters after one more
optimizer step on a fixed loss.
"""
import argparse
import copy
import gzip
import io
import os

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from enerzyme.models.ff import build_model

TARGETS = ("E", "Fa", "Qa", "Ea", "M2", "E_ele_a", "E_disp_a", "nh_loss")
UNIT = {"Hartree_in_E": 1, "Bohr_in_R": 0.5291772108}
AFFINE = {
    "name": "AtomicAffine",
    "params": {
        "shifts": {"Ea": {"values": 0, "learnable": True}, "Qa": {"values": 0, "learnable": True}},
        "scales": {"Ea": {"values": 1, "learnable": True}, "Qa": {"values": 1, "learnable": True}},
    },
}
SSP = {"dim_feature": 1, "initial_alpha": 1, "initial_beta": 1, "learnable": False}
SPOOKY_CORE = {
    "num_modules": 2, "num_residual_pre": 1, "num_residual_local_x": 1, "num_residual_local_s": 1,
    "num_residual_local_p": 1, "num_residual_local_d": 1, "num_residual_local": 1,
    "num_residual_nonlocal_q": 1, "num_residual_nonlocal_k": 1, "num_residual_nonlocal_v": 1,
    "num_residual_post": 1, "num_residual_output": 1, "use_irreps": True, "dropout_rate": 0.0,
}
MACE_CORE = {
    "max_ell": 3, "additional_hidden_irreps": "16x1o",
    "interaction_cls_first": "RealAgnosticResidualInteractionBlock",
    "interaction_cls": "RealAgnosticResidualInteractionBlock",
    "correlation": 3, "num_interactions": 2, "avg_num_neighbors": 8.0,
    "MLP_irreps": "4x0e", "radial_MLP": [16, 16, 16], "gate": "silu",
}


def _tail(electrostatic_flavor, d3=False):
    layers = [
        copy.deepcopy(AFFINE),
        {"name": "ChargeConservation"},
        {"name": "AtomicCharge2Dipole"},
        {"name": "ElectrostaticEnergy", "params": {"cutoff_lr": None, "flavor": electrostatic_flavor}},
    ]
    if d3:
        layers.append({"name": "GrimmeD3Energy", "params": {"learnable": True}})
    layers += [{"name": "EnergyReduce"}, {"name": "Force"}]
    return layers


def variants():
    v = {}
    v["schnet_ff07"] = dict(
        architecture="SchNet",
        build_params={"dim_embedding": 16, "num_rbf": 12, "max_Za": 94, "cutoff_sr": 5.0, **UNIT},
        layers=[
            {"name": "RangeSeparation"},
            {"name": "GaussianSmearing", "params": {"cuton": 0.0}},
            {"name": "RandomAtomEmbedding"},
            {"name": "Core", "params": {"num_interactions": 2, "hidden_channels": 16,
                                        "activation_fn": "shifted_softplus", "activation_params": SSP}},
        ] + _tail("PhysNet"),
    )
    v["schnet_default"] = dict(
        architecture="SchNet",
        build_params={"dim_embedding": 16, "num_rbf": 12, "max_Za": 94, "cutoff_sr": 5.0, **UNIT},
        layers=[
            {"name": "RangeSeparation"},
            {"name": "GaussianSmearing", "params": {"no_basis_at_infinity": False, "init_alpha": 1,
                                                    "exp_weighting": False, "learnable_shape": True,
                                                    "init_width_flavor": "PhysNet"}},
            {"name": "RandomAtomEmbedding"},
            {"name": "Core", "params": {"hidden_channels": 16, "num_interactions": 2,
                                        "activation_fn": "shifted_softplus", "activation_params": SSP}},
        ] + _tail("PhysNet", d3=True),
    )
    v["physnet_ff01"] = dict(
        architecture="PhysNet",
        build_params={"dim_embedding": 16, "num_rbf": 12, "max_Za": 94, "cutoff_sr": 10.0,
                      "cutoff_lr": None, "cutoff_fn": "polynomial", **UNIT},
        layers=[
            {"name": "RangeSeparation"},
            {"name": "ExponentialGaussianRBF", "params": {"no_basis_at_infinity": False, "init_alpha": 1,
                                                          "exp_weighting": False, "learnable_shape": True,
                                                          "init_width_flavor": "PhysNet"}},
            {"name": "RandomAtomEmbedding"},
            {"name": "ScalarDenseEmbedding", "params": {
                "embed_field": "Qa_xTB", "activation_fn": "swish", "activation_params": SSP,
                "initial_weight": "orthogonal", "initial_bias": "zero", "use_bias": True}},
            {"name": "GatherAtomEmbedding"},
            {"name": "Core", "params": {"num_blocks": 2, "num_residual_atomic": 2, "num_residual_interaction": 2,
                                        "num_residual_output": 1, "activation_fn": "shifted_softplus",
                                        "activation_params": SSP, "dropout_rate": 0.0}},
        ] + _tail("PhysNet"),
        extra_atomic=("Qa_xTB",),
    )
    v["physnet_mtase"] = dict(
        architecture="PhysNet",
        build_params={"dim_embedding": 16, "num_rbf": 12, "max_Za": 16, "cutoff_sr": 10.0,
                      "cutoff_fn": "polynomial", **UNIT},
        layers=[
            {"name": "RangeSeparation"},
            {"name": "ExponentialGaussianRBF", "params": {"exp_weighting": False, "init_alpha": 1,
                                                          "init_width_flavor": "PhysNet",
                                                          "learnable_shape": True,
                                                          "no_basis_at_infinity": False}},
            {"name": "RandomAtomEmbedding"},
            {"name": "Core", "params": {"activation_fn": "shifted_softplus", "activation_params": SSP,
                                        "dropout_rate": 0.0, "num_blocks": 3, "num_residual_atomic": 1,
                                        "num_residual_interaction": 1, "num_residual_output": 1}},
            copy.deepcopy(AFFINE),
            {"name": "ElectrostaticEnergy", "params": {"cutoff_lr": None, "flavor": "PhysNet"}},
            {"name": "ChargeConservation"},
            {"name": "AtomicCharge2Dipole"},
            {"name": "EnergyReduce"},
            {"name": "Force"},
        ],
    )
    v["spookynet_ff02"] = dict(
        architecture="SpookyNet",
        build_params={"dim_embedding": 16, "num_rbf": 8, "max_Za": 86, "cutoff_sr": 5.291772105638412,
                      "activation_fn": "swish", **UNIT},
        layers=[
            {"name": "RangeSeparation", "params": {"cutoff_fn": "bump"}},
            {"name": "ExponentialBernsteinRBF", "params": {"no_basis_at_infinity": False,
                                                           "init_alpha": 0.944863062918464,
                                                           "exp_weighting": False, "learnable_shape": True}},
            {"name": "NuclearEmbedding", "params": {"zero_init": False, "use_electron_config": True}},
            {"name": "ElectronicEmbedding", "params": {"num_residual": 1, "attribute": "charge"}},
            {"name": "ElectronicEmbedding", "params": {"num_residual": 1, "attribute": "spin"}},
            {"name": "ScalarResidualMLPEmbedding", "params": {
                "embed_field": "Qa_xTB", "num_residual": 1, "activation_fn": "swish",
                "activation_params": {"dim_feature": 1, "initial_alpha": 1.0, "initial_beta": 1.0},
                "learnable": False, "initial_weight1": "orthogonal", "initial_weight2": "orthogonal",
                "initial_weight_out": "orthogonal", "initial_bias_residual": "zero", "initial_bias_out": "zero",
                "use_bias_residual": True, "use_bias_out": True, "use_residual": True}},
            {"name": "Core", "params": dict(SPOOKY_CORE)},
        ] + _tail("SpookyNet"),
        extra_atomic=("Qa_xTB",),
        with_charge_spin=True,
    )
    v["spookynet_mtase"] = dict(
        architecture="SpookyNet",
        build_params={"dim_embedding": 16, "num_rbf": 8, "max_Za": 86, "cutoff_sr": 5.291772108,
                      "activation_fn": "swish", "activation_params": {"learnable": True}, **UNIT},
        layers=[
            {"name": "RangeSeparation", "params": {"cutoff_fn": "bump"}},
            {"name": "ExponentialBernsteinRBF", "params": {"no_basis_at_infinity": False,
                                                           "init_alpha": 0.944863062918464,
                                                           "exp_weighting": False, "learnable_shape": True}},
            {"name": "NuclearEmbedding", "params": {"zero_init": False, "use_electron_config": True}},
            {"name": "ElectronicEmbedding", "params": {"num_residual": 1, "attribute": "charge"}},
            {"name": "ElectronicEmbedding", "params": {"num_residual": 1, "attribute": "spin"}},
            {"name": "Core", "params": dict(SPOOKY_CORE)},
        ] + _tail("SpookyNet"),
        with_charge_spin=True,
    )
    v["spookynet_noirreps"] = copy.deepcopy(v["spookynet_mtase"])
    v["spookynet_noirreps"]["layers"][5]["params"]["use_irreps"] = False
    v["spookynet_noirreps"]["build_params"].pop("activation_params")

    mace_pre = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"trainable": False}},
        {"name": "NuclearEmbedding", "params": {"zero_init": False, "use_electron_config": True}},
        {"name": "ElectronicEmbedding", "params": {"num_residual": 1, "attribute": "charge"}},
        {"name": "ElectronicEmbedding", "params": {"num_residual": 1, "attribute": "spin"}},
    ]
    mace_build = {"cutoff_sr": 5.0, "dim_embedding": 16, "num_rbf": 8, "max_Za": 86,
                  "cutoff_fn": "polynomial", **UNIT}
    mace_tail = [
        copy.deepcopy(AFFINE),
        {"name": "ChargeConservation"},
        {"name": "ElectrostaticEnergy", "params": {"flavor": "PhysNet"}},
        {"name": "AtomicCharge2Dipole"},
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    v["mace_ff04"] = dict(
        architecture="MACE", build_params=dict(mace_build),
        layers=copy.deepcopy(mace_pre) + [{"name": "Core", "params": dict(MACE_CORE)}] + copy.deepcopy(mace_tail),
        with_charge_spin=True,
    )
    mtase_core = dict(MACE_CORE, max_ell=2, interaction_cls_first="RealAgnosticInteractionBlock",
                      avg_num_neighbors=23.330124060312908)
    v["mace_mtase"] = dict(
        architecture="MACE", build_params=dict(mace_build),
        layers=copy.deepcopy(mace_pre) + [{"name": "Core", "params": mtase_core}] + copy.deepcopy(mace_tail),
        with_charge_spin=True,
    )
    l2_core = dict(MACE_CORE, max_ell=2, additional_hidden_irreps="16x1o+16x2e", correlation=2,
                   num_interactions=3)
    v["mace_l2_c2"] = dict(
        architecture="MACE", build_params=dict(mace_build),
        layers=copy.deepcopy(mace_pre) + [{"name": "Core", "params": l2_core}] + copy.deepcopy(mace_tail),
        with_charge_spin=True,
    )
    c1_core = dict(MACE_CORE, max_ell=1, correlation=[1, 2], additional_hidden_irreps="16x1o")
    v["mace_l1_c12"] = dict(
        architecture="MACE", build_params=dict(mace_build),
        layers=copy.deepcopy(mace_pre) + [{"name": "Core", "params": c1_core}] + copy.deepcopy(mace_tail),
        with_charge_spin=True,
    )
    return v


def random_molecules(rng, sizes, elements=(1, 6, 7, 8), min_dist=0.95, density=0.08):
    mols = []
    for n in sizes:
        box = (n / density) ** (1.0 / 3.0)
        pos = []
        while len(pos) < n:
            p = rng.random(3) * box
            if all(np.linalg.norm(p - q) >= min_dist for q in pos):
                pos.append(p)
        mols.append(dict(Ra=np.asarray(pos), Za=rng.choice(elements, size=n)))
    return mols


def full_pairs(n):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = ii != jj
    return ii[mask], jj[mask]


def make_batch(mols, rng, dtype, extra_atomic=(), with_charge_spin=False):
    Ra, Za, idx_i, idx_j, seg, N = [], [], [], [], [], []
    count = 0
    for k, m in enumerate(mols):
        n = len(m["Za"])
        i, j = full_pairs(n)
        Ra.append(m["Ra"]); Za.append(m["Za"]); idx_i.append(i + count); idx_j.append(j + count)
        seg.append(np.full(n, k)); N.append(n); count += n
    out = {
        "Ra": torch.tensor(np.concatenate(Ra), dtype=dtype),
        "Za": torch.tensor(np.concatenate(Za), dtype=torch.long),
        "idx_i": torch.tensor(np.concatenate(idx_i), dtype=torch.long),
        "idx_j": torch.tensor(np.concatenate(idx_j), dtype=torch.long),
        "batch_seg": torch.tensor(np.concatenate(seg), dtype=torch.long),
        "N": N,
        "Q": torch.tensor([float(rng.integers(-1, 2)) for _ in mols], dtype=dtype),
    }
    if with_charge_spin:
        out["S"] = torch.tensor([float(rng.integers(0, 2)) for _ in mols], dtype=dtype)
    for key in extra_atomic:
        out[key] = torch.tensor(rng.normal(scale=0.3, size=count), dtype=dtype)
    return out


def cast_batch(batch, dtype):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.detach().clone().to(dtype)
        elif torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def run(model, batch, dtype, train=False):
    model = model.to(dtype)
    model.train(train)
    inp = cast_batch(batch, dtype)
    inp["Ra"].requires_grad_(True)
    out = model(inp)
    return {k: v for k, v in out.items() if k in TARGETS and torch.is_tensor(v)}


def detach(outs):
    return {k: v.detach().clone() for k, v in outs.items()}


def loss_fn(out):
    return (out["E"] ** 2).sum() + (out["Fa"] ** 2).sum() + (out["Qa"] ** 2).sum()


def build(cfg):
    return build_model(cfg["architecture"], copy.deepcopy(cfg["layers"]), copy.deepcopy(cfg["build_params"]), verbose=0)


def generate(name, cfg, out_dir, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    extra = cfg.get("extra_atomic", ())
    cs = cfg.get("with_charge_spin", False)
    train_batch = make_batch(random_molecules(rng, [7, 5, 9]), rng, torch.float32, extra, cs)
    test_batch = make_batch(random_molecules(rng, [8, 6]), rng, torch.float64, extra, cs)

    model = build(cfg).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-6, amsgrad=True)
    ema = ExponentialMovingAverage(model.parameters(), 0.9, True)
    for _ in range(3):
        optimizer.zero_grad()
        loss_fn(run(model, train_batch, torch.float32, train=True)).backward()
        optimizer.step()
        ema.update()
    ckpt = {
        "model_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "ema_state_dict": copy.deepcopy(ema.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "epoch": 3,
    }

    ref = {}
    fresh = build(cfg)
    fresh.load_state_dict(ckpt["model_state_dict"])
    ref["raw_fp64"] = detach(run(fresh, test_batch, torch.float64))
    ref["raw_fp32"] = detach(run(fresh, test_batch, torch.float32))

    fresh = build(cfg).float()
    fresh.load_state_dict(ckpt["model_state_dict"])
    tmp = ExponentialMovingAverage(fresh.parameters(), decay=1, use_num_updates=True)
    tmp.load_state_dict(ckpt["ema_state_dict"])
    tmp.copy_to(fresh.parameters())
    ref["ema_fp64"] = detach(run(fresh, test_batch, torch.float64))

    fresh = build(cfg).float()
    fresh.load_state_dict(ckpt["model_state_dict"])
    opt = torch.optim.Adam(fresh.parameters(), lr=1e-3, eps=1e-6, amsgrad=True)
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    opt.zero_grad()
    loss_fn(run(fresh, train_batch, torch.float32, train=True)).backward()
    opt.step()
    ref["after_step_params"] = {k: v.detach().clone() for k, v in fresh.named_parameters()}

    payload = {
        "name": name,
        "architecture": cfg["architecture"],
        "layers": cfg["layers"],
        "build_params": cfg["build_params"],
        "checkpoint": ckpt,
        "main_param_names": [k for k, _ in model.named_parameters()],
        "train_batch": train_batch,
        "test_batch": test_batch,
        "reference": ref,
        "torch_version": torch.__version__,
    }
    os.makedirs(out_dir, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    with gzip.open(os.path.join(out_dir, f"{name}.pt.gz"), "wb", compresslevel=9) as f:
        f.write(buffer.getvalue())
    print(name, "ok", {k: tuple(v.shape) for k, v in ref["raw_fp64"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    for k, (name, cfg) in enumerate(sorted(variants().items())):
        if args.only and name not in args.only.split(","):
            continue
        generate(name, cfg, args.out, seed=1000 + k)


if __name__ == "__main__":
    main()
