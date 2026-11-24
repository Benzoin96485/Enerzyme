"""Rebuilt Cores reproduce the float64 outputs frozen before the refactor.

Snapshots come from ``scripts/snapshot_cores.py`` run on the pre-refactor
commit; they are local artefacts, so the test skips when none exist.
"""

import glob
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, HERE)

from core_harness import build, forward, targets  # noqa: E402
from presnapshot_maps import remap_snapshot  # noqa: E402

SNAP_DIR = os.path.join(HERE, "fixtures", "presnapshot")
SNAPSHOTS = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))
ATOL = 1e-10
ATOL_FP32 = 1e-5


def _arch_id(path):
    return os.path.basename(path)[:-3]


@pytest.mark.skipif(not SNAPSHOTS, reason="no pre-refactor snapshots; run scripts/snapshot_cores.py on the base commit")
@pytest.mark.parametrize("path", SNAPSHOTS, ids=[_arch_id(p) for p in SNAPSHOTS])
def test_presnapshot_parity(path):
    payload = torch.load(path, weights_only=False)
    arch = payload["architecture"]
    torch.manual_seed(7)
    model = build(arch).float()
    state = remap_snapshot(arch, payload["state_dict"], payload["param_names"], model)
    model.load_state_dict(state, strict=True)
    dtype = payload.get("dtype", torch.float64)
    model = model.to(dtype)
    out = targets(forward(model, payload["batch"], dtype))
    tol = ATOL if dtype == torch.float64 else ATOL_FP32
    assert set(out) == set(payload["outputs"]), (sorted(out), sorted(payload["outputs"]))
    for key, ref in payload["outputs"].items():
        scale = max(1.0, float(ref.abs().max())) if ref.numel() else 1.0
        diff = float((out[key] - ref).abs().max()) if ref.numel() else 0.0
        assert diff <= tol * scale, f"{arch} {key}: max |diff| {diff:.3e} (scale {scale:.3e})"
