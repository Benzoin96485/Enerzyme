import ase.io
import numpy as np
import pytest
import torch
from addict import Dict
from ase import Atoms
from ase.units import Hartree

from enerzyme.data.transform import Transform
from enerzyme.models.layers.geometry import DistanceLayer
from enerzyme.tasks.calculator import ASECalculator
from enerzyme.tasks.server import Server
from enerzyme.tasks.simulator import Simulation


class PairEnergy(torch.nn.Module):
    def forward(self, inputs):
        distances = DistanceLayer().get_output(
            inputs["Ra"], inputs["idx_i"], inputs["idx_j"], inputs.get("offsets")
        )["Dij"]
        energy = distances.square().sum().view(1) / 4
        (forces,) = torch.autograd.grad(-energy, inputs["Ra"])
        return {"E": energy, "Fa": forces}


def test_simulation_and_server_use_the_periodic_image(tmp_path):
    atoms = Atoms("H2", positions=[[0.1, 0, 0], [3.9, 0, 0]], cell=[4, 5, 6], pbc=[True, False, False])
    xyz = tmp_path / "sample.extxyz"
    ase.io.write(xyz, atoms)
    model = PairEnergy()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    options = dict(neighbor_list="cutoff", neighbor_list_cutoff=0.6)
    config = Dict(
        System=dict(structure_file=str(xyz), charge=0, multiplicity=1),
        Simulation=dict(environment="ase", task="sp", **options), Server=options,
    )
    simulation = Simulation(config, model, checkpoint, str(tmp_path), Transform())
    server = Server(config, model, checkpoint, transform=Transform())
    forces = [[-0.2, 0, 0], [0.2, 0, 0]]

    for shift in (0.0, -4.0):
        simulation.system.positions[1, 0] = 3.9 + shift
        assert simulation.system.get_potential_energy() / Hartree == pytest.approx(0.02)
        np.testing.assert_allclose(simulation.system.get_forces() / Hartree, forces, atol=1e-12)
        outputs = server.calculate({"features": {
            "Ra": simulation.system.positions.tolist(), "Za": [1, 1],
            "cell": atoms.cell.tolist(), "pbc": atoms.pbc.tolist(),
        }})
        assert outputs["E"][0] == pytest.approx(0.02)
        np.testing.assert_allclose(outputs["Fa"][0], forces, atol=1e-12)

    simulation.system.positions[1, 0] = 3.9
    simulation.system.set_pbc(False)
    assert simulation.system.get_potential_energy() == 0.0
    np.testing.assert_array_equal(simulation.system.get_forces(), np.zeros((2, 3)))


def test_periodic_calculator_requires_cutoff_and_accepts_an_untransformed_model():
    atoms = Atoms("H2", positions=[[0.1, 0, 0], [3.9, 0, 0]], cell=[4, 5, 6], pbc=True)
    model = PairEnergy()
    atoms.calc = ASECalculator(model=model, dtype=torch.float64)
    with pytest.raises(ValueError, match="Periodic atoms require"):
        atoms.get_potential_energy()
    atoms.calc = ASECalculator(
        model=model, dtype=torch.float64, neighbor_list_type="cutoff", neighbor_list_cutoff=0.6,
    )
    assert atoms.get_potential_energy() / Hartree == pytest.approx(0.02)
