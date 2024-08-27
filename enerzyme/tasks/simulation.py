import os.path as osp
import numpy as np
import ase
import torch
from ase.constraints import FixAtoms, FixBondLengths
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
from ..data import full_neighbor_list
from ..utils import logger
from .trainer import DTYPE_MAPPING, _decorate_batch_output, _decorate_batch_input


class ASECalculator(Calculator):
    implemented_properties = ["energy", "forces", "dipole", "charges"]

    def __init__(
        self,
        model=None,
        restart=None,
        label=None,
        atoms=None,
        **params
    ):
        Calculator.__init__(
            self, restart=restart, label=label, atoms=atoms, **params
        )
        self.model = model
        self.N = 0
        self.positions = None
        self.device = self.parameters.device
        self.pbc = np.array([False])
        self.cell = None
        self.cell_offsets = None
        self.dtype = self.parameters.dtype
        self.neighbor_list_type = self.parameters.neighbor_list_type
        self.transform = self.parameters.transform

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        features = {
            "Q": self.parameters.charge,
            "Ra": atoms.positions,
            "Za": atoms.numbers,
            "N": len(atoms)
        }
        if self.neighbor_list_type == "full":
            idx_i, idx_j = full_neighbor_list(features["N"])
            features["idx_i"] = idx_i
            features["idx_j"] = idx_j
            features["N_pair"] = len(idx_i)
        net_input, _ = _decorate_batch_input(
            batch=[(features, None)],
            device=self.device,
            dtype=self.dtype
        )
        net_output = self.model(net_input)
        output, _ = _decorate_batch_output(
            output=net_output,
            features=net_input,
            targets=None
        )
        self.transform.inverse_transform(output)
        self.results["energy"] = output["E"][0]
        self.results["dipole"] = output["M2"][0]
        self.results["forces"] = output["Fa"][0]
        self.results["charges"] = output["Qa"][0]


class Simulation:
    def __init__(self, config, model, out_dir, transform):
        self.environment = config.Simulation.environment
        self.task = config.Simulation.task
        self.structure_file = config.System.structure_file
        self.charge = config.System.charge
        self.multiplicity = config.System.multiplicity
        self.transform = transform
        self.idx_start_from = config.Simulation.get("idx_start_from", 1)
        self.constraint_config = config.Simulation.get("constraint", None)
        self.sampling_config = config.Simulation.get("sampling", None)
        self.neighbor_list_type = config.Simulation.get("neighbor_list", "full")
        self.cuda = config.Simulation.get('cuda', False)
        self.dtype = DTYPE_MAPPING[config.Simulation.get("dtype", "float64")]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        # single ff simulation
        self.model = model.to(self.device).type(self.dtype)
        self.model.eval()
        self.out_dir = out_dir
        self.simulation_config = {k: v for k, v in config.Simulation.items() if not hasattr(self, k)}
        # initialize
        getattr(self, f"_init_{self.environment}_env")()

    def _init_ase_env(self):
        self.system = ase.io.read(self.structure_file)
        calculator = ASECalculator(
            model=self.model,
            charge=self.charge,
            magmom=self.multiplicity - 1,
            device=self.device,
            dtype=self.dtype,
            neighbor_list_type=self.neighbor_list_type,
            transform=self.transform,
            **self.simulation_config
        )
        self.system.calc = calculator
        if self.constraint_config is not None:
            self.constraints = []
            for k, v in self.constraint_config.items():
                if k == "fix_atom":
                    c = FixAtoms(indices=[idx - self.idx_start_from for idx in v.indices])
                    self.constraints.append(c)
            self.system.set_constraint(self.constraints)
                
    def _run_scan(self):
        if self.sampling_config.cv == "distance":
            i0 = self.sampling_config.params.i0 - self.idx_start_from
            i1 = self.sampling_config.params.i1 - self.idx_start_from
            x0 = self.sampling_config.params.x0
            x1 = self.sampling_config.params.x1
            num = self.sampling_config.params.num
            x_scan = np.linspace(x0, x1, num)
            for i, x in enumerate(x_scan):
                # reset constraint
                del self.system.constraints
                logger.info(f"Setting distance to: {self.system.get_distance(i0, i1)}")
                self.system.set_distance(i0, i1, x)
                c = FixBondLengths([(i0, i1)], bondlengths=[x], tolerance=1e-6)
                self.system.set_constraint(self.constraints + [c])
                optimizer = BFGS(self.system)
                def write_xyz(atoms=None):
                    ase.io.write(osp.join(self.out_dir, f"traj-{i}.xyz"), atoms, append=True)
                optimizer.attach(write_xyz, interval=1, atoms=self.system) 
                optimizer.run(fmax=4.5e-4)
                ase.io.write(osp.join(f"scan_optim.xyz", f"traj-{i}.xyz"), self.system, append=True)
                logger.info(f"Final energy: {self.system.get_potential_energy()}")
                logger.info(f"Final distance: {self.system.get_distance(i0, i1)}")

    def run(self):
        getattr(self, f"_run_{self.task}")()
