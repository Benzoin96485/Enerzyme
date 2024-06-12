import ase
import torch
import numpy as np
import pandas as pd
from ase.constraints import FixAtoms, FixBondLengths
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
from ase.optimize.sciopt import SciPyFminBFGS
from sklearn.neighbors import BallTree


class ASECalculator(Calculator):
    implemented_properties = ["energy", "forces", "hessian", "dipole", "charges"]

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
        self.dtype = self.model.dtype
        self.N = 0
        self.positions = None
        self.device = self.parameters.device
        self.pbc = np.array([False])
        self.cell = None
        self.cell_offsets = None
        

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        feature = {
            "Q": pd.Series([self.parameters.charge]),
            "Ra": pd.Series([atoms.positions]),
            "Za": pd.Series([atoms.numbers]),
        }
        net_input, aux_target = self.model.batch_collate_fn(feature, has_label=False)
        for k, v in net_input.items():
            if isinstance(v, torch.Tensor):
                net_input[k] = v.to(self.device)
        for k, v in aux_target.items():
            if isinstance(v, torch.Tensor):
                aux_target[k] = v.to(self.device)
        net_output = self.model(task="qep", **net_input)
        output, _ = self.model.batch_output_collate_fn(net_output, aux_target, has_label=False)
        self.results["energy"] = output["E"].detach().cpu().numpy()
        self.results["dipole"] = output["P"][0].detach().cpu().numpy()
        self.results["forces"] = output["F"][0].detach().cpu().numpy()
        self.results["charges"] = output["Qa"][0].detach().cpu().numpy()


class Simulation:
    def __init__(self, config, model):
        self.environment = config.Simulation.environment
        self.task = config.Simulation.task
        self.structure_file = config.System.structure_file
        self.charge = config.System.charge
        self.multiplicity = config.System.multiplicity
        self.idx_start_from = config.Simulation.get("idx_start_from", 1)
        self.constraint_config = config.Simulation.get("constraint", None)
        self.sampling_config = config.Simulation.get("sampling", None)
        self.simulation_config = config.Simulation
        self.cuda = config.Simulation.get('cuda', False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        # single ff simulation
        self.model = model.to(self.device)
        self.model.eval()
            
        # initialize
        getattr(self, f"_init_{self.environment}_env")()

    def _init_ase_env(self):
        self.system = ase.io.read(self.structure_file)
        calculator = ASECalculator(
            model=self.model,
            charge=self.charge,
            magmom=self.multiplicity - 1,
            device=self.device,
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
                print(x)
                del self.system.constraints
                print(self.system.get_distance(i0, i1))
                self.system.set_distance(i0, i1, x)
                c = FixBondLengths([(i0, i1)], bondlengths=[x], tolerance=1e-6)
                self.system.set_constraint(self.constraints + [c])
                optimizer = BFGS(self.system)
                def write_xyz(atoms=None):
                    ase.io.write(f"traj-{i}.xyz", atoms, append=True)
                optimizer.attach(write_xyz, interval=1, atoms=self.system) 
                optimizer.run(fmax=4.5e-4)
                ase.io.write(f"scan_optim.xyz", self.system, append=True)
                print(f"Final_Energy: {self.system.get_potential_energy()}")
                print(self.system.get_distance(i0, i1))

    def run(self):
        getattr(self, f"_run_{self.task}")()
