import os.path as osp
from typing import Dict, Optional
from torch.nn import Module
import numpy as np
import ase
import ase.io
import torch
from copy import copy
from typing import Literal
from itertools import combinations
from functools import partial
from ase.constraints import FixBondLengths
from ase.units import Hartree, fs, kB, Bohr
from ..data.transform import Transform
from ..utils import logger
from .trainer import DTYPE_MAPPING, _load_state_dict


def get_optimizer(
    optimizer_name: Literal["BFGS", "LBFGS", "MDMin", "FIRE", "GPMin", "BFGSLineSearch", "LBFGSLineSearch", "odesolver", "static"], 
    neb: bool=False
):
    if optimizer_name == "BFGS":
        from ase.optimize import BFGS
        return BFGS
    elif optimizer_name == "LBFGS":
        from ase.optimize import LBFGS
        return LBFGS
    elif optimizer_name == "MDMin":
        from ase.optimize import MDMin
        return MDMin
    elif optimizer_name == "FIRE":
        from ase.optimize import FIRE
        return FIRE
    elif optimizer_name == "GPMin":
        from ase.optimize import GPMin
        return GPMin
    elif optimizer_name == "BFGSLineSearch":
        from ase.optimize import BFGSLineSearch
        return BFGSLineSearch
    elif optimizer_name == "LBFGSLineSearch":
        from ase.optimize import LBFGSLineSearch
        return LBFGSLineSearch
    elif neb == True:
        from ase.mep.neb import NEBOptimizer
        if optimizer_name == "odesolver":
            return partial(NEBOptimizer, method="ODE")
        elif optimizer_name == "static":
            return partial(NEBOptimizer, method="static")
        else:
            raise ValueError(f"NEB optimizer {optimizer_name} not supported")
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")


class Simulation:
    def __init__(self, config: Dict, model: Module, model_path: str, out_dir: str, transform: Transform, calculator_patch_module: Optional[object]=None, plumed_patch_module: Optional[object]=None) -> None:
        self.environment = config.Simulation.environment
        self.task = config.Simulation.task
        self.structure_file = config.System.structure_file
        self.charge = config.System.charge
        self.multiplicity = config.System.multiplicity
        self.transform = transform
        self.external_calculator_config = config.Simulation.get("external_calculator", dict())
        self.uncertainty_calculator_config = config.Simulation.get("uncertainty_calculator", None)
        self.plumed_config_generator_config = config.Simulation.get("plumed_config_generator", None)
        self.internal_calculator_weight = config.Simulation.get("internal_calculator_weight", 1.0)
        self.idx_start_from = config.Simulation.get("idx_start_from", 1)
        self.integrate_config = config.Simulation.get("integrate", None)
        self.initialize_config = config.Simulation.get("initialize", None)
        self.constraint_config = config.Simulation.get("constraint", None)
        self.sampling_config = config.Simulation.get("sampling", None)
        self.optimize_config = config.Simulation.get("optimize", None)
        self.fs_in_t = config.Simulation.get("fs_in_t", 1)
        self.log_interval = config.Simulation.get("log_interval", 20)
        self.neighbor_list_type = config.Simulation.get("neighbor_list", "full")
        self.cuda = config.Simulation.get('cuda', False)
        self.dtype = DTYPE_MAPPING[config.Simulation.get("dtype", "float64")]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        # single ff simulation
        self.model = model.to(self.device).type(self.dtype)
        _load_state_dict(model, self.device, model_path, inference=True)
        self.model.eval()
        self.calculator = None
        self.out_dir = out_dir
        self.calculator_patch_module = calculator_patch_module
        self.plumed_patch_module = plumed_patch_module
        getattr(self, f"_init_{self.environment}_env")()

    def _init_ase_env(self):
        from .calculator import ASECalculator
        self.initial_structures = ase.io.read(self.structure_file, ":")
        for atoms in self.initial_structures:
            atoms.info.update({"spin": self.multiplicity, "charge": self.charge})
        self.systems = self.initial_structures.copy()
        self.system = self.systems[-1]
        if self.calculator_patch_module is not None:
            external_calculator_name = self.external_calculator_config.get("name", None)
            if external_calculator_name is not None:
                if hasattr(self.calculator_patch_module, external_calculator_name):
                    self.external_calculator = getattr(self.calculator_patch_module, external_calculator_name)
                else:
                    raise ValueError(f"External calculator {external_calculator_name} not found in {self.calculator_patch_module}")
            else:
                raise ValueError(f"External calculator name not specified!")
            logger.info(f"Initialized external calculator: {external_calculator_name}")
        else:
            self.external_calculator = None
        self.calculator = ASECalculator(
            model=self.model,
            device=self.device,
            dtype=self.dtype,
            neighbor_list_type=self.neighbor_list_type,
            transform=self.transform,
            internal_calculator_weight=self.internal_calculator_weight,
            external_calculator=self.external_calculator,
            external_calculator_config=self.external_calculator_config,
            uncertainty_calculator_config=self.uncertainty_calculator_config
        )
        if self.constraint_config is not None:
            self.constraints = []
            for k, v in self.constraint_config.items():
                if k == "fix_atom":
                    from ase.constraints import FixAtoms
                    c = FixAtoms(indices=[idx - self.idx_start_from for idx in v.indices])
                    self.constraints.append(c)
                elif k == "Hookean_allpairs":
                    from ase.constraints import Hookean
                    c = [
                        Hookean(i, j, 
                            k=v.k * Hartree / self.calculator.Hartree_in_E,
                            rt=self.system.get_distance(i - self.idx_start_from, j - self.idx_start_from)
                        ) for i, j in combinations(v.indices, 2)
                    ]
                    self.constraints.extend(c)

        for system in self.systems:
            system.calc = copy(self.calculator)
            system.set_constraint(self.constraints)

    def _run_sp(self):
        for system in self.systems:
            for property in ["energy", "forces", "dipole", "charges"]:
                if property in system.info:
                    system.info.pop(property)
            system.get_potential_energy()
            ase.io.write(osp.join(self.out_dir, f"sp.xyz"), system, append=True)

    def _run_opt(self):
        logger.info(f"Running optimization with {self.optimize_config.optimizer} optimizer")
        optimizer = get_optimizer(self.optimize_config.optimizer)(self.system)
        def write_xyz(atoms=None):
            ase.io.write(osp.join(self.out_dir, f"traj-opt.xyz"), atoms, append=True)
        optimizer.attach(write_xyz, interval=1, atoms=self.system) 
        optimizer.run(fmax=self.optimize_config.get("fmax", 4.5e-4) / self.system.calc.Hartree_in_E * Hartree / Bohr)
        ase.io.write(osp.join(self.out_dir, f"optim.xyz"), self.system, append=True)
        logger.info(f"Final energy: {self.system.get_potential_energy()}")

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
                optimizer = get_optimizer(self.optimize_config.optimizer)(self.system)
                def write_xyz(atoms=None):
                    ase.io.write(osp.join(self.out_dir, f"traj-{i}.xyz"), atoms, append=True)
                optimizer.attach(write_xyz, interval=1, atoms=self.system) 
                optimizer.run(fmax=4.5e-4 / self.system.calc.Hartree_in_E * Hartree / Bohr)
                ase.io.write(osp.join(self.out_dir, f"scan_optim.xyz"), self.system, append=True)
                logger.info(f"Final energy: {self.system.get_potential_energy()}")
                logger.info(f"Final distance: {self.system.get_distance(i0, i1)}")

    def _get_integrator(self, traj_file):
        if self.integrate_config.integrator.lower() == "langevin":
            from ase.md.langevin import Langevin
            dyn = Langevin(self.system, 
                timestep=self.integrate_config.time_step * fs / self.fs_in_t,
                temperature_K=self.integrate_config.temperature_in_K,
                friction=self.integrate_config.friction,
                logfile="-",
                loginterval=self.log_interval
            )
            def write_xyz(atoms=None):
                write_atoms = atoms.copy()
                write_atoms.constraints = []
                ase.io.write(osp.join(self.out_dir, traj_file), write_atoms, append=True)
            dyn.attach(write_xyz, interval=self.log_interval, atoms=self.system) 
        return dyn

    def _md_initialize(self):
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        MaxwellBoltzmannDistribution(self.system, temperature_K=self.initialize_config.get("temperature_in_K", self.integrate_config.temperature_in_K))

    def _run_md(self) -> None:
        if self.initialize_config is not None:
            self._md_initialize()
        dyn = self._get_integrator(traj_file="md.traj.xyz")
        dyn.run(self.integrate_config.n_step)

    def _run_neb(self):
        from ase.mep.neb import NEB, NEBTools
        num_images = self.sampling_config.params.num_images
        assert num_images > 2, "Number of images must be greater than 2"
        requires_interpolation = 0
        if len(self.initial_structures) == 2:
            requires_interpolation = 1
            images = []
            for _ in range(num_images - 1):
                images.append(self.initial_structures[0].copy())
            images.append(self.initial_structures[1].copy())
        elif len(self.initial_structures) == num_images:
            images = self.systems
        elif len(self.initial_structures) == 3:
            requires_interpolation = 2
            images = []
        else:
            raise ValueError(f"Number of initial structures {len(self.initial_structures)} is not capatible with the number of images {num_images}")
        
        if requires_interpolation:
            for image in images:
                image.calc = copy(self.calculator)
                image.set_constraint(self.constraints)

        if self.sampling_config.params.get("relax_endpoints", True):
            logger.info("Relaxing endpoints")
            # relaxing reactant
            optimizer = get_optimizer(self.optimize_config.optimizer)(images[0])
            def write_xyz(atoms=None):
                ase.io.write(osp.join(self.out_dir, f"neb-relax-reactant.xyz"), atoms, append=True)
            optimizer.attach(write_xyz, interval=1, atoms=images[0])
            optimizer.run(fmax=4.5e-4 / self.calculator.Hartree_in_E * Hartree / Bohr)
            # relaxing product
            optimizer = get_optimizer(self.optimize_config.optimizer)(images[-1])
            def write_xyz(atoms=None):
                ase.io.write(osp.join(self.out_dir, f"neb-relax-product.xyz"), atoms, append=True)
            optimizer.attach(write_xyz, interval=1, atoms=images[-1])
            optimizer.run(fmax=4.5e-4 / self.calculator.Hartree_in_E * Hartree / Bohr)

        if requires_interpolation:
            if requires_interpolation == 1:
                logger.info(f"Interpolating {num_images} images between reactant and product")
                interpolated_neb = NEB(images)
                interpolated_neb.interpolate(
                    method=self.sampling_config.params.interpolation.method,
                    apply_constraint=self.sampling_config.params.interpolation.apply_constraint
                )
            elif requires_interpolation == 2:
                logger.info(f"Interpolating {num_images} images between reactant, guessed TS and product")
                middle_idx = num_images // 2
                first_half_neb = NEB(images[:middle_idx])
                first_half_neb.interpolate(
                    method=self.sampling_config.params.interpolation.method,
                    apply_constraint=self.sampling_config.params.interpolation.apply_constraint
                )
                second_half_neb = NEB(images[middle_idx:])
                second_half_neb.interpolate(
                    method=self.sampling_config.params.interpolation.method,
                    apply_constraint=self.sampling_config.params.interpolation.apply_constraint
                )
            
        neb = NEB(
            images,
            k=self.sampling_config.params.spring_constants / self.calculator.Hartree_in_E * Hartree / (Bohr ** 2), 
            climb=False, 
            allow_shared_calculator=True
        )
        neb_tools = NEBTools(neb.images)

        # plain neb
        neb_optimizer_name = self.sampling_config.params.get("neb_optimizer", "odesolver")
        optimizer = get_optimizer(neb_optimizer_name, neb=True)(neb, trajectory=osp.join(self.out_dir, "neb.traj"))
        
        def write_xyz(images=None):
            for i, image in enumerate(images):
                ase.io.write(osp.join(self.out_dir, f"neb-{i}.xyz"), image, append=True)
            ase.io.write(osp.join(self.out_dir, f"neb.xyz"), images, append=False)
        optimizer.attach(write_xyz, interval=1, images=images)
        if self.sampling_config.params.get("climb", False):
            optimizer.run(fmax=4.5e-4 / self.calculator.Hartree_in_E * Hartree / Bohr * 2)
        else:
            optimizer.run(fmax=4.5e-4 / self.calculator.Hartree_in_E * Hartree / Bohr)
        barrier, dE = neb_tools.get_barrier()
        logger.info(f"NEB barrier: {barrier}, dE: {dE}")
        ase.io.write(osp.join(self.out_dir, f"neb.xyz"), images, append=False)

        if self.sampling_config.params.get("climb", False):
            neb.climb = True
            ci_neb_optimizer_name = self.sampling_config.params.get("ci_neb_optimizer", neb_optimizer_name)
            optimizer = get_optimizer(ci_neb_optimizer_name, neb=True)(neb, trajectory=osp.join(self.out_dir, "ci-neb.traj"))
            def write_xyz(images=None):
                for i, image in enumerate(images):
                    ase.io.write(osp.join(self.out_dir, f"ci-neb-{i}.xyz"), image, append=True)
                ase.io.write(osp.join(self.out_dir, f"ci-neb.xyz"), images, append=False)
            optimizer.attach(write_xyz, interval=1, images=images)
            optimizer.run(fmax=4.5e-4 / self.calculator.Hartree_in_E * Hartree / Bohr)
            barrier, dE = neb_tools.get_barrier()
            logger.info(f"CI-NEB barrier: {barrier}, dE: {dE}")
            ase.io.write(osp.join(self.out_dir, f"ci-neb.xyz"), images, append=False)

    def _run_plumed(self):
        if self.plumed_patch_module is not None:
            plumed_config_generator_name = self.plumed_config_generator_config.get("name", None)
            if plumed_config_generator_name is not None:
                if hasattr(self.plumed_patch_module, plumed_config_generator_name):
                    self.plumed_config_generator = getattr(self.plumed_patch_module, plumed_config_generator_name)
                else:
                    raise ValueError(f"Plumed config generator {plumed_config_generator_name} not found in {self.plumed_patch_module}")
            else:
                raise ValueError(f"Plumed config generator name not specified!")
            logger.info(f"Initialized plumed config generator: {plumed_config_generator_name}")
            plumed_config = self.plumed_config_generator(
                self.system, 
                integrate_config=self.integrate_config,
                idx_start_from=self.idx_start_from,
                **self.sampling_config.params.plumed_config
            )
            with open(osp.join(self.out_dir, "plumed.dat"), "w") as f:
                f.writelines([line + "\n" for line in plumed_config])
        else:
            plumed_config = self.sampling_config.params.plumed_config
        from ase.calculators.plumed import Plumed
        plumed_calc = Plumed(self.calculator, plumed_config, 
            timestep=self.integrate_config.time_step * fs / self.fs_in_t,
            atoms=self.system,
            kT=self.integrate_config.temperature_in_K * kB,
            log=osp.join(self.out_dir, "plumed.log"),
            restart=False
        )
        if self.initialize_config is not None:
            self._md_initialize()
        dyn = self._get_integrator(traj_file="plumed.traj.xyz")
        dyn.run(self.integrate_config.n_step)

    def run(self):
        getattr(self, f"_run_{self.task}")()
