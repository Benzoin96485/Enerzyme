import pathlib, importlib, sys, os
from typing import Dict, Optional, Any
from torch.nn import Module
import numpy as np
import ase
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree
from ..data.transform import Transform
from ..data.neighbor_list import full_neighbor_list
from .trainer import _decorate_batch_output, _decorate_batch_input, _to_device


def _init_patch_module(patch_path: str) -> None:
    p = pathlib.Path(patch_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix != ".py":
        raise ValueError(f"Plugin must be a .py file, got: {p}")

    # Hash the patch file to get a unique module name
    module_name = f"plugin_{p.stem}_{abs(hash(str(p)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {patch_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

class ASECalculator(Calculator):
    implemented_properties = ["energy", "forces", "dipole", "charges"]

    def __init__(
        self,
        model: Module,
        restart: Optional[str]=None,
        label: Optional[str]=None,
        atoms: Optional[ase.Atoms]=None,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None,
        transform: Optional[Transform]=None,
        neighbor_list_type: Optional[str]="full",
        Hartree_in_E: float=1,
        internal_calculator_weight: float=1.0,
        uncertainty_calculator_config: Optional[Dict[str, Any]]=None,
        external_calculator: Optional[Calculator]=None,
        external_calculator_config: Optional[Dict[str, Any]]=None,
        **params: Dict[str, Any]
    ) -> None:
        Calculator.__init__(
            self, restart=restart, label=label, atoms=atoms, **params
        )
        self.model = model
        self.positions = None
        self.device = device
        self.pbc = np.array([False])
        self.cell = None
        self.cell_offsets = None
        self.dtype = dtype
        self.neighbor_list_type = neighbor_list_type
        self.transform = transform
        self.Hartree_in_E = Hartree_in_E
        self.E_conversion_factor = Hartree / self.Hartree_in_E
        self.internal_calculator_weight = internal_calculator_weight
        self.external_calculator = external_calculator
        if self.external_calculator is None:
            self.external_calculator_weight = None
            self.use_external_calculator = False
        else:
            self.external_calculator_weight = external_calculator_config.get("weight", 0.0)
            self.use_external_calculator = True
        self.uncertainty_calculator_config = uncertainty_calculator_config

    def _calculate_UDD(self, output: Dict[str, Any], A: float, B: float, NM: Optional[int]=None) -> Dict[str, Any]:
        results, biases = dict(), dict()
        
        if "E_var_grad" in output:
            E_var = output["E_var"][0]
            E_var_grad = output["E_var_grad"][0]
            E_mean = output["E"][0]
            F_mean = output["Fa"][0]
            NM = self.model.shallow_ensemble_size
        else:
            NM = output["E"][0].shape[-1]
            E_mean = np.mean(output["E"][0], axis=-1, keepdims=True) # shape: (1, 1)
            E_dev = output["E"][0] - E_mean # shape: (1, NM)
            E_var = np.mean(E_dev**2, axis=-1, keepdims=True).squeeze()
            F_mean = np.mean(output["Fa"][0], axis=-1, keepdims=True) # shape: (NA, 3, 1)
            F_dev = output["Fa"][0] - F_mean # shape: (NA, 3, NM)
            E_var_grad = -np.sum(F_dev * E_dev, axis=-1) # shape: (NA, 3)

        scale = 1 / (2 * NM * B)
        neg_scaled_var = -E_var * scale
        exp_neg_scaled_var = np.exp(neg_scaled_var)
        E_bias = A * (exp_neg_scaled_var - 1)
        F_bias = A / scale * exp_neg_scaled_var * E_var_grad # shape: (NA, 3)

        results["energy"] = E_mean.squeeze() * self.E_conversion_factor
        biases["energy"] = E_bias * self.E_conversion_factor
        results["forces"] = F_mean.squeeze() * self.E_conversion_factor
        biases["forces"] = F_bias * self.E_conversion_factor
        if "M2" in output:
            results["dipole"] = output["M2"][0]
        if "Qa" in output:
            results["charges"] = output["Qa"][0]
        return results, biases

    def _calculate_internal(self, atoms: ase.Atoms) -> None:
        results = dict()
        features = {
            "Q": atoms.info.get("charge", 0),
            "S": atoms.info.get("spin", 0) - 1,
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
        net_input, _ = _to_device((net_input, {}), self.device)
        net_output = self.model(net_input)
        output, _ = _decorate_batch_output(
            output=net_output,
            features=net_input,
            targets=None
        )
        self.transform.inverse_transform(output)
        if self.uncertainty_calculator_config is None:
            results["energy"] = output["E"][0] * self.E_conversion_factor
            results["forces"] = output["Fa"][0] * self.E_conversion_factor
            if "M2" in output:
                results["dipole"] = output["M2"][0]
            if "Qa" in output:
                results["charges"] = output["Qa"][0]
            biases = dict()
        else:
            uncertainty_calculator_name = self.uncertainty_calculator_config.get("name", None)
            if uncertainty_calculator_name == "UDD":
                results, biases = self._calculate_UDD(output, **self.uncertainty_calculator_config.get("params"))
            else:
                raise ValueError(f"Uncertainty calculator {uncertainty_calculator_name} not supported")
        return results, biases

    def calculate(self, atoms=None, properties=["energy", "forces", "dipole", "charges"], system_changes=all_changes) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        if self.internal_calculator_weight != 0 or self.uncertainty_calculator_config is not None:
            internal_results, biases = self._calculate_internal(atoms)
        if self.use_external_calculator:
            external_calculator_properties = ['energy', 'forces']  
            self.external_calculator.calculate(atoms, properties=properties, system_changes=system_changes)
            external_results = dict()
            for property in external_calculator_properties:
                external_results[property] = self.external_calculator.results[property]
        
        for property in properties:
            if self.use_external_calculator and property in external_calculator_properties:
                if self.internal_calculator_weight == 0:
                    self.results[property] = self.external_calculator_weight * external_results[property]
                else:
                    self.results[property] = self.internal_calculator_weight * internal_results[property] + self.external_calculator_weight * external_results[property]
            else:
                self.results[property] = self.internal_calculator_weight * internal_results[property]

            if self.uncertainty_calculator_config is not None and property in ["energy", "forces"]:
                self.results[property] += biases[property]


def get_patched_calculator(calculator_patch_module: object, external_calculator_config: Optional[Dict[str, Any]]=None) -> Calculator:
    external_calculator_name = external_calculator_config.get("name", None)
    if external_calculator_name is not None:
        if hasattr(calculator_patch_module, f"get_{external_calculator_name}"):
            external_calculator = getattr(calculator_patch_module, f"get_{external_calculator_name}")(
                **external_calculator_config.get("params", dict())
            )
        elif hasattr(calculator_patch_module, external_calculator_name):
            external_calculator = getattr(calculator_patch_module, external_calculator_name)
    else:
        raise ValueError(f"External calculator name not specified!")
    return external_calculator


def get_calculator(model_dir: str, device: str="cuda", dtype: str="float64", model_config_path: Optional[str] = None, calculator_patch: Optional[str] = None, neighbor_list_type: Optional[str]="full", Hartree_in_E: float=1, internal_calculator_weight: float=1.0, uncertainty_calculator_config: Optional[Dict[str, Any]]=None, external_calculator: Optional[Calculator]=None, external_calculator_config: Optional[Dict[str, Any]]=None):
    from ..models import get_model_str, build_model, get_pretrain_path
    from ..utils import YamlHandler, logger
    from .trainer import DTYPE_MAPPING, _load_state_dict
    device = torch.device(device)
    dtype = DTYPE_MAPPING[dtype]
    if model_config_path is None:
        model_config_path = os.path.join(model_dir, 'config.yaml')
    model_config = YamlHandler(model_config_path).read_yaml()
    logger.info('Model Config: {}'.format(model_config))
    transform = Transform(model_config.Datahub.transforms, simulation_mode=True)
    if calculator_patch is not None:
        calculator_patch_module = _init_patch_module(calculator_patch)
        logger.info(f"Initialized calculator patch module: {calculator_patch}")
    else:
        calculator_patch_module = None
    if calculator_patch_module is not None:
        external_calculator_name = external_calculator_config.get("name", None)
        external_calculator = get_patched_calculator(calculator_patch_module, external_calculator_config)
        logger.info(f"Initialized external calculator: {external_calculator_name}")
    else:
        external_calculator = None
    for FF_key, FF_params in (model_config.Modelhub.internal_FFs | model_config.Modelhub.external_FFs).items():
        if FF_params.get("active", False):
            model_str = get_model_str(FF_key, FF_params)
            model = build_model(FF_params.architecture, FF_params.layers, FF_params.build_params)
            model_path = get_pretrain_path(os.path.join(model_dir, model_str), "best")
            model = model.to(device).type(dtype)
            _load_state_dict(model, device, model_path, inference=True)
            model.eval()
            calculator = ASECalculator(
                model=model,
                device=device,
                dtype=dtype,
                transform=transform,
                neighbor_list_type=neighbor_list_type,
                Hartree_in_E=Hartree_in_E,
                internal_calculator_weight=internal_calculator_weight,
                uncertainty_calculator_config=uncertainty_calculator_config,
                external_calculator=external_calculator,
                external_calculator_config=external_calculator_config
            )
            break
    return calculator
