from typing import Any, Dict, Optional

from addict import Dict as AddictDict
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree
import numpy as np
import torch
from torch.nn import Module

from ..data.transform import Transform
from .calculator import ASECalculator
from .trainer import DTYPE_MAPPING, _load_state_dict


class Server:
    def __init__(
        self,
        config: AddictDict,
        model: Optional[Module] = None,
        model_path: Optional[str] = None,
        out_dir: Optional[str] = None,
        transform: Optional[Transform] = None,
        external_calculator: Optional[Calculator] = None,
        external_calculator_config: Optional[Dict[str, Any]] = None,
        internal_calculator_weight: float = 1.0,
        uncertainty_calculator_config: Optional[Dict[str, Any]] = None,
    ):
        self.neighbor_list_type = config.Server.get("neighbor_list", "full")
        self.cuda = config.Server.get("cuda", False)
        self.dtype = DTYPE_MAPPING[config.Server.get("dtype", "float64")]
        self.Hartree_in_E = config.Server.get("Hartree_in_E", 1)
        self.Bohr_in_R = config.Server.get("Bohr_in_R", Bohr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.out_dir = out_dir
        self.transform = transform
        self.internal_calculator_weight = internal_calculator_weight
        self.uncertainty_calculator_config = uncertainty_calculator_config
        self.external_calculator_config = external_calculator_config or dict()
        self.use_internal_calculator = (
            self.internal_calculator_weight != 0 or self.uncertainty_calculator_config is not None
        )

        if self.use_internal_calculator:
            if model is None or model_path is None:
                raise ValueError(
                    "Internal calculator requires a loaded model and model_path"
                )
            self.model = model.to(self.device).type(self.dtype)
            _load_state_dict(self.model, self.device, model_path, inference=True)
            self.model.eval()
        else:
            self.model = None

        self.calculator = ASECalculator(
            model=self.model,
            device=self.device,
            dtype=self.dtype,
            transform=self.transform,
            neighbor_list_type=self.neighbor_list_type,
            Hartree_in_E=self.Hartree_in_E,
            internal_calculator_weight=self.internal_calculator_weight,
            uncertainty_calculator_config=self.uncertainty_calculator_config,
            external_calculator=external_calculator,
            external_calculator_config=self.external_calculator_config,
        )

    def calculate(self, info):
        features = info.get("features", None)
        if features is None:
            return {}
        if features.get("N") is None:
            features["N"] = len(features["Ra"])

        atoms = Atoms(
            numbers=np.asarray(features["Za"]),
            positions=np.asarray(features["Ra"]),
        )
        atoms.info["charge"] = features.get("Q", 0)
        atoms.info["spin"] = features.get("S", 0) + 1

        self.calculator.calculate(
            atoms,
            properties=["energy", "forces", "dipole", "charges"],
        )
        results = self.calculator.results
        inv = self.Hartree_in_E / Hartree

        output = {
            "E": [np.asarray(results["energy"]).reshape(()) * inv],
            "Fa": [np.asarray(results["forces"]) * inv],
        }
        if "dipole" in results:
            output["M2"] = [np.asarray(results["dipole"])]
        if "charges" in results:
            output["Qa"] = [np.asarray(results["charges"])]
        return output
