from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from .nn.layers import LinearLayer, RescaledSiLULayer, ScaleShiftLayer
from .nn.representations import CartesianMACE
from ..layers import BaseFFCore


def build_model(**config: Any):
    """Builds feed-forward atomistic neural network from the config file.

    Args:
        atomic_structures (Optional[AtomicStructures], optional): Atomic structures, typically those from the training data sets, 
                                                                  used to compute scale and shift for model predictions as well as 
                                                                  the average number of neighbors. Defaults to None.
        
    Returns:
        ForwardAtomisticNetwork: Atomistic neural network.
    """
    
    # compute scale/shift parameters for the total energy
    # also, compute the average number of neighbors, i.e., the normalization factor for messages

    shift_params = np.zeros(config['n_species'])
    scale_params = np.ones(config['n_species'])

    # prepare (semi-)local atomic representation
    representation = CartesianMACE(**config)

    # prepare readouts
    readouts = nn.ModuleList([])
    for i in range(config['n_interactions']):
        if i == config['n_interactions'] - 1:
            layers = []
            for in_size, out_size in zip([config['n_hidden_feats']] + config['readout_MLP'],
                                        config['readout_MLP'] + [1]):
                layers.append(LinearLayer(in_size, out_size))
                layers.append(RescaledSiLULayer())
            readouts.append(nn.Sequential(*layers[:-1]))
        else:
            readouts.append(LinearLayer(config['n_hidden_feats'], 1))

    scale_shift = ScaleShiftLayer(shift_params=shift_params, scale_params=scale_params)

    return ForwardAtomisticNetwork(representation=representation, readouts=readouts, scale_shift=scale_shift, config=config)


class ForwardAtomisticNetwork(nn.Module):
    """An atomistic model based on feed-forward neural networks.

    Args:
        representation (nn.Module): Local atomic representation layer.
        readouts (nn.ModuleList): List of readout layers.
        scale_shift (nn.Module): Schale/shift transformation applied to the output, i.e., energy re-scaling and shift.
    """
    def __init__(self,
                 representation: nn.Module,
                 readouts: List[nn.Module],
                 scale_shift: nn.Module,
                 config: Dict[str, Any]):
        super().__init__()
        # all necessary modules
        self.representation = representation
        self.readouts = readouts
        self.scale_shift = scale_shift
        
        # provide config file to store it
        self.config = config

    def forward(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes atomic energies for the provided batch.

        Args:
            graph (Dict[str, torch.Tensor]): Atomic data dictionary.

        Returns:
            torch.Tensor: Atomic energies.
        """
        # compute representation (atom/node features)
        atom_feats_list = self.representation(graph)
        
        # apply a readout layer to each representation
        atomic_energies_list = []
        for i, readout in enumerate(self.readouts):
            atomic_energies_list.append(readout(atom_feats_list[i]).squeeze(-1))
        atomic_energies = torch.sum(torch.stack(atomic_energies_list, dim=0), dim=0)
        
        # scale and shift the output
        atomic_energies = self.scale_shift(atomic_energies, graph)
        
        return atomic_energies


class ICTPCore(BaseFFCore):
    def __init__(self):
        super().__init__(input_fields={"Ra", "Za", "batch_seg"}, output_fields={"E", "Fa"})
        pass