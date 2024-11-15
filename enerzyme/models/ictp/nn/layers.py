"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     layers.py
  Authors:  Viktor Zaverkin (viktor.zaverkin@neclab.eu)
            Francesco Alesiani (francesco.alesiani@neclab.eu)
            Takashi Maruyama (takashi.maruyama@neclab.eu)
            Federico Errica (federico.errica@neclab.eu)
            Henrik Christiansen (henrik.christiansen@neclab.eu)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Nicolas Weber (nicolas.weber@neclab.eu)
            Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)

NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.  

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
       PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..o3.tensor_product import WeightedTensorProduct
from ..o3.linear_transform import LinearTransform
from ..o3.product_basis import WeightedProductBasis
from .radial import BesselRBF, PolynomialCutoff
from ..utils.math import segment_sum


class RescaledSiLULayer(nn.Module):
    """Rescaled SiLU layer.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies rescaled SiLU to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the rescaled SiLU layer.
        """
        return 1.6765324703310907 * F.silu(x)


class LinearLayer(nn.Module):
    """Simple linear layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, apply bias. Defaults to False.
    """
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = False):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # define weight and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies linear layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the linear layer.
        """
        return F.linear(x, self.weight / (self.in_features) ** 0.5, self.bias)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} ({self.in_features} -> {self.out_features} | {self.weight.numel()} weights)")


class RadialEmbeddingLayer(nn.Module):
    """Non-linear embedding layer for the radial part.

    Adapted from MACE (https://github.com/ACEsuit/mace/blob/main/mace/modules/blocks.py).
    
    Args:
        r_cutoff (float): Cutoff radius.
        n_basis (int): Number of radial basis functions.
        n_polynomial_cutoff (int): Parameter `p` of the envelope function.
    """
    def __init__(self,
                 r_cutoff: float,
                 n_basis: int,
                 n_polynomial_cutoff: int):
        super(RadialEmbeddingLayer, self).__init__()
        self.bessel_fn = BesselRBF(r_cutoff=r_cutoff, n_basis=n_basis)
        self.cutoff_fn = PolynomialCutoff(r_cutoff=r_cutoff, p=n_polynomial_cutoff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the non-linear embedding layer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the radial embedding layer.
        """
        radial = self.bessel_fn(x)
        cutoff = self.cutoff_fn(x)
        return radial * cutoff
    

class ProductBasisLayer(nn.Module):
    """Equivariant product basis layer with contractions based on the tensor product between 
    irreducible Cartesian tensors.
        
    Args:
        l_max_node_feats (int): Maximal rank of the irreducible node features (node embeddings).
        l_max_target_feats (int): Maximal rank of the irreducible features (target ones, can be 
                                  different from node features).
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        n_species (int): Number of species/atom types.
        correlation (int): Correlation order, i.e., the number of contracted tensors. It also 
                           corresponds to the many-body order + 1.
        coupled_feats (bool): If True, use mix channels when computing the product basis.
        symmetric_product (bool): If True, exploit symmetry of the tensor product to reduce 
                                  the number of possible tensor contractions.
        use_sc (bool): If True, use self-connection.
    """
    def __init__(self,
                 l_max_node_feats: int,
                 l_max_target_feats: int,
                 in_features: int,
                 out_features: int,
                 n_species: int,
                 correlation: int,
                 coupled_feats: bool,
                 symmetric_product: bool,
                 use_sc: bool):
        super(ProductBasisLayer, self).__init__()
        self.use_sc = use_sc
        
        # define weighted product basis
        self.product_basis = WeightedProductBasis(in1_l_max=l_max_node_feats, out_l_max=l_max_target_feats,
                                                  in1_features=in_features, in2_features=n_species,
                                                  correlation=correlation, coupled_feats=coupled_feats,
                                                  symmetric_product=symmetric_product)
        
        # linear transform
        self.linear = LinearTransform(in_l_max=l_max_target_feats, out_l_max=l_max_target_feats, 
                                      in_features=in_features, out_features=out_features)
    
    def forward(self, 
                node_feats: torch.Tensor,
                sc: Optional[torch.Tensor],
                node_attrs: torch.Tensor) -> torch.Tensor:
        """Computes the product basis.

        Args:
            node_feats (torch.Tensor): Node features (node embeddings).
            sc (torch.Tensor): Residual connection.
            node_attrs (torch.Tensor): Node attributes, e.g., one-hot encoded species.

        Returns:
            torch.Tensor: Product basis.
        """
        product_basis = self.product_basis(node_feats, node_attrs)
        
        # use self-connection if necessary
        if self.use_sc and sc is not None:
            return self.linear(product_basis) + sc
        
        return self.linear(product_basis)


class InteractionLayer(nn.Module):
    """Equivariant interaction layer with the convolution based on the tensor product between 
    irreducible Cartesian tensors/Cartesian harmonics.

    Args:
        l_max_node_feats (int): Maximal rank of the irreducible node features (node embeddings).
        l_max_edge_attrs (int): Maximal rank of the irreducible edge attributes (Cartesian harmonics).
        l_max_target_feats (int): Maximal rank of the irreducible features (target ones, can be 
                                  different from node features).
        l_max_hidden_feats (int): Maximal rank of the irreducible hidden features (for the first 
                                  layer can be different from node features).
        n_basis (int): Number of radial basis functions.
        n_species (int): Number of species/elements.
        in_features (int): Number of input features.
        out_features (int): Number of input features.
        avg_n_neighbors (float): Average number of neighbors.
        radial_MLP (List[int]): List of hidden features for the radial embedding network.
    """
    def __init__(self,
                 l_max_node_feats: int,
                 l_max_edge_attrs: int,
                 l_max_target_feats: int,
                 l_max_hidden_feats: int,
                 n_basis: int,
                 n_species: int,
                 in_features: int,
                 out_features: int,
                 avg_n_neighbors: float,
                 radial_MLP: List[int]):
        super(InteractionLayer, self).__init__()
        self.l_max_node_feats = l_max_node_feats
        self.l_max_edge_attrs = l_max_edge_attrs
        self.l_max_target_feats = l_max_target_feats
        self.l_max_hidden_feats = l_max_hidden_feats
        self.n_basis = n_basis
        self.n_species = n_species
        self.in_features = in_features
        self.out_features = out_features
        self.avg_n_neighbors = avg_n_neighbors
        self.radial_MLP = radial_MLP
        
        self._setup()
    
    def _setup(self):
        """Setup specific to the interaction layer."""
        raise NotImplementedError()
        
    def forward(self,
                node_attrs: torch.Tensor,
                node_feats: torch.Tensor,
                edge_attrs: torch.Tensor,
                edge_feats: torch.Tensor,
                idx_i: torch.Tensor,
                idx_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the output of the interaction layer.

        Args:
            node_attrs (torch.Tensor): Node attributes, e.g., one-hot encoded species.
            node_feats (torch.Tensor): Node features (node embeddings).
            edge_attrs (torch.Tensor): Edge attributes (Cartesian harmonics).
            edge_feats (torch.Tensor): Edge features (radial basis).
            idx_i (torch.Tensor): Receivers (central nodes).
            idx_j (torch.Tensor): Senders (neighboring nodes).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Node messages and residual connections.
        """
        raise NotImplementedError()


class RealAgnosticResidualInteractionLayer(InteractionLayer):
    """Equivariant interaction layer with residual connection."""
    def _setup(self):
        # first linear transform
        self.linear_first = LinearTransform(in_l_max=self.l_max_node_feats, out_l_max=self.l_max_node_feats, 
                                            in_features=self.in_features, out_features=self.in_features)
        
        # tensor product between hidden features and Cartesian harmonics
        self.conv_tp = WeightedTensorProduct(in1_l_max=self.l_max_node_feats, in2_l_max=self.l_max_edge_attrs, out_l_max=self.l_max_target_feats,
                                             in1_features=self.in_features, in2_features=1, out_features=self.in_features,
                                             connection_mode='uvu', internal_weights=False, shared_weights=False)
        
        # convolution weights
        layers = []
        for in_size, out_size in zip([self.n_basis] + self.radial_MLP,
                                     self.radial_MLP + [self.conv_tp.n_total_paths * self.in_features]):
            layers.append(LinearLayer(in_size, out_size))
            layers.append(RescaledSiLULayer())
        self.conv_tp_weights = torch.nn.Sequential(*layers[:-1])

        # second linear layer
        self.linear_second = LinearTransform(in_l_max=self.l_max_target_feats, out_l_max=self.l_max_target_feats, 
                                             in_features=self.in_features, out_features=self.out_features,
                                             in_paths=self.conv_tp.n_paths)

        # tensor product between node features and node attributes for the residual connection
        self.skip_tp = WeightedTensorProduct(in1_l_max=self.l_max_node_feats, in2_l_max=0, out_l_max=self.l_max_hidden_feats,
                                             in1_features=self.in_features, in2_features=self.n_species, out_features=self.in_features,
                                             connection_mode='uvw', internal_weights=True, shared_weights=True)

    def forward(self,
                node_attrs: torch.Tensor,
                node_feats: torch.Tensor,
                edge_attrs: torch.Tensor,
                edge_feats: torch.Tensor,
                idx_i: torch.Tensor,
                idx_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: n_atoms x n_feats * (1 + 3 + 3^2 + ...)
        sc = self.skip_tp(node_feats, node_attrs)
        
        # shape: n_atoms x n_feats * (1 + 3 + 3^2 + ...)
        node_feats = self.linear_first(node_feats)
        
        # shape: n_neighbors x n_total_paths x n_hidden_feats x 1
        tp_weights = self.conv_tp_weights(edge_feats).view(-1, self.conv_tp.n_total_paths, self.in_features, 1)
        
        # shape: n_neighbors x n_feats * (n_paths_0 + 3 * n_paths_1 + 3^2 * n_paths_2 + ...)
        m_ij = self.conv_tp(node_feats.index_select(0, idx_j), edge_attrs, tp_weights)
        
        # shape: n_atoms x n_feats * (n_paths_0 + 3 * n_paths_1 + 3^2 * n_paths_2 + ...)
        message = segment_sum(m_ij, idx_i, node_feats.shape[0], 0)
        
        # shape: n_atoms x n_feats * (1 + 3 + 3^2 + ...)
        message = self.linear_second(message) / self.avg_n_neighbors
        
        return message, sc


class ScaleShiftLayer(nn.Module):
    """Re-scales and shifts atomic energies predicted by the model.
    
    Args:
        shift_param (float): Parameter by which atomic energies should be shifted.
        scale_param (float): Parameter by which atomic energies should be scaled.
    """
    def __init__(self,
                 shift_params: np.ndarray,
                 scale_params: np.ndarray):
        super().__init__()
        self.register_buffer("scale_params", torch.tensor(scale_params, dtype=torch.get_default_dtype()))
        self.register_buffer("shift_params", torch.tensor(shift_params, dtype=torch.get_default_dtype()))

    def forward(self, 
                x: torch.Tensor,
                graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Re-scales and shifts the ouptput of the atomistic model.

        Args:
            x (torch.Tensor): Iutput of the atomistic model.
            graph (Dict[str, torch.Tensor]): Atomic data dictionary.

        Returns:
            torch.Tensor: Re-scaled and shifted ouptput of the atomistic model.
        """
        species = graph['species']
        scale_species = self.scale_params.index_select(0, species)
        shift_species = self.shift_params.index_select(0, species)
        return scale_species * x + shift_species
    
    def __repr__(self):
        return f'{self.__class__.__name__}(scale_params={self.scale_params}({self.scale_params.dtype}), shift_params={self.shift_params}({self.shift_params.dtype}))'
