"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     product_basis.py
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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import torch.fx
import opt_einsum_fx

from .tensor_product import PlainTensorProduct
from ..utils.o3 import get_slices, get_shapes


L_MAX = 3
BATCH_SIZE = 10


class WeightedPathSummationBlock(nn.Module):
    """Computes weighted, species-dependent summation over the separate paths for the specific rotational 
    order l of irreducible Cartesian tensors.
    
    Args:
        einsum_subscripts (str): Specifies the subscripts for the Einstein summation.
        in_slices (List[int]): Slices to get the specific irreducible Cartesian tensor of rank l.
        in_shapes (List[int]): Shape of the irreducible Cartesian tensor of rank l.
        weight (torch.Tensor): Weight of the weighted path summation.
        example_in1 (torch.Tensor): Example tensor for the first input (irreducible Cartesian tensors).
        example_in2 (torch.Tensor): Example tensor for the second input (one-hot encoded species).
    """
    def __init__(self, 
                 einsum_subscripts: str,
                 in_slices: List[int],
                 in_shapes: List[int],
                 weight: torch.Tensor,
                 example_in1: torch.Tensor,
                 example_in2: torch.Tensor):
        super(WeightedPathSummationBlock, self).__init__()
        self.start, self.stop = in_slices[:2]
        self.step = in_slices[2] or 1
        self.in_shapes	= [-1] + in_shapes
        self.weight	= weight
        
        # trace and optimize contraction
        contraction_tr = torch.fx.symbolic_trace(lambda w, x, y: torch.einsum(einsum_subscripts, w, x, y))
        self.contraction = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                               example_inputs=(weight, example_in1, example_in2))
        
    def forward(self, x, y):
        """Computes weighted sum across the contraction path.
        
        Args:
            x (torch.Tensor): First input tensor (irreducible Cartesian tensors).
            y (torch.Tensor): Second input tensor (one-hot encoded species).
        
        Returns:
            torch.Tensor: Output tensor (irreducible Cartesian tensors).
        """
        # x shape: n_neighbors x (3 x ... x l-times x ... x 3 * n_paths) x in_features x n_paths
        x = x[:, self.start:self.stop:self.step].view(self.in_shapes)
        return torch.flatten(self.contraction(self.weight, x, y), 1)


class WeightedPathSummation(nn.Module):
    """Computes weighted, species-dependent summation over the separate paths leading to the specific 
    rotational order l of Cartesian harmonics provided in the first input tensor. The second input tensor 
    is typically one-hot encoded species and is used to get species-dependent weights. The number of 
    features in the output tensor must be the same as for the first input tensor.

    Args:
        in1_l_max (int): Maximal rank of the first input tensor.
        out_l_max (int): Maximal rank of the output tensor.
        in1_features (int): Number of features in the first input tensor.
        in2_features (int): Number of features in the second input tensor.
        in1_paths (List[int], optional): Provides the number of paths used to generate Cartesian 
                                         harmonics of a particular rank provided in the first 
                                         input tensor. The weighted sum is computed across these 
                                         paths for each rank l.
        coupled_feats (bool, optional): If True, use coupled feature channels.
    """
    def __init__(self,
                 in1_l_max: int,
                 out_l_max: int,
                 in1_features: int,
                 in2_features: int,
                 in1_paths: Optional[List[int]] = None,
                 coupled_feats: bool = False):
        super(WeightedPathSummation, self).__init__()
        self.in1_l_max = in1_l_max
        self.out_l_max = out_l_max
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.in1_paths = in1_paths
        
        if self.out_l_max > L_MAX or self.in1_l_max > L_MAX:
            raise RuntimeError(f'Product basis is implemented for l <= {L_MAX=}.')
        
        # define the number of paths used to compute irreducible Cartesian tensors in the first input tensor
        if in1_paths is None:
            self.in1_paths = [1 for _ in range(in1_l_max + 1)]
        else:
            self.in1_paths = in1_paths
        assert len(self.in1_paths) == in1_l_max + 1
        
        # slices and shapes for tensors of rank l in the flattened input tensor
        self.in1_slices = get_slices(in1_l_max, in1_features, self.in1_paths)
        self.in1_shapes = get_shapes(in1_l_max, in1_features, self.in1_paths, use_prod=True if coupled_feats else False)
        
        # dimensions of the input tensors for sanity checks
        self.in1_dim = sum([(3 ** l) * in1_features * self.in1_paths[l] for l in range(in1_l_max + 1)])
        self.in2_dim = in2_features
        
        # define weight
        self.weight = nn.ParameterList([])
        for n_paths in self.in1_paths[:self.out_l_max+1]:
            if coupled_feats:
                self.weight.append(nn.Parameter(torch.randn(in2_features, in1_features * n_paths, in1_features) / n_paths / in1_features ** 0.5))
            else:
                self.weight.append(nn.Parameter(torch.randn(in2_features, in1_features, n_paths) / n_paths))
        
        # define weighted path summation blocks
        self.blocks = nn.ModuleList()
        
        def add_block(idx, einsum_subscripts):
            example_in1 = torch.randn(
                [BATCH_SIZE] + [3] * idx + ([in1_features * self.in1_paths[idx]] if coupled_feats else [in1_features, self.in1_paths[idx]])
                )
            example_in2 = torch.randn(BATCH_SIZE, in2_features)
            self.blocks.append(WeightedPathSummationBlock(einsum_subscripts, self.in1_slices[idx], self.in1_shapes[idx], self.weight[idx],
                                                          example_in1, example_in2))
            
        add_block(0, 'wvu, av, aw -> au' if coupled_feats else 'wvp, avp, aw -> av')
        if self.out_l_max > 0: add_block(1, 'wvu, aiv, aw -> aiu' if coupled_feats else 'wvp, aivp, aw -> aiv')
        if self.out_l_max > 1: add_block(2, 'wvu, aijv, aw -> aiju' if coupled_feats else 'wvp, aijvp, aw -> aijv')
        if self.out_l_max > 2: add_block(3, 'wvu, aijkv, aw -> aijku' if coupled_feats else 'wvp, aijkvp, aw -> aijkv')

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """Computes weighted sum across the contraction paths.
        
        Args:
            x (torch.Tensor): First input tensor. It contains concatenated irreducible Cartesian tesnors.
            y (torch.Tensor): Second input tensor. It contains one-hot encoded species.
        
        Returns:
            torch.Tensor: Tensor with concatenated and flattened irreducible Cartesian tesnors.
        """
        torch._assert(x.shape[-1] == self.in1_dim, 'Incorrect last dimension for x.')
        torch._assert(y.shape[-1] == self.in2_dim, 'Incorrect last dimension for y.')
        
        outputs = []
        for b in self.blocks:
            outputs.append(b(x, y))
        return torch.cat(outputs, -1)
            
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} ({self.in1_l_max} -> {self.out_l_max} | {self.in1_paths[:self.out_l_max+1]} -> {[1 for _ in range(self.out_l_max+1)]} paths | {sum([w.numel() for w in self.weight])} weights)")


class WeightedProductBasisBlock(torch.nn.Module):
    """Basic block for the weighted product basis obtained by a product of two irreducible Cartesian tensors.
    
    Args:
        tp (PlainTensorProduct): Tensor product.
        weighted_sum (WeightedPathSummation): Weighted path summation.
    """
    def __init__(self, 
                 tp: PlainTensorProduct, 
                 weighted_sum: WeightedPathSummation):
        super(WeightedProductBasisBlock, self).__init__()
        self.tp = tp
        self.weighted_sum = weighted_sum
    
    def forward(self, 
                out_tp: torch.Tensor, 
                basis: torch.Tensor, 
                x: torch.Tensor, 
                y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the weighted product basis features.
        
        Args:
            out_tp (torch.Tensor): Features obtained from the previous tensor product.
            basis (torch.Tensor): Current product basis features.
            x (torch.Tensor): First input tensor. It contains concatenated irreducible Cartesian tesnors.
            y (torch.Tensor): Second input tensor. It contains one-hot encoded species.

        Returns:
            torch.Tensor: Updated product basis features.
        """
        out_tp = self.tp(out_tp, x)
        basis = basis + self.weighted_sum(out_tp, y)
        return out_tp, basis


class WeightedProductBasis(nn.Module):
    """Weighted product basis obtained by contracting irreducible Cartesian tensors.
    
    Args:
        in1_l_max (int): Maximal rank of the first input tensor.
        out_l_max (int): Maximal rank of the output tensor.
        in1_features (int): Number of features in the first input tensor.
        in2_features (int): Number of features in the second input tensor.
        correlation (int): Correlation order, i.e., number of contracted tensors.
        coupled_feats (bool, optional): If True, use mixed features.
        symmetric_product (bool, optional): If True, exploit symmetry of the tensor product to reduce 
                                            the number of possible tensor contractions.
    """
    def __init__(self,
                 in1_l_max: int,
                 out_l_max: int,
                 in1_features: int,
                 in2_features: int,
                 correlation: int,
                 coupled_feats: bool = False,
                 symmetric_product: bool = True):
        super(WeightedProductBasis, self).__init__()
        self.correlation = correlation
        self.in1_l_max = in1_l_max
        self.out_l_max = out_l_max
        self.in1_features = in1_features
        
        # prepare tensor products for computing the product basis from the first input tensor
        # tensor products are computed only if correlation > 1
        self.weighted_sum = WeightedPathSummation(in1_l_max=in1_l_max, out_l_max=out_l_max, 
                                                  in1_features=in1_features, in2_features=in2_features,
                                                  coupled_feats=coupled_feats)

        self.blocks = nn.ModuleList()
        for i in range(self.correlation - 1):
            target_l_max = out_l_max if i == self.correlation - 2 else in1_l_max
            in1_paths = None if i == 0 else tp.n_paths
            
            tp = PlainTensorProduct(in1_l_max=in1_l_max, in2_l_max=in1_l_max, out_l_max=target_l_max,
                                    in1_features=in1_features, in2_features=in1_features, out_features=in1_features,
                                    in1_paths=in1_paths, symmetric_product=symmetric_product if i == 0 else False)
            
            ws = WeightedPathSummation(in1_l_max=target_l_max, out_l_max=out_l_max,
                                       in1_features=in1_features, in2_features=in2_features,
                                       in1_paths=tp.n_paths, coupled_feats=coupled_feats)
            
            self.blocks.append(WeightedProductBasisBlock(tp, ws))

    def forward(self, 
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """Computes the weighted product basis features.

        Args:
            x (torch.Tensor): First input tensor. It contains concatenated irreducible Cartesian tesnors.
            y (torch.Tensor): Second input tensor. It contains one-hot encoded species.

        Returns:
            torch.Tensor: Output tensor with product basis features.
        """
        # shape: n_batch x n_feats * (1 + 3 + ... + 3^l)
        # correlation = 1
        basis = self.weighted_sum(x, y)
        
        # compute tensor products for correlation > 1
        out_tp = x
        for b in self.blocks:
            out_tp, basis = b(out_tp, basis, x, y)
        return basis
