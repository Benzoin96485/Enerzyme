"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     tensor_product.py
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
from typing import List, Optional, Dict, Tuple, Union

import torch
import torch.nn as nn

import torch.fx
import opt_einsum_fx

from ..utils.o3 import get_slices, get_shapes


L_MAX = 3
BATCH_SIZE = 10


class Base(nn.Module):
    """Base class to handle irreducible Cartesian tensors and their products.
    
    Args:
        l1 (int): Rank of the first input irreducible Cartesian tensor.
        l2 (int): Rank of the second input irreducible Cartesian tensor.
        l3 (int): Rank of the output irreducible Cartesian tensor.
        
    Note: We add the "+1", i.e., l1+1 in the following to avoid collisions.
    """
    def __init__(self,
                 l1: int, 
                 l2: int,
                 l3: int):
        super(Base, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def key_x(self) -> int:
        """
        
        Returns:
            int: Key for the tensor of rank l1 from the first input tensor x.
        """
        return (self.l1 + 1) * 100
    
    def key_y(self) -> int:
        """
        
        Returns:
            int: Key for the tensor of rank l2 from the second input tensor y. 
        """
        return (self.l2 + 1) * 10
    
    def key_z(self) -> int:
        """
        
        Returns:
            int: Key for the input tensor of rank l3-2 obtained by computing the tensor product 
                 between x and y tensors of ranks l1 and l2, respectively.
        """
        return self.key_x() + self.key_y() + (self.l3 + 1 - 2)
    
    def key(self) -> int:
        """
        
        Returns:
            int: Key for the output tensor obtained by computing the tensor product between x 
                 and y tensors of ranks l1 and l2, respectively. It can also be used to 
                 provide a key for, e.g., x if l2 and l3 are set to -1.
        """
        return self.key_x() + self.key_y() + (self.l3 + 1)


class Slicing(Base):
    """Provides slices from input tensors corresponding to the tensors of a specific rank l.
    
    Args:
        l1 (int): Rank of the first input irreducible Cartesian tensor.
        l2 (int): Rank of the second input irreducible Cartesian tensor.
        in_slices (List[int]): Slices for the corresponding tensor.
        in_shapes (List[int]): Shape of the corresponding tensor.
    """
    def __init__(self, 
                 l1: int,
                 l2: int, 
                 in_slices: List[int], 
                 in_shapes: List[int]):
        super(Slicing, self).__init__(l1, l2, -1)
        self.start, self.stop = in_slices[:2]
        self.step = in_slices[2] or 1
        self.in_shapes = [-1] + in_shapes


class SlicingX(Slicing):
    """Get the corresponding slice from the first input tensor."""
    def __init__(self, 
                 l1: int, 
                 in_slices: List[int], 
                 in_shapes: List[int]):
        super(SlicingX, self).__init__(l1, -1, in_slices, in_shapes)
    
    def forward(self, 
                x: torch.Tensor, 
                y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        
        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.
            
        Returns:
            torch.Tensor: Slice from the first input tensor corresponding to a tensor of rank l1.
        """
        return x[:, self.start:self.stop:self.step].view(self.in_shapes)
	
class SlicingY(Slicing):
    """Get the corresponding slice from the second input tensor."""
    def __init__(self, 
                 l2: int, 
                 in_slices: List[int], 
                 in_shapes: List[int]):
        super(SlicingY, self).__init__(-1, l2, in_slices, in_shapes)
        
    def forward(self, 
                x: Optional[torch.Tensor], 
                y: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.
            
        Returns:
            torch.Tensor: Slice from the second input tensor corresponding to a tensor of rank l2.
        """
        return y[:, self.start:self.stop:self.step].view(self.in_shapes)


class PlainTensorProductBlock(Base):
    """Basic block for computing the irreducible Cartesian product without learnable weights. 
    This block computes a product between two input tensors which results in a traceless tensors 
    without any further computational steps.
    
    Args:
        einsum_subscripts (str): Specifies the subscripts for the Einstein summation.
        in1_features (int): Number of features for the first input tensor.
        in2_features (int): Number of features for the second input tensor.
        out_features (int): Number of features for the output tensor.
        in1_paths (List[int]): Number of contraction paths used to obtain the first input tensor.
        in2_paths (List[int]): Number of contraction paths used to obtain the second input tensor.
        l1 (int): Rank of the first input irreducible Cartesian tensor.
        l2 (int): Rank of the second input irreducible Cartesian tensor.
        l3 (int): Rank of the output irreducible Cartesian tensor.
    """
    def __init__(self,
                 einsum_subscripts: str,
                 in1_features: int,
                 in2_features: int,
                 out_features: int,
                 in1_paths: List[int],
                 in2_paths: List[int],
                 l1: int, 
                 l2: int, 
                 l3: int):
        super(PlainTensorProductBlock, self).__init__(l1, l2, l3)
        # trace and optimize contraction
        contraction_tr = torch.fx.symbolic_trace(lambda x, y: torch.einsum(einsum_subscripts, x, y))
        self.contraction = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                               example_inputs=(torch.randn([BATCH_SIZE] + [3] * l1 + [in1_features, in1_paths]),
                                                                               torch.randn([BATCH_SIZE] + [3] * l2 + [in2_features, in2_paths])))
        
        self.shape = [-1] + [3] * l3 + [out_features, in1_paths * in2_paths]
        
    def get(self, tensors: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Input tensors correspodning to l1, l2, and the z tensor of rank l3 - 2 (None in this case).
        """
        return tensors[self.key_x()], tensors[self.key_y()], None
        
    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                z: Optional[torch.Tensor], 
                eye: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.
            z (torch.Tensor, optional): Third input tensor (not used in this block).
            eye (torch.Tensor, optional): Identity matrix (not used in this block).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of the tensor product, both unnormalized and normalized, respectively.      
        """
        out = self.contraction(x, y).view(self.shape)
        return out, _norm_l1l2l3(self.l1, self.l2, self.l3) * out


class PlainTensorProductBlock2(Base):
    """Basic block for computing the irreducible Cartesian product without learnable weights. 
    This block computes a product between two input tensors which is not traceless. To remove 
    the trace we use the tensor z and the identity matrix eye. 
    
    Note: This is specific for `out_l = 2` in this version."""
    def __init__(self,
                 einsum_subscripts1: str,
                 einsum_subscripts2: str, 
                 eye: torch.Tensor,
                 in1_features: int,
                 in2_features: int,
                 out_features: int,
                 in1_paths: List[int],
                 in2_paths: List[int],
                 l1: int, 
                 l2: int, 
                 l3: int):
        super(PlainTensorProductBlock2, self).__init__(l1, l2, l3)
        # trace and optimize contractions
        # first contraction
        contraction_tr = torch.fx.symbolic_trace(lambda x, y: torch.einsum(einsum_subscripts1, x, y))
        self.contraction1 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(torch.randn([BATCH_SIZE] + [3] * l1 + [in1_features, in1_paths]),
                                                                                torch.randn([BATCH_SIZE] + [3] * l2 + [in2_features, in2_paths])))
        # second contraction
        contraction_tr = torch.fx.symbolic_trace(lambda x, y: torch.einsum(einsum_subscripts2, x, y))
        self.contraction2 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(torch.randn(BATCH_SIZE, out_features, in1_paths * in2_paths), eye))
        
        self.shape = [-1, 3, 3, out_features, in1_paths * in2_paths]
        
    def get(self, tensors: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Input tensors correspodning to l1, l2, and the z tensor of rank l3 - 2.
        """
        return tensors[self.key_x()], tensors[self.key_y()], tensors[self.key_z()]
        
    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                z: torch.Tensor, 
                eye: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        
        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.
            z (torch.Tensor, optional): Third input tensor.
            eye (torch.Tensor, optional): Identity matrix.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of the tensor product, both unnormalized (None in this case) and normalized, respectively.    
        """
        x1	= self.contraction1(x, y).view(self.shape)
        x1	= x1 + x1.permute(0, 2, 1, 3, 4)
        x2	= self.contraction2(z, eye)
        # we don't need unnormalized tensor products from this block
        return None, _norm_l1l2l3(self.l1, self.l2, self.l3) * (x1 - 2. / 3. * x2)


class PlainTensorProductBlock3(Base):
    """Similar to `PlainTensorProductBlock2` this class computes tensor products for `out_l = 3` and removes the trace."""
    def __init__(self,
                 einsum_subscripts1: str,
                 einsum_subscripts2: str, 
                 eye: torch.Tensor,
                 in1_features: int,
                 in2_features: int,
                 out_features: int,
                 in1_paths: List[int],
                 in2_paths: List[int],
                 l1: int, 
                 l2: int, 
                 l3: int):
        super(PlainTensorProductBlock3, self).__init__(l1, l2, l3)
        # trace and optimize contractions
        # first contraction
        contraction_tr = torch.fx.symbolic_trace(lambda x, y: torch.einsum(einsum_subscripts1, x, y))
        self.contraction1 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(torch.randn([BATCH_SIZE] + ([3] * l1) + [in1_features, in1_paths]),
                                                                                torch.randn([BATCH_SIZE] + ([3] * l2) + [in2_features, in2_paths])))
        # second contraction
        contraction_tr = torch.fx.symbolic_trace(lambda x, y: torch.einsum(einsum_subscripts2, x, y))
        self.contraction2 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(torch.randn(BATCH_SIZE, 3, out_features, in1_paths * in2_paths), eye))
        
        self.shape = [-1, 3, 3, 3, out_features, in1_paths * in2_paths]
        
    def get(self, tensors: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tensors[self.key_x()], tensors[self.key_y()], tensors[self.key_z()]
    
    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                z: torch.Tensor, 
                eye: torch.Tensor) ->  Tuple[Optional[torch.Tensor], torch.Tensor]:
        x1 = self.contraction1(x, y).view(self.shape)
        x1 = x1 + x1.permute(0, 2, 3, 1, 4, 5) + x1.permute(0, 3, 1, 2, 4, 5)
        x2	= self.contraction2(z, eye)
        x2	= x2 + x2.permute(0, 2, 3, 1, 4, 5) + x2.permute(0, 3, 1, 2, 4, 5)
        # we don't need unnormalized tensor products from this block
        return None, _norm_l1l2l3(self.l1, self.l2, self.l3) * (x1 - 2. / 5. * x2)


class PlainTensorProductModule(nn.Module):
    """Base class for computing all tensor products between l1 and l2 rank tensors leading to a tensor of rank `out_l`.
    
    Args:
        out_l (int): Rank of the output tensors.
    """
    def __init__(self, out_l: int):
        super(PlainTensorProductModule, self).__init__()
        self.blocks = nn.ModuleList()
        self.out_l	= out_l

    def append(self, block: Union[PlainTensorProductBlock, PlainTensorProductBlock2, PlainTensorProductBlock3]):
        """Appends tensor product blocks.
        
        Args:
            block (Union[PlainTensorProductBlock, PlainTensorProductBlock2, PlainTensorProductBlock3]): Tensor product block.
        """
        self.blocks.append(block)

    def forward(self,
                tensors: Dict[int, torch.Tensor], 
                eye: torch.Tensor, 
                out_features: int) -> torch.Tensor:
        """Evaluates tensor products between l1 and l2 rank tensors leading to a tensor of rank `out_l`.
        
        Args:
            tensors (Dict[int, torch.Tensor]): Dictionary containing input irreducible Cartesian tensors.
            eye (torch.Tensor): Identity matrix.
            out_features: Number of output features. TODO: Can be moved into `__init__`.
        """
        out = []
        for block in self.blocks:
            x, y, z	= block.get(tensors)
            xyze, norm_xyze = block(x, y, z, eye)
            if xyze is not None:
                tensors[block.key()] = xyze
            out.append(norm_xyze)
		
        if len(out) == 0: # append zeros if out_l_max > 0 is requested
            tmp = tensors[100]
            out.append(torch.zeros([tmp.size(0)] + [3] * self.out_l + [out_features], device=tmp.device, dtype=tmp.dtype))
		
        return torch.flatten(torch.cat(out, dim=-1), 1)


class PlainTensorProduct(nn.Module):
    """Basic class representing irreducible tensor product without learnable weights. This class allows contracting tensors 
    obtained using varius contraction paths in preceding steps.

    Args:
        in1_l_max (int): Maximal rotational order/rank of the first input tensor.
        in2_l_max (int): Maximal rotational order/rank of the second input tensor.
        out_l_max (int): Maximal rotational order/rank of the output tensor.
        in1_features (int): Number of features for the first input tensor.
        in2_features (int): Number of features for the second input tensor.
        out_features (int): Number of features for the output tensor.
        in1_paths (Optional[List[int]], optional): Number of contraction paths used to obtain the first input tensor. 
                                                   Defaults to None.
        in2_paths (Optional[List[int]], optional): Number of contraction paths used to obtain the second input tensor. 
                                                   Defaults to None.
        symmetric_product (bool, optional): If True, skip the calculation of symmetric contractions. Defaults to False.
    """
    def __init__(self,
                 in1_l_max: int,
                 in2_l_max: int,
                 out_l_max: int,
                 in1_features: int, 
                 in2_features: int,
                 out_features: int,
                 in1_paths: Optional[List[int]] = None,
                 in2_paths: Optional[List[int]] = None,
                 symmetric_product: bool = False):
        super(PlainTensorProduct, self).__init__()
        self.in1_l_max = in1_l_max
        self.in2_l_max = in2_l_max
        self.out_l_max = out_l_max
        self.out_features = out_features
        self.symmetric_product = symmetric_product
        
        # check the number of features in the input and output tensors
        if in1_features != in2_features or in1_features != out_features:
            raise RuntimeError('The number of input and output features has to be the same.')
        
        # define the number of paths resulted in irreducible tensors in the first and the second input tensor
        if in1_paths is None: in1_paths = [1 for _ in range(in1_l_max + 1)]
        if in2_paths is None: in2_paths = [1 for _ in range(in2_l_max + 1)]
        
        if self.out_l_max > L_MAX or self.in1_l_max > L_MAX or self.in2_l_max > L_MAX:
            raise RuntimeError(f'Tensor product is implemented for l <= {L_MAX=}.')
        
        # slices and shapes for tensors of rank l in flattened input tensors
        in1_slices = get_slices(in1_l_max, in1_features, in1_paths)
        in2_slices = get_slices(in2_l_max, in2_features, in2_paths)
        in1_shapes = get_shapes(in1_l_max, in1_features, in1_paths, use_prod=False)
        in2_shapes = get_shapes(in2_l_max, in2_features, in2_paths, use_prod=False)
        
        # dimensions of the input tensors for sanity checks
        self.in1_dim = sum([(3 ** l) * in1_features * in1_paths[l] for l in range(in1_l_max + 1)])
        self.in2_dim = sum([(3 ** l) * in2_features * in2_paths[l] for l in range(in2_l_max + 1)])
        
        # define the number of paths (total and specific for l <= out_l_max)
        self.n_total_paths = _get_n_paths(in1_l_max, in2_l_max, out_l_max, in1_paths=in1_paths, 
                                          in2_paths=in2_paths, symmetric_product=symmetric_product)
        self.n_paths = []
        for l in range(out_l_max + 1):
            n_paths = _get_n_paths(in1_l_max, in2_l_max, l, in1_paths=in1_paths, 
                                   in2_paths=in2_paths, symmetric_product=symmetric_product)
            self.n_paths.append(n_paths - sum(self.n_paths))
        
        # correct the number of paths for the case where out_l_max cannot be obtained by the tensor product 
        if out_l_max > 0 and self.n_paths[1] == 0: self.n_paths[1] = 1
        if out_l_max > 1 and self.n_paths[2] == 0: self.n_paths[2] = 1
        if out_l_max > 2 and self.n_paths[3] == 0: self.n_paths[3] = 1

        # define identity matrix
        self.register_buffer('eye', torch.eye(3))
        
        # define blocks
        def add_slicing_x(lst, idx): lst.append(SlicingX(idx, in1_slices[idx], in1_shapes[idx]))
        def add_slicing_y(lst, idx): lst.append(SlicingY(idx, in2_slices[idx], in2_shapes[idx]))
        
        def add_block(lst, einsum_subscripts, l1, l2, l3):
            lst.append(PlainTensorProductBlock(einsum_subscripts, in1_features, in2_features, out_features,
                                               in1_paths[l1], in2_paths[l2], l1, l2, l3))
            
        def add_block2(lst, einsum_subscripts1, einsum_subscripts2, l1, l2, l3):
            lst.append(PlainTensorProductBlock2(einsum_subscripts1, einsum_subscripts2, self.eye, in1_features, 
                                                in2_features, out_features, in1_paths[l1], in2_paths[l2], 
                                                l1, l2, l3))
        
        def add_block3(lst, einsum_subscripts1, einsum_subscripts2, l1, l2, l3):
            lst.append(PlainTensorProductBlock3(einsum_subscripts1, einsum_subscripts2, self.eye, in1_features, 
                                                in2_features, out_features, in1_paths[l1], in2_paths[l2], 
                                                l1, l2, l3))

        self.slicing = nn.ModuleList()
        
        add_slicing_x(self.slicing, 0)
        add_slicing_y(self.slicing, 0)
        if self.in1_l_max > 0:	add_slicing_x(self.slicing, 1)
        if self.in2_l_max > 0:	add_slicing_y(self.slicing, 1)
        if self.in1_l_max > 1:	add_slicing_x(self.slicing, 2)
        if self.in2_l_max > 1:	add_slicing_y(self.slicing, 2)
        if self.in1_l_max > 2:	add_slicing_x(self.slicing, 3)
        if self.in2_l_max > 2:	add_slicing_y(self.slicing, 3)
        
        self.cps = nn.ModuleList()
        
        # l = 0, shape: n_neighbors x n_feats x n_paths x n_paths
        cp_0 = PlainTensorProductModule(0)
        self.cps.append(cp_0)
        
        add_block(cp_0, 'aup, aur -> aupr', 0, 0, 0)
        if self.in1_l_max > 0 and self.in2_l_max > 0: add_block(cp_0, 'aiup, aiur -> aupr', 1, 1, 0)
        if self.in1_l_max > 1 and self.in2_l_max > 1: add_block(cp_0, 'aijup, aijur -> aupr', 2, 2, 0)
        if self.in1_l_max > 2 and self.in2_l_max > 2: add_block(cp_0, 'aijkup, aijkur -> aupr', 3, 3, 0)
        
        # l = 1, shape: n_neighbors x 3 x n_feats x n_paths x n_paths
        if self.out_l_max > 0:
            cp_1 = PlainTensorProductModule(1)
            self.cps.append(cp_1)
            
            if self.in1_l_max > 0: add_block(cp_1, 'aiup, aur -> aiupr', 1, 0, 1)
            if self.in1_l_max > 1 and self.in2_l_max > 0: add_block(cp_1, 'aijup, ajur -> aiupr', 2, 1, 1)
            if self.in1_l_max > 2 and self.in2_l_max > 1: add_block(cp_1, 'aijkup, ajkur -> aiupr', 3, 2, 1)
            if not symmetric_product:
                if self.in2_l_max > 0: add_block(cp_1, 'aup, aiur -> aiupr', 0, 1, 1)
                if self.in1_l_max > 0 and self.in2_l_max > 1: add_block(cp_1, 'aiup, aijur -> ajupr', 1, 2, 1)
                if self.in1_l_max > 1 and self.in2_l_max > 2: add_block(cp_1, 'aijup, aijkur -> akupr', 2, 3, 1)
        
        # l = 2, shape: n_neighbors x 3 x 3 x n_feats x n_paths x n_paths
        if self.out_l_max > 1:
            cp_2 = PlainTensorProductModule(2)
            self.cps.append(cp_2)
            
            if self.in1_l_max > 1: add_block(cp_2, 'aijup, aur -> aijupr', 2, 0, 2)
            if self.in1_l_max > 0 and self.in2_l_max > 0: add_block2(cp_2, 'aiup, ajur -> aijupr', 'aup, ij -> aijup', 1, 1, 2)
            if self.in1_l_max > 1 and self.in2_l_max > 1: add_block2(cp_2, 'aijup, ajkur -> aikupr', 'aup, ij -> aijup', 2, 2, 2)
            if self.in1_l_max > 2 and self.in2_l_max > 2: add_block2(cp_2, 'aijkup, ajklur -> ailupr', 'aup, ij -> aijup', 3, 3, 2)
            if self.in1_l_max > 2 and self.in2_l_max > 0: add_block (cp_2, 'aijkup, akur -> aijupr', 3, 1, 2)
            if not symmetric_product:
                if self.in2_l_max > 1: add_block(cp_2, 'aup, aijur -> aijupr', 0, 2, 2)
                if self.in1_l_max > 0 and self.in2_l_max > 2: add_block(cp_2, 'aiup, aijkur -> ajkupr', 1, 3, 2)
		
        # l = 3, shape: n_neighbors x 3 x 3 x 3 x n_feats x n_paths x n_paths
        if self.out_l_max > 2:
            cp_3 = PlainTensorProductModule(3)
            self.cps.append(cp_3)
            
            if self.in1_l_max > 2: add_block (cp_3, 'aijkup, aur -> aijkupr', 3, 0, 3)
            if self.in1_l_max > 1 and self.in2_l_max > 0: add_block3(cp_3, 'aijup, akur -> aijkupr', 'aiup, jk -> aijkup', 2, 1, 3)			
            if self.in1_l_max > 2 and self.in2_l_max > 1: add_block3(cp_3, 'aijkup, aklur -> aijlupr', 'aiup, jk -> aijkup', 3, 2, 3)
            if not symmetric_product:
                if self.in2_l_max > 2: add_block (cp_3, 'aup, aijkur -> aijkupr', 0, 3, 3)
                if self.in1_l_max > 0 and self.in2_l_max > 1: add_block3(cp_3, 'aiup, ajkur -> aijkupr', 'aiup, jk -> aijkup', 1, 2, 3)
                if self.in1_l_max > 1 and self.in2_l_max > 2: add_block3(cp_3, 'aijup, ajklur -> aiklupr', 'aiup, jk -> aijkup', 2, 3, 3)
	
    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor) -> torch.Tensor:
        """Computes tensor products/contractions between the two input tensors.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        torch._assert(x.shape[-1] == self.in1_dim, 'Incorrect last dimension for x.')
        torch._assert(y.shape[-1] == self.in2_dim, 'Incorrect last dimension for y.')
        
        torch._assert(not self.symmetric_product or torch.equal(x, y), 'Symmetric product is possible only if x == y.')
        
        # slicing
        tensors: Dict[int, torch.Tensor] = {}
        for block in self.slicing:
            tensors[block.key()] = block(x, y)

        # contractions
        out: List[torch.Tensor] = []
        for cp in self.cps:
            out.append(cp(tensors, self.eye, self.out_features))
            
        return torch.cat(out, -1)
            
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} ({self.in1_l_max} x {self.in2_l_max} -> {self.out_l_max} | {self.n_total_paths} total paths | {self.n_paths} paths)")


class WeightedTensorProductBlock(Base):
    """Basic block for computing the irreducible Cartesian product with learnable weights. 
    This block computes a product between two input tensors which results in a traceless tensors 
    without any further computational steps. Here, we directly contract tensor products with 
    weigths as these are not used in any further tensor products.
    
    Args:
        einsum_subscripts (str): Specifies the subscripts for the Einstein summation.
        example_weight (torch.Tensor): Example weights to optimize contractions.
        in1_features (int): Number of features for the first input tensor.
        in2_features (int): Number of features for the second input tensor.
        l1 (int): Rank of the first input irreducible Cartesian tensor.
        l2 (int): Rank of the second input irreducible Cartesian tensor.
        l3 (int): Rank of the output irreducible Cartesian tensor.
    """
    def __init__(self,
                 einsum_subscripts: str, 
                 example_weight: torch.Tensor,
                 in1_features: int, 
                 in2_features: int,
                 l1: int, 
                 l2: int, 
                 l3: int):
        super(WeightedTensorProductBlock, self).__init__(l1, l2, l3)
        # trace and optimize contraction
        # first contraction (weight x tensor x tensor)
        contraction_tr = torch.fx.symbolic_trace(lambda w, x, y: torch.einsum(einsum_subscripts, w, x, y))
        self.contraction = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                               example_inputs=(example_weight,
                                                                               torch.randn([BATCH_SIZE] + [3] * l1 + [in1_features]),
                                                                               torch.randn([BATCH_SIZE] + [3] * l2 + [in2_features])))

    def get(self, tensors: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return tensors[self.key_x()], tensors[self.key_y()], None

    def forward(self, 
                w: torch.Tensor, 
                a: float, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                z: Optional[torch.Tensor], 
                eye: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # we don't produce an output without contracting the products with weights
        return None, _norm_l1l2l3(self.l1, self.l2, self.l3) * self.contraction(w * a, x, y)


class WeightedTensorProductBlock2(Base):
    """Basic block for computing the irreducible Cartesian product with learnable weights. 
    This block computes a product between two input tensors which results in a traceless tensors 
    without any further computational steps. Different from `WeightedTensorProductBlock` 
    contraction with the weights computed separately to re-use uncontracted tensors in 
    further tensor products.
    
    Args:
        einsum_subscripts1 (str): Specifies the subscripts for the Einstein summation of tensors x and y.
        einsum_subscripts2 (str): Specifies the subscripts for the Einstein summation of the result of the 
                                  first contraction and the learnable weights.
        example_weight (torch.Tensor): Example weights to optimize contractions.
        in1_features (int): Number of features for the first input tensor.
        in2_features (int): Number of features for the second input tensor.
        l1 (int): Rank of the first input irreducible Cartesian tensor.
        l2 (int): Rank of the second input irreducible Cartesian tensor.
        l3 (int): Rank of the output irreducible Cartesian tensor.
    """
    def __init__(self,
                 einsum_subscripts1: str,
                 einsum_subscripts2: str,
                 example_weight: torch.Tensor,
                 in1_features: int,
                 in2_features: int,
                 l1: int, 
                 l2: int, 
                 l3: int):
        super(WeightedTensorProductBlock2, self).__init__(l1, l2, l3)
        # trace and optimize contractions
        # first contraction (tensor x tensor)
        contraction_tr = torch.fx.symbolic_trace(lambda x, y: torch.einsum(einsum_subscripts1, x, y))
        self.contraction1 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(torch.randn([BATCH_SIZE] + [3] * l1 + [in1_features]),
                                                                                torch.randn([BATCH_SIZE] + [3] * l2 + [in2_features])))
        # second contraction (weight x tensor)
        contraction_tr = torch.fx.symbolic_trace(lambda w, x: torch.einsum(einsum_subscripts2, w, x))
        self.contraction2 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(example_weight,
                                                                                torch.randn([BATCH_SIZE] + [3] * l3 + [in1_features, in2_features])))

    def get(self, tensors: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return tensors[self.key_x()], tensors[self.key_y()], None

    def forward(self, 
                w: torch.Tensor, 
                a: float, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                z: Optional[torch.Tensor], 
                eye) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.contraction1(x, y)
        # this is the only method for the weighted tensor product from which we need the results of 
        # the tensor product, not contracted with the weights
        return out, _norm_l1l2l3(self.l1, self.l2, self.l3) * self.contraction2(w * a, out)


class WeightedTensorProductBlock3(Base):
    """This block computes the irreducible Cartesian product with learnable weights but the output of 
    the contraction is not traceless. We remove the trace by usin the tensor z and the identity 
    matrix eye.
    
    Note: This is specific for `out_l = 2` in this version."""
    def __init__(self,
                 einsum_subscripts1: str,
                 einsum_subscripts2: str,
                 example_weight: torch.Tensor,
                 eye: torch.Tensor,
                 in1_features: int,
                 in2_features: int,
                 l1: int, 
                 l2: int, 
                 l3: int):
        super(WeightedTensorProductBlock3, self).__init__(l1, l2, l3)
        # trace and optimize contractions
        # first contraction (weight x tensor x tensor)
        contraction_tr = torch.fx.symbolic_trace(lambda w, x, y: torch.einsum(einsum_subscripts1, w, x, y))
        self.contraction1 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(example_weight,
                                                                                torch.randn([BATCH_SIZE] + [3] * l1 + [in1_features]),
                                                                                torch.randn([BATCH_SIZE] + [3] * l2 + [in2_features])))
		# second contraction (weight x tensor x tensor)
        contraction_tr = torch.fx.symbolic_trace(lambda w, x, e: torch.einsum(einsum_subscripts2, w, x, e))
        self.contraction2 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(example_weight, 
                                                                                torch.randn(BATCH_SIZE, in1_features, in2_features),
                                                                                eye))

    def get(self, tensors: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tensors[self.key_x()], tensors[self.key_y()], tensors[self.key_z()]

    def forward(self, 
                w: torch.Tensor, 
                a: float, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                z: torch.Tensor, 
                eye: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        wa = w * a
        waxy = self.contraction1(wa, x, y)
        waxy = waxy + waxy.permute(0, 2, 1, 3)
        waze = self.contraction2(wa, z, eye)
        # we don't produce an output without contracting the products with weights
        return None, _norm_l1l2l3(self.l1, self.l2, self.l3) * (waxy - 2. / 3. * waze)


class WeightedTensorProductBlock4(Base):
    """This block computes the irreducible Cartesian product with learnable weights but the output of 
    the contraction is not traceless. We remove the trace by usin the tensor z and the identity 
    matrix eye.
    
    Note: This is specific for `out_l = 3` in this version."""
    def __init__(self,
                 einsum_subscripts1: str,
                 einsum_subscripts2: str,
                 example_weight: torch.Tensor,
                 eye: torch.Tensor,
                 in1_features: int,
                 in2_features: int,
                 l1: int, 
                 l2: int, 
                 l3: int):
        super(WeightedTensorProductBlock4, self).__init__(l1, l2, l3)
        # trace and optimize contractions
        # first contraction (weight x tensor x tensor)
        contraction_tr = torch.fx.symbolic_trace(lambda w, x, y: torch.einsum(einsum_subscripts1, w, x, y))
        self.contraction1 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(example_weight,
                                                                                torch.randn([BATCH_SIZE] + [3] * l1 + [in1_features]),
                                                                                torch.randn([BATCH_SIZE] + [3] * l2 + [in2_features])))
		# second contraction (weight x tensor x tensor)
        contraction_tr = torch.fx.symbolic_trace(lambda w, x, e: torch.einsum(einsum_subscripts2, w, x, e))
        self.contraction2 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr, 
                                                                example_inputs=(example_weight,
                                                                                torch.randn(BATCH_SIZE, 3, in1_features, in2_features),
                                                                                eye))

    def get(self, tensors: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tensors[self.key_x()], tensors[self.key_y()], tensors[self.key_z()]

    def forward(self, w: torch.Tensor, a: float, x, y, z, eye):
        wa = w * a
        waxy = self.contraction1(wa, x, y)
        waxy = waxy + waxy.permute(0, 2, 3, 1, 4) + waxy.permute(0, 3, 1, 2, 4)
        waze =  self.contraction2(wa, z, eye)
        waze = waze + waze.permute(0, 2, 3, 1, 4) + waze.permute(0, 3, 1, 2, 4)
        return None, _norm_l1l2l3(self.l1, self.l2, self.l3) * (waxy - 2. / 5. * waze)


class WeightedTensorProductModule(nn.Module):
    """Base class for computing all tensor products between l1 and l2 rank tensors (with learnable weights) 
    leading to a tensor of rank `out_l`.
    
    Args:
        out_l (int): Rank of the output tensors.
    """
    def __init__(self, out_l):
        super(WeightedTensorProductModule, self).__init__()
        self.blocks = nn.ModuleList()
        self.out_l	= out_l

    def append(self, block: Union[WeightedTensorProductBlock, WeightedTensorProductBlock2, WeightedTensorProductBlock3, WeightedTensorProductBlock4]):
        """Appends tensor product blocks.
        
        Args:
            block (Union[WeightedTensorProductBlock, WeightedTensorProductBlock2, 
                         WeightedTensorProductBlock3, WeightedTensorProductBlock4]): Tensor product block.
        """
        self.blocks.append(block)

    def forward(self,
                tensors: Dict[int, torch.Tensor], 
                weight: List[torch.Tensor], 
                alpha: List[float], 
                eye: torch.Tensor, 
                out_features: int) -> torch.Tensor:
        """Evaluates tensor products between l1 and l2 rank tensors leading to a tensor of rank `out_l`.
        
        Args:
            tensors (Dict[int, torch.Tensor]): Dictionary containing input irreducible Cartesian tensors.
            weight (List[torch.Tensor]): List of weights for the tensor products.
            alpha (List[float]): List of scale factors for the corresponding weights.
            eye (torch.Tensor): Identity matrix.
            out_features: Number of output features. TODO: Can be moved into `__init__`.
        """
        torch._assert(len(weight) == len(self.blocks), 'Weight must have the same dimension as the number of blocks!')
        torch._assert(len(alpha) == len(self.blocks), 'Scale factors must have the same dimension as the number of blocks!')
				
        out = []
        for i, block in enumerate(self.blocks):
            x, y, z	= block.get(tensors)
            xyze, norm_waxyze = block(weight[i], alpha[i], x, y, z, eye)
            if xyze is not None:
                tensors[block.key()] = xyze
            out.append(norm_waxyze)
		
        if len(out) == 0: # append zeros if out_l_max > 0 is requested
            tmp = tensors[100]
            out.append(torch.zeros([tmp.size(0)] + [3] * self.out_l + [out_features], device=tmp.device, dtype=tmp.dtype))

        return torch.flatten(torch.cat(out, dim=-1), 1)


class WeightedTensorProduct(nn.Module):
    """Basic class representing irreducible tensor product with learnable weights. 

    Args:
        in1_l_max (int): Maximal rotational order/rank of the first input tensor.
        in2_l_max (int): Maximal rotational order/rank of the second input tensor.
        out_l_max (int): Maximal rotational order/rank of the output tensor.
        in1_features (int): Number of features for the first input tensor.
        in2_features (int): Number of features for the second input tensor.
        out_features (int): Number of features for the output tensor.
        symmetric_product (bool, optional): If True, skip the calculation of symmetric contractions. Defaults to False.
        connection_mode (str, optional): Connection mode for computing the products with learnable weights. 
                                         Defaults to 'uvu'. 'uvw' and 'uvu' are the possible choises, in line with the 
                                         e3nn code (https://github.com/e3nn/e3nn).
        internal_weights (bool, optional): If True, use internal weights. Defaults to True.
        shared_weights (bool, optional): If True, share weights across the batch dimension. Defaults to True.
    """
    def __init__(self,
                 in1_l_max: int,
                 in2_l_max: int,
                 out_l_max: int,
                 in1_features: int, 
                 in2_features: int,
                 out_features: int,
                 symmetric_product: bool = False,
                 connection_mode: str = 'uvu',
                 internal_weights: bool = True,
                 shared_weights: bool = True):
        super(WeightedTensorProduct, self).__init__()
        self.in1_l_max = in1_l_max
        self.in2_l_max = in2_l_max
        self.out_l_max = out_l_max
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.symmetric_product = symmetric_product
        self.connection_mode = connection_mode
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights
        
        if connection_mode not in ['uvu', 'uvw']:
            raise RuntimeError(f'{connection_mode=} is not implemented. Use "uvu" or "uvw" instead.')
        
        if self.out_l_max > L_MAX or self.in1_l_max > L_MAX or self.in2_l_max > L_MAX:
            raise RuntimeError(f'Tensor product is implemented for l <= {L_MAX=}.')
        
        # slices and shapes for tensors of rank l in flattened input tensors
        in1_slices = get_slices(in1_l_max, in1_features)
        in2_slices = get_slices(in2_l_max, in2_features)
        in1_shapes = get_shapes(in1_l_max, in1_features)
        in2_shapes = get_shapes(in2_l_max, in2_features)
        
        # dimensions of the input tensors for sanity checks
        self.in1_dim = sum([(3 ** l) * in1_features for l in range(in1_l_max + 1)])
        self.in2_dim = sum([(3 ** l) * in2_features for l in range(in2_l_max + 1)])
        
        # define the number of paths (total and specific for l <= out_l_max)
        self.n_total_paths = _get_n_paths(in1_l_max, in2_l_max, out_l_max, symmetric_product=symmetric_product)
        self.n_paths = []
        for out_l in range(out_l_max + 1):
            self.n_paths.append(_get_n_paths(in1_l_max, in2_l_max, out_l, symmetric_product=symmetric_product) - sum(self.n_paths))
        
        # correct the number of paths for the case where out_l_max cannot be obtained by the tensor product 
        if out_l_max > 0 and self.n_paths[1] == 0:
            self.n_paths[1] = 1
        if out_l_max > 1 and self.n_paths[2] == 0:
            self.n_paths[2] = 1
        if out_l_max > 2 and self.n_paths[3] == 0:
            self.n_paths[3] = 1
        
        # define prefix and postfix for einsum
        if connection_mode == 'uvw':
            self.prefix, self.postfix = 'uvw,', 'w'
        else:
            self.prefix, self.postfix = 'uv,', 'u'
        
        # add the batch dimension
        if not self.shared_weights:
            self.prefix = 'a' + self.prefix
            
        # define normalization
        self.alpha = []
        for n_paths in self.n_paths:
            if connection_mode == 'uvw':
                self.alpha.extend([(n_paths * in1_features * in2_features) ** (-0.5)] * n_paths)
            else:
                self.alpha.extend([(n_paths * in2_features) ** (-0.5)] * n_paths)
                
        # define weight for the tensor product
        if internal_weights:
            assert self.shared_weights, 'Having internal weights impose shared weights'
            if connection_mode == 'uvw':
                self.weight = nn.ParameterList([])
                for _ in range(self.n_total_paths):
                    self.weight.append(nn.Parameter(torch.randn(in1_features, in2_features, out_features)))
            else:
                assert in1_features == out_features
                self.weight = nn.ParameterList([])
                for _ in range(self.n_total_paths):
                    self.weight.append(nn.Parameter(torch.randn(in1_features, in2_features)))
        else:
            self.register_buffer('weight', torch.Tensor())

        # define identity matrix
        self.register_buffer('eye', torch.eye(3))
        
        # define blocks
        def add_slicing_x(lst, idx): lst.append(SlicingX(idx, in1_slices[idx], in1_shapes[idx]))
        def add_slicing_y(lst, idx): lst.append(SlicingY(idx, in2_slices[idx], in2_shapes[idx]))

        self.slicing = nn.ModuleList()
        
        add_slicing_x(self.slicing, 0)
        add_slicing_y(self.slicing, 0)
        if self.in1_l_max > 0:	add_slicing_x(self.slicing, 1)
        if self.in2_l_max > 0:	add_slicing_y(self.slicing, 1)
        if self.in1_l_max > 1:	add_slicing_x(self.slicing, 2)
        if self.in2_l_max > 1:	add_slicing_y(self.slicing, 2)
        if self.in1_l_max > 2:	add_slicing_x(self.slicing, 3)
        if self.in2_l_max > 2:	add_slicing_y(self.slicing, 3)

        if self.shared_weights:
            if self.connection_mode == 'uvw':
                example_weight = torch.randn(in1_features, in2_features, out_features)
            else:
                example_weight = torch.randn(in1_features, in2_features)
        else:
            if self.connection_mode == 'uvw':
                example_weight = torch.randn(BATCH_SIZE, in1_features, in2_features, out_features)
            else:
                example_weight = torch.randn(BATCH_SIZE, in1_features, in2_features)
			
        def add_block(lst, einsum_subscripts, l1, l2, l3):
            lst.append(WeightedTensorProductBlock(einsum_subscripts, example_weight, 
                                                  in1_features, in2_features, l1, l2, l3))
			
        def add_block2(lst, einsum_subscripts1, einsum_subscripts2, l1, l2, l3):
            lst.append(WeightedTensorProductBlock2(einsum_subscripts1, einsum_subscripts2, example_weight, 
                                                   in1_features, in2_features, l1, l2, l3))
			
        def add_block3(lst, einsum_subscripts1, einsum_subscripts2, l1, l2, l3):
            lst.append(WeightedTensorProductBlock3(einsum_subscripts1, einsum_subscripts2, example_weight, 
                                                   self.eye, in1_features, in2_features, l1, l2, l3))
		
        def add_block4(lst, einsum_subscripts1, einsum_subscripts2, l1, l2, l3):
            lst.append(WeightedTensorProductBlock4(einsum_subscripts1, einsum_subscripts2, example_weight, 
                                                   self.eye, in1_features, in2_features, l1, l2, l3))

        self.cps = nn.ModuleList()

        cp_0 = WeightedTensorProductModule(0)
        self.cps.append(cp_0)
        
        add_block(cp_0, f'{self.prefix}au, av -> a{self.postfix}', 0, 0, 0)
        if self.in1_l_max > 0 and self.in2_l_max > 0: add_block2(cp_0, 'aiu, aiv -> auv', f'{self.prefix}auv -> a{self.postfix}', 1, 1, 0)
        if self.in1_l_max > 1 and self.in2_l_max > 1: add_block2(cp_0, 'aiju, aijv -> auv', f'{self.prefix}auv -> a{self.postfix}', 2, 2, 0)
        if self.in1_l_max > 2 and self.in2_l_max > 2: add_block2(cp_0, 'aijku, aijkv -> auv', f'{self.prefix}auv -> a{self.postfix}', 3, 3, 0)

        # l = 1, shape: n_neighbors x 3 x n_feats
        if self.out_l_max > 0:
            cp_1 = WeightedTensorProductModule(1)
            self.cps.append(cp_1)
            if self.in1_l_max > 0: add_block(cp_1, f'{self.prefix}aiu, av -> ai{self.postfix}', 1, 0, 1)
            if self.in1_l_max > 1 and self.in2_l_max > 0: add_block2(cp_1, 'aiju, ajv -> aiuv', f'{self.prefix}aiuv -> ai{self.postfix}', 2, 1, 1)
            if self.in1_l_max > 2 and self.in2_l_max > 1: add_block2(cp_1, 'aijku, ajkv -> aiuv', f'{self.prefix}aiuv -> ai{self.postfix}', 3, 2, 1)

            if not symmetric_product:
                if self.in2_l_max > 0: add_block(cp_1, f'{self.prefix}au, aiv -> ai{self.postfix}', 0, 1, 1)
                if self.in1_l_max > 0 and self.in2_l_max > 1: add_block2(cp_1, 'aiu,aijv -> ajuv', f'{self.prefix}aiuv -> ai{self.postfix}', 1, 2, 1)
                if self.in1_l_max > 1 and self.in2_l_max > 2: add_block2(cp_1, 'aiju,aijkv -> akuv', f'{self.prefix}aiuv -> ai{self.postfix}', 2, 3, 1)
					
        # l = 2, shape: n_neighbors x 3 x 3 x n_feats
        if self.out_l_max > 1:
            cp_2 = WeightedTensorProductModule(2)
            self.cps.append(cp_2)
            if self.in1_l_max > 1: add_block(cp_2, f'{self.prefix}aiju, av -> aij{self.postfix}', 2, 0, 2)
            if self.in1_l_max > 0 and self.in2_l_max > 0: add_block3(cp_2, f'{self.prefix}aiu, ajv -> aij{self.postfix}', f'{self.prefix}auv, ij -> aij{self.postfix}', 1, 1, 2)
            if self.in1_l_max > 1 and self.in2_l_max > 1: add_block3(cp_2, f'{self.prefix}aiju, ajkv -> aik{self.postfix}', f'{self.prefix}auv, ij -> aij{self.postfix}', 2, 2, 2)
            if self.in1_l_max > 2 and self.in2_l_max > 2: add_block3(cp_2, f'{self.prefix}aijku, ajklv -> ail{self.postfix}', f'{self.prefix}auv, ij -> aij{self.postfix}', 3, 3, 2)
            if self.in1_l_max > 2 and self.in2_l_max > 0: add_block(cp_2, f'{self.prefix}aijku, akv -> aij{self.postfix}', 3, 1, 2)

            if not symmetric_product:
                if self.in2_l_max > 1: add_block(cp_2, f'{self.prefix}au, aijv -> aij{self.postfix}', 0, 2, 2)
                if self.in1_l_max > 0 and self.in2_l_max > 2: add_block(cp_2, f'{self.prefix}aiu, aijkv -> ajk{self.postfix}', 1, 3, 2)

        # l = 3, shape: n_neighbors x 3 x 3 x 3 x n_feats
        if self.out_l_max > 2:
            cp_3 = WeightedTensorProductModule(3)
            self.cps.append(cp_3)
            if self.in1_l_max > 2: add_block(cp_3, f'{self.prefix}aijku, av -> aijk{self.postfix}', 3, 0, 3)
            if self.in1_l_max > 1 and self.in2_l_max > 0: add_block4(cp_3, f'{self.prefix}aiju, akv -> aijk{self.postfix}', f'{self.prefix}aiuv, jk -> aijk{self.postfix}', 2, 1, 3)
            if self.in1_l_max > 2 and self.in2_l_max > 1: add_block4(cp_3, f'{self.prefix}aijku, aklv -> aijl{self.postfix}', f'{self.prefix}aiuv, jk -> aijk{self.postfix}', 3, 2, 3)

            if not symmetric_product:
                if self.in2_l_max > 2: add_block(cp_3, f'{self.prefix}au, aijkv -> aijk{self.postfix}', 0, 3, 3)
                if self.in1_l_max > 0 and self.in2_l_max > 1: add_block4(cp_3, f'{self.prefix}aiu, ajkv -> aijk{self.postfix}', f'{self.prefix}aiuv, jk -> aijk{self.postfix}', 1, 2, 3)
                if self.in1_l_max > 1 and self.in2_l_max > 2: add_block4(cp_3, f'{self.prefix}aiju, ajklv -> aikl{self.postfix}', f'{self.prefix}aiuv, jk -> aijk{self.postfix}', 2, 3, 3)
    
    def _get_weights(self, weight: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Prepares weights before computing the tensor product.

        Args:
            weight (Optional[torch.Tensor], optional): Weight tensor. Defaults to None.

        Returns:
            torch.Tensor: Weight tensor.
        """
        if weight is None:
            lst = []
            if not self.internal_weights:
                raise RuntimeError('Weights must be provided if no `internal_weights` are defined.')
            for w in self.weight:
                lst.append(w)
            return lst
        else:
            if self.shared_weights:
                dim = 0
                if self.connection_mode == 'uvw':
                    assert weight.shape == (self.n_total_paths, self.in1_features, self.in2_features, self.out_features), 'Invalid weight shape.'
                else:
                    assert weight.shape == (self.n_total_paths, self.in1_features, self.in2_features), 'Invalid weight shape.'
            else:
                dim = 1
                if self.connection_mode == 'uvw':
                    assert weight.shape[1:] == (self.n_total_paths, self.in1_features, self.in2_features, self.out_features), 'Invalid weight shape.'
                    assert weight.ndim == 5, 'When shared weights is False, weights must have batch dimension.'
                else:
                    assert weight.shape[1:] == (self.n_total_paths, self.in1_features, self.in2_features), 'Invalid weight shape.'
                    assert weight.ndim == 4, 'When shared weights is False, weights must have batch dimension.'
            return list(w.squeeze(dim) for w in weight.split(1, dim=dim))
    
    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes tensor product between input tensors `x` and `y`. Both tensors must contain flattened 
        irreducible Cartesian tensors that are accessed using pre-computed slices and shapes.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.
            weight (Optional[torch.Tensor], optional): Optional external weights. Defaults to None.

        Returns:
            torch.Tensor: Irreducible Cartesian tensors obtained as products of input tensors. The number 
                          of features is larger by the number of paths.
        """
        torch._assert(x.shape[-1] == self.in1_dim, 'Incorrect last dimension for x.')
        torch._assert(y.shape[-1] == self.in2_dim, 'Incorrect last dimension for y.')
        
        torch._assert(not self.symmetric_product or torch.equal(x, y), 'Symmetric product is possible only if x == y.')
        
        # weight shape ('uvw'): n_paths x (n_neighbors x) in1_features x in2_features x out_features
        # weight shape ('uvu'): n_paths x (n_neighbors x) in1_features x in2_features
        weight = self._get_weights(weight)
        
        # slicing
        tensors: Dict[int, torch.Tensor] = {}
        for block in self.slicing:
            tensors[block.key()] = block(x, y)

        # contractions
        start = 0
        end = 0
        
        out: List[torch.Tensor] = []
        for cp in self.cps:
            end = start + len(cp.blocks)
            w = weight[start:end]
            a = self.alpha[start:end]
            start = end
            out.append(cp(tensors, w, a, self.eye, self.out_features))
            
        return torch.cat(out, -1)

    def __repr__(self) -> str:
        if self.connection_mode == 'uvw':
            weight_numel = self.n_total_paths * self.in1_features * self.in2_features * self.out_features
        else:
            weight_numel = self.n_total_paths * self.in1_features * self.in2_features
        return (f"{self.__class__.__name__} ({self.in1_l_max} x {self.in2_l_max} -> {self.out_l_max} | {self.n_total_paths} total paths | {self.n_paths} paths | {weight_numel} weights)")


def _get_n_paths(in1_l_max: int, 
                 in2_l_max: int, 
                 out_l_max: int,
                 in1_paths: Optional[List[int]] = None,
                 in2_paths: Optional[List[int]] = None,
                 symmetric_product: bool = False) -> int:
    """Counts the number of re-coupling paths for Cartesian harmonics depending on the rank of input tensors 
    and the maximal rank of the expected output tensor. 
    
    We count the number of paths obtained by re-coupling Cartesian harmonics, but also paths obtained through 
    re-coupling/contracting Cartesian harmonics in previous calculations.

    Args:
        in1_l_max (int): Maximal rotational order/rank of the first input tensor.
        in2_l_max (int): Maximal rotational order/rank of the second input tensor.
        out_l_max (int): Maximal rotational order/rank of the output tensor.
        in1_paths (Optional[List[int]], optional): Number of contraction paths used to obtain the first input tensor. 
                                                   Defaults to None.
        in2_paths (Optional[List[int]], optional): Number of contraction paths used to obtain the second input tensor. 
                                                   Defaults to None.
        symmetric_product (bool, optional): If True, skip the calculation of symmetric contractions. Defaults to False.

    Returns:
        int: Total number of contraction paths.
    """
    if in1_paths is None: in1_paths = [1 for _ in range(in1_l_max + 1)]
    if in2_paths is None: in2_paths = [1 for _ in range(in2_l_max + 1)]
    
    # input tensors have by default scalar features
    n_paths = in1_paths[0] * in2_paths[0]
    
    # count paths leading to l = 0
    if in1_l_max > 0 and in2_l_max > 0: n_paths += in1_paths[1] * in2_paths[1]
    if in1_l_max > 1 and in2_l_max > 1: n_paths += in1_paths[2] * in2_paths[2]
    if in1_l_max > 2 and in2_l_max > 2: n_paths += in1_paths[3] * in2_paths[3]
    if out_l_max == 0: return n_paths
    
    # count paths leading to l = 1
    if in1_l_max > 0: n_paths += in1_paths[1] * in2_paths[0]
    if in1_l_max > 1 and in2_l_max > 0: n_paths += in1_paths[2] * in2_paths[1]
    if in1_l_max > 2 and in2_l_max > 1: n_paths += in1_paths[3] * in2_paths[2]
    if not symmetric_product:
        if in2_l_max > 0: n_paths += in1_paths[0] * in2_paths[1]
        if in1_l_max > 0 and in2_l_max > 1: n_paths += in1_paths[1] * in2_paths[2]
        if in1_l_max > 1 and in2_l_max > 2: n_paths += in1_paths[2] * in2_paths[3]
    if out_l_max == 1: return n_paths
    
    # count paths leading to l=2
    if in1_l_max > 1: n_paths += in1_paths[2] * in2_paths[0]
    if in1_l_max > 0 and in2_l_max > 0: n_paths += in1_paths[1] * in2_paths[1]
    if in1_l_max > 1 and in2_l_max > 1: n_paths += in1_paths[2] * in2_paths[2]
    if in1_l_max > 2 and in2_l_max > 2: n_paths += in1_paths[3] * in2_paths[3]
    if in1_l_max > 2 and in2_l_max > 0: n_paths += in1_paths[3] * in2_paths[1]
    if not symmetric_product:
        if in2_l_max > 1: n_paths += in1_paths[0] * in2_paths[2]
        if in1_l_max > 0 and in2_l_max > 2: n_paths += in1_paths[1] * in2_paths[3]
    if out_l_max == 2: return n_paths
    
    # count paths leading to l=3
    if in1_l_max > 2: n_paths += in1_paths[3] * in2_paths[0]
    if in1_l_max > 1 and in2_l_max > 0: n_paths += in1_paths[2] * in2_paths[1]
    if in1_l_max > 2 and in2_l_max > 1: n_paths += in1_paths[3] * in2_paths[2]
    if not symmetric_product:
        if in2_l_max > 2: n_paths += in1_paths[0] * in2_paths[3]
        if in1_l_max > 0 and in2_l_max > 1: n_paths += in1_paths[1] * in2_paths[2]
        if in1_l_max > 1 and in2_l_max > 2: n_paths += in1_paths[2] * in2_paths[3]
    if out_l_max == 3: return n_paths


def _factorial(n: int) -> int:
    """Computes factorial.

    Args:
        n (int): Input integer.

    Returns:
        int: Output integer.
    """
    x = 1
    for _ in range(1, n+1):
        x = x * _
    return x


def _doublefactorial(n: int) -> int:
    """Computes double factorial.

    Args:
        n (int): Input integer.

    Returns:
        int: Output integer.
    """
    x = 1
    for _ in range(1, n+1, 2):
        x = x * _
    return x


def _norm_l1l2l3(l1: int,
                 l2: int,
                 l3: int) -> float:
    """Computes the normalization factor for the irreducible recoupling of Cartesian harmonics.
    This function implements the normalization factor for an even recoupling.

    Args:
        l1 (int): Rank of the first input Cartesian harmonics.
        l2 (int): Rank of the second input Cartesian harmonics.
        l3 (int): Rank of the output Cartesian harmonics.

    Returns:
        float: Normalization factor.
    """
    assert (l1 + l2 - l3) % 2 == 0
    J = l1 + l2 + l3
    J1 = J - 2 * l1 - 1
    J2 = J - 2 * l2 - 1
    J3 = J - 2 * l3 - 1
    num = _factorial(l1) * _factorial(l2) * _doublefactorial(int(2 * l3 - 1)) * _factorial(int((J1+1)/2)) * _factorial(int((J2+1)/2))
    den = _factorial(l3) * _doublefactorial(J1) * _doublefactorial(J2) * _doublefactorial(J3) * _factorial(int(J/2))
    return (num / den)
