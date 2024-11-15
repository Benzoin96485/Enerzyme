"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     linear_transform.py
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
from typing import List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.o3 import get_slices, get_shapes


class LinearTransformBlock(nn.Module):
    """Basic linear transform block for an irreducible Cartesian tensor of a specific rank l. 
    The slices define the latter.
    
    Args:
        in_slices (List[int]): Slices to get the specific irreducible Cartesian tensor of rank l.
        in_shapes (List[int]): Shape of the irreducible Cartesian tensor of rank l.
        weight (torch.Tensor): Weight of the linear transformation block.
        alpha (float): Normalization for the weight of the linear transformation block.
        bias (torch.Tensor, optional): Bias of the linear transformation. Defaults to None.
    """
    def __init__(self, 
                 in_slices: List[int], 
                 in_shapes: List[int], 
                 weight: torch.Tensor, 
                 alpha: float, 
                 bias: Optional[torch.Tensor] = None):
        super(LinearTransformBlock, self).__init__()
        self.start, self.stop = in_slices[:2]
        self.step = in_slices[2] or 1
        self.in_shapes = [-1] + in_shapes
        self.weight	= weight
        self.alpha = alpha
        self.bias = bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transform to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor (contains all ranks l). The tensor of the specific rank 
                              is obtained using `in_slices`.

        Returns:
            torch.Tensor: Output irreducible Cartesian tensor of rank l after the linear transform.
        """
        # x shape: n_neighbors x (3 x ... x l-times x ... x 3) x in_features
        x = x[:, self.start:self.stop:self.step].view(self.in_shapes)
        x = F.linear(x, self.weight * self.alpha, self.bias)
        return x


class LinearTransform(nn.Module):
    """Simple linear transformation for irreducible Cartesian tensors. It preserves the properties 
    of the latter, i.e., the resulting Cartesian tensors are symmetric and traceless.

    Args:
        in_l_max (int): Maximal rank of the input tensor.
        out_l_max (int): Maximal rank of the output tensor.
        in_features (int): Numbers of features in the input tensor.
        out_features (int): Number of features in the output tensor.
        in_paths (List[int], optional): List of paths used to generate irreducible Cartesian tensors 
                                        contained in the input tensor. Defaults to None (in this case 
                                        a list of length `in_l_max`+1 filled with ones is produced).
        bias (bool, optional): If True, apply bias to scalars. Defaults to False.
    """
    def __init__(self,
                 in_l_max: int,
                 out_l_max: int,
                 in_features: int, 
                 out_features: int,
                 in_paths: Optional[List[int]] = None,
                 bias: bool = False):
        super(LinearTransform, self).__init__()
        self.in_l_max = in_l_max
        self.out_l_max = out_l_max
        self.in_features = in_features
        self.out_features = out_features
        
        # define the number of paths used to compute irreducible Cartesian tensor in the input tensor
        if in_paths is None:
            self.in_paths = [1 for _ in range(in_l_max + 1)]
        else:
            self.in_paths = in_paths
        assert len(self.in_paths) == in_l_max + 1
        
        # slices and shapes for tensors of rank l in the flattened input tensor
        in_slices = get_slices(in_l_max, in_features, self.in_paths)
        in_shapes = get_shapes(in_l_max, in_features, self.in_paths, use_prod=True)
        
        # dimensions of the input tensors for sanity checks
        self.in_dim = sum([(3 ** l) * in_features * self.in_paths[l] for l in range(in_l_max + 1)])
        
        # define normalization
        self.alpha = [(in_features * n_paths) ** (-0.5) for n_paths in self.in_paths]
        
        # define weight and bias
        self.weight = nn.ParameterList([])
        for n_paths in self.in_paths[:self.out_l_max+1]:
            self.weight.append(nn.Parameter(torch.randn(out_features, in_features * n_paths)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)
        
        # define linear transform blocks
        self.blocks = nn.ModuleList()
        for i, (w, a) in enumerate(zip(self.weight, self.alpha)):
            if i == 0:
                # we can apply bias only to scalar features
                self.blocks.append(LinearTransformBlock(in_slices[i], in_shapes[i], w, a, self.bias))
            else:
                self.blocks.append(LinearTransformBlock(in_slices[i], in_shapes[i], w, a, None))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies linear transform to the input tensor x. This tensor contains flattened, irreducible 
        Cartesian tensors that are accessed with slices and shapes.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor containing irreducible Cartesian tensors after the linear 
                          transform.
        """
        torch._assert(x.shape[-1] == self.in_dim, 'Incorrect last dimension for x.')
        
        outputs = []
        for block in self.blocks:
            outputs.append(torch.flatten(block(x), 1))
        return torch.cat(outputs, -1)

            
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} ({self.in_l_max} -> {self.out_l_max} | {self.in_paths[:self.out_l_max+1]} -> {[1 for _ in range(self.out_l_max+1)]} paths | {sum([w.numel() for w in self.weight])} weights)")
