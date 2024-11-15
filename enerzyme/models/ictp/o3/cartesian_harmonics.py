"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     cartesian_harmonics.py
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
from typing import Optional, Tuple

import torch
import torch.nn as nn

import torch.fx
import opt_einsum_fx


L_MAX = 3
BATCH_SIZE = 10


class RankTwoCartesianHarmonics(nn.Module):
    """A basic class for calculating rank-two Cartesian harmonics. We use this class as a workaround 
    for torch.jit.script, though."""
    def __init__(self):
        super(RankTwoCartesianHarmonics, self).__init__()
        # trace and optimize contraction
        contraction_eq = 'ai, aj -> aij'
        contraction_tr = torch.fx.symbolic_trace(lambda x, y: torch.einsum(contraction_eq, x, y))
        self.contraction = opt_einsum_fx.optimize_einsums_full(model=contraction_tr,
                                                               example_inputs=(torch.randn(BATCH_SIZE, 3),
                                                                               torch.randn(BATCH_SIZE, 3)))
    
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                eye: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Args:
            x (torch.Tensor): Input unit vector.
            y (torch.Tensor): Input unit vector. Note that x == y in this case.
            eye (torch.Tensor): Identity matrix.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rank-two Cartesian harmonics. The first tensor represents the component from 
                                               which the trace has not been removed. It is used in subsequent calculations.
        """
        torch._assert(torch.equal(x, y), 'The rank-two Cartesian harmonics is defined for x == y!')
        # l = 2, shape: n_neighbors x 3 x 3
        xy = self.contraction(x, y)
        return xy, 3. / 2. * (xy - 1. / 3. * eye.unsqueeze(0))
    

class RankThreeCartesianHarmonics(nn.Module):
    """A basic class for calculating rank-three Cartesian harmonics. We use this class as a workaround 
    for torch.jit.script, though."""
    def __init__(self):
        super(RankThreeCartesianHarmonics, self).__init__()
        # trace and optimize contractions
        contraction_eq = 'ai, jk -> aijk'
        contraction_tr = torch.fx.symbolic_trace(lambda x, y: torch.einsum(contraction_eq, x, y))
        self.contraction1 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr,
                                                                example_inputs=(torch.randn(BATCH_SIZE, 3),
                                                                                torch.eye(3)))
        
        contraction_eq = 'ai, ajk -> aijk'
        contraction_tr = torch.fx.symbolic_trace(lambda x, y: torch.einsum(contraction_eq, x, y))
        self.contraction2 = opt_einsum_fx.optimize_einsums_full(model=contraction_tr,
                                                                example_inputs=(torch.randn(BATCH_SIZE, 3),
                                                                                torch.randn(BATCH_SIZE, 3, 3)))
    
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                eye: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        
        Args:
            x (torch.Tensor): Input unit vector.
            y (torch.Tensor): Input tensor. Note that y = x \otimes x in this case.
            eye (torch.Tensor): Identity matrix.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rank-two Cartesian harmonics. We set the first tensor to None different 
                                               from `RankTwoCartesianHarmonics` because we don't need it for aubsequent 
                                               calculation in this version of the code.
        """
        # l = 3, shape: n_neighbors x 3 x 3 x 3
        xe = self.contraction1(x, eye)
        xe = xe + xe.permute(0, 2, 3, 1) + xe.permute(0, 3, 1, 2)
        xy = self.contraction2(x, y)
        # we don't need xy = x \otimes (x \otimes x) in this version of the code
        return None, 5. / 2. * (xy - 1. / 5. * xe)
       

class CartesianHarmonics(nn.Module):
    """Computes irreducible Cartesian tensors, also referred to as Cartesian harmonics.

    Args:
        l_max (int): Maximal rank of Cartesian harmonics.

    Note:
        1. All Cartesian harmonics are built using unit vectors in the current implementation.
        2. We normalize Cartesian harmonics of rank l such that their products with the respective 
           unit vector reduce the rank of these tensors by one, i.e., to l-1. Applying l unit 
           vectors to an irreducible Cartesian tensor of rank l yields unity.
    """
    def __init__(self, l_max: int):
        super(CartesianHarmonics, self).__init__()
        self.l_max = l_max
        if self.l_max > L_MAX:
            raise RuntimeError(f'Cartesian harmonics are implemented for l <= {L_MAX=}.')
        
        # define identity matrix
        self.register_buffer('eye', torch.eye(3))
        
        # define blocks for tensors of rank two and three
        self.blocks = nn.ModuleList()
        if self.l_max > 1: self.blocks.append(RankTwoCartesianHarmonics())    
        if self.l_max > 2: self.blocks.append(RankThreeCartesianHarmonics())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes Cartesian harmonics from the input unit vector.

        Args:
            x (torch.Tensor): Input unit vector.

        Returns:
            torch.Tensor: Cartesian harmonics.
        """
        x = nn.functional.normalize(x, dim=-1)
        
        # l = 0, shape: n_neighbors x 1
        ch_list = [torch.ones(x.size(0), 1, device=x.device)]
        
        # l = 1, shape: n_neighbors x 3
        if self.l_max > 0:
            ch_list.append(x)
        
        # l >= 2, shape n_neighbors x 3 x l-times x 3
        xy = x
        for block in self.blocks:
            xy, ch = block(x, xy, self.eye)
            ch_list.append(torch.flatten(ch, 1))
        
        return torch.cat(ch_list, -1)
            
    def __repr__(self):
        return f'{self.__class__.__name__}(l_max={self.l_max})'
