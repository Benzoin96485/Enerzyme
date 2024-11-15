"""
       ICTP: Irreducible Cartesian Tensor Potentials
	  
  File:     representations.py
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
from typing import List, Any, Dict

import torch
import torch.nn as nn

from ..o3.linear_transform import LinearTransform
from ..o3.cartesian_harmonics import CartesianHarmonics
from .layers import RadialEmbeddingLayer, RealAgnosticResidualInteractionLayer, ProductBasisLayer


class CartesianMACE(nn.Module):
    """MACE-like (semi-)local atomic representation using irreducible Cartesian tensors.

    Args:
        r_cutoff (float): Cutoff radius.
        n_basis (int): Number of radial basis functions.
        n_polynomial_cutoff (int): Polynomial order for the cutoff function. 
        n_species (int): Number of atomic species.
        n_hidden_feats (int): Number of hidden features.
        n_product_feats (int): Number of product basis features.
        coupled_product_feats (bool): If True, use coupled feature channels when computing the product basis.
        symmetric_product (bool): If True, exploit symmetry of the tensor product to reduce 
                                  the number of possible tensor contractions.
        l_max_hidden_feats (int): Maximal rotational order/rank of the Cartesian tensor for the hidden features.
        l_max_edge_attrs (int): Maximal rotational order/rank of the Cartesian tensor for the Cartesian harmonics.
        avg_n_neighbors (float): Avergae number of neighbors. It is used to normalize messages.
        correlation (int): Correlation order, i.e., number of contracted Cartesian tensors.
        n_interactions (int): Number of interaction layers.
        radial_MLP (List[int]): List of hidden features for the radial embedding network.
    """
    def __init__(self,
                 r_cutoff: float,
                 n_basis: int,
                 n_polynomial_cutoff: int,
                 n_species: int,
                 n_hidden_feats: int,
                 n_product_feats: int,
                 coupled_product_feats: bool,
                 symmetric_product: bool,
                 l_max_hidden_feats: int,
                 l_max_edge_attrs: int,
                 avg_n_neighbors: float,
                 correlation: int,
                 n_interactions: int,
                 radial_MLP: List[int],
                 **config: Any):
        super(CartesianMACE, self).__init__()
        self.n_hidden_feats = n_hidden_feats
        
        self.node_embedding = LinearTransform(in_l_max=0, out_l_max=0, in_features=n_species, out_features=n_hidden_feats)
        
        self.radial_embedding = RadialEmbeddingLayer(r_cutoff=r_cutoff, n_basis=n_basis, n_polynomial_cutoff=n_polynomial_cutoff)
        
        self.cartesian_harmonics = CartesianHarmonics(l_max_edge_attrs)
        
        if n_interactions == 1:
            l_max_out_feats = 0
        else:
            l_max_out_feats = l_max_hidden_feats
        
        inter = RealAgnosticResidualInteractionLayer(l_max_node_feats=0, l_max_edge_attrs=l_max_edge_attrs, l_max_target_feats=l_max_edge_attrs, 
                                                     l_max_hidden_feats=l_max_out_feats, n_basis=n_basis, n_species=n_species, in_features=n_hidden_feats, 
                                                     out_features=n_product_feats, avg_n_neighbors=avg_n_neighbors, radial_MLP=radial_MLP)
        
        self.interactions = torch.nn.ModuleList([inter])
        
        prod = ProductBasisLayer(l_max_node_feats=l_max_edge_attrs, l_max_target_feats=l_max_out_feats, in_features=n_product_feats, 
                                 out_features=n_hidden_feats, n_species=n_species, correlation=correlation, coupled_feats=coupled_product_feats, 
                                 symmetric_product=symmetric_product, use_sc=True)
        
        self.products = torch.nn.ModuleList([prod])
        
        for i in range(1, n_interactions):
            if i == n_interactions - 1:
                l_max_out_feats = 0
            else:
                l_max_out_feats = l_max_hidden_feats
            inter = RealAgnosticResidualInteractionLayer(l_max_node_feats=l_max_hidden_feats, l_max_edge_attrs=l_max_edge_attrs, l_max_target_feats=l_max_edge_attrs, 
                                                         l_max_hidden_feats=l_max_out_feats, n_basis=n_basis, n_species=n_species, in_features=n_hidden_feats, 
                                                         out_features=n_product_feats, avg_n_neighbors=avg_n_neighbors, radial_MLP=radial_MLP)
            self.interactions.append(inter)
            
            prod = ProductBasisLayer(l_max_node_feats=l_max_edge_attrs, l_max_target_feats=l_max_out_feats, in_features=n_product_feats, 
                                     out_features=n_hidden_feats, n_species=n_species, correlation=correlation, coupled_feats=coupled_product_feats,
                                     symmetric_product=symmetric_product, use_sc=True)
            self.products.append(prod)
                
    def forward(self, graph: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Computes node features.

        Args:
            graph (Dict[str, torch.Tensor]): Atomic graph dictionary.

        Returns:
            torch.Tensor: Node features.
        """
        edge_index, positions, node_attrs, shifts = graph['edge_index'], graph['positions'], graph['node_attrs'], graph['shifts']
        
        idx_i, idx_j = edge_index[0, :], edge_index[1, :]
        vectors = positions.index_select(0, idx_i) - positions.index_select(0, idx_j) - shifts
        lengths = torch.norm(vectors, dim=-1, keepdim=True)
        
        node_feats = self.node_embedding(node_attrs)
        edge_feats = self.radial_embedding(lengths)
        edge_attrs = self.cartesian_harmonics(vectors)  # vectors are normalized when computing Cartesian harmonics

        node_feats_list = []
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(node_attrs=node_attrs, node_feats=node_feats,
                                         edge_attrs=edge_attrs, edge_feats=edge_feats,
                                         idx_i=idx_i, idx_j=idx_j)
            
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            
            node_feats_list.append(node_feats[:, :self.n_hidden_feats])

        return node_feats_list
