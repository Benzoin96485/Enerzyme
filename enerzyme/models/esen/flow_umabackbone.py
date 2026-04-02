"""
UMA eSCNMDBackbone subclass for flow matching: add pre-core ``atom_embedding`` (sum of
``*_embedding`` fields from GatherAtomEmbedding) into the l=0 node channels before
edge_degree_embedding and interaction blocks.

The parent forward is reproduced from fairchem (eSCNMDBackbone) so we can inject
at the early invariant fusion point; keep in sync when upgrading fairchem.
"""

from __future__ import annotations

import torch
from fairchem.core.common.utils import conditional_grad
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from torch.profiler import record_function


class ESCNMDBackboneFlow(eSCNMDBackbone):
    """eSCNMDBackbone with optional pre-gathered ``atom_embedding`` fused into scalar (l=0) channels early."""

    @conditional_grad(torch.enable_grad())
    def forward(
        self,
        data_dict: AtomicData,
        atom_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]

        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
            dataset=data_dict.get("dataset", None),
        )

        self.set_MOLE_coefficients(
            atomic_numbers_full=data_dict["atomic_numbers_full"],
            batch_full=data_dict["batch_full"],
            csd_mixed_emb=csd_mixed_emb,
        )

        if not self.regress_config.direct_forces:
            if self.regress_config.forces or self.regress_config.stress:
                data_dict["pos"].requires_grad_(True)
            if self.regress_config.stress:
                data_dict["cell"].requires_grad_(True)

        with record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)

        with record_function("obtain wigner"):
            wigner, wigner_inv = self._get_rotmat_and_wigner(
                graph_dict["edge_distance_vec"],
            )
            coefficient_index = (
                self.coefficient_index if self.mmax != self.lmax else None
            )
            wigner, wigner_inv = self.backend.prepare_wigner(
                wigner,
                wigner_inv,
                self.mappingReduced,
                coefficient_index,
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        with record_function("atom embedding"):
            x_message = torch.zeros(
                data_dict["atomic_numbers"].shape[0],
                self.sph_feature_size,
                self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            x_message[:, 0, :] = self.sphere_embedding(data_dict["atomic_numbers"])

        sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        if atom_embedding is not None:
            x_message[:, 0, :] = x_message[:, 0, :] + atom_embedding.to(
                device=x_message.device,
                dtype=x_message.dtype,
            )

        self.set_MOLE_sizes(
            nsystems=csd_mixed_emb.shape[0],
            batch_full=data_dict["batch_full"],
            edge_index=graph_dict["edge_index"],
        )
        self.log_MOLE_stats()

        with record_function("edge embedding"):
            dist_scaled = graph_dict["edge_distance"] / self.cutoff
            edge_envelope = self.envelope(dist_scaled).reshape(-1, 1, 1)
            edge_distance_embedding = self.distance_expansion(
                graph_dict["edge_distance"]
            )
            source_embedding = self.source_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][0]]
            )
            target_embedding = self.target_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][1]]
            )
            x_edge = torch.cat(
                (edge_distance_embedding, source_embedding, target_embedding),
                dim=1,
            )

            wigner_inv_envelope = wigner_inv * edge_envelope

            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_index"],
                wigner_inv_envelope,
                data_dict["gp_node_offset"],
            )

        with record_function("layer_radial_emb"):
            x_edge_per_layer = self.backend.get_layer_radial_emb(x_edge, self)

        for i in range(self.num_layers):
            with record_function(f"message passing {i}"):
                x_message = self.blocks[i](
                    x_message,
                    x_edge_per_layer[i],
                    graph_dict["edge_index"],
                    wigner,
                    wigner_inv_envelope,
                    total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[
                        0
                    ],
                    sys_node_embedding=sys_node_embedding,
                    node_offset=data_dict["gp_node_offset"],
                )
                x_message = self.balance_channels(
                    x_message,
                    charge=data_dict["charge"],
                    spin=data_dict["spin"],
                    natoms=data_dict["natoms"],
                    batch=data_dict["batch"],
                )

        x_message = self.norm(x_message)
        out = {
            "node_embedding": x_message,
            "batch": data_dict["batch"],
        }
        return out
