"""GemNet-OC Core — interaction hierarchy on the GemNet message stack.

Gasteiger, Shuaibi, Sriram, Günnemann, Ulissi, Zitnick, and Das, TMLR 2022.
The Core owns the capped graphs and the interaction blocks. The species
embedding and the Gaussian radial basis stay outside
(``GaussianRBF`` with ``flavor: gemnet-oc``). Each block emits an atom
latent for :class:`~enerzyme.models.layers.readout.HierachicalReadout`.
Forces of this stack are ``Fa = -∇E``.

The published OC20 model predicts forces directly from edge states and
restricts quadruplets with atom tags. Neither choice is part of this Core.
fairchem checkpoints do not load.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers import BaseAtomEmbedding, BaseFFCore
from ..layers.geometry import RangeSeparationLayer
from ..layers.rbf import GaussianRBFLayer
from .basis import legendre_outer, y_lm0
from .geometry import (
    cap_incoming_mask,
    mixed_triplets,
    quadruplet_graph,
    symmetrize_edges,
)
from .layers import BasisDownProjection, EdgeEmbedding, GemNetDense
from .oc import GemNetOCBlock, GemNetOCOutput


def _positive(name: str, value: int) -> int:
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _nonnegative(name: str, value: int) -> int:
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _cutoff(name: str, value: float) -> float:
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


class GemNetOCCore(BaseFFCore):
    """Directional GemNet-OC interaction hierarchy.

    ``cutoff`` is the embedding graph: those edges carry the edge state.
    ``cutoff_qint`` limits the central bond of a dihedral.
    ``cutoff_aeaint`` limits atom–edge and edge–atom triplets.
    ``cutoff_aint`` limits atom–atom pairs and is the longest cutoff. It
    must be at most ``cutoff_sr``, which is the neighbor list built by
    :class:`~enerzyme.models.layers.geometry.RangeSeparationLayer`. Each
    graph also keeps at most its ``max_neighbors`` closest incoming edges.

    ``atom_feature`` is ``(N, emb_atom, num_blocks + 1)`` in
    ``output_mode="feature"``. An isolated atom is not a zero latent: the
    output block residual-adds the atom embedding.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        cutoff_sr: float,
        num_blocks: int = 4,
        num_spherical: int = 7,
        emb_size_edge: int = 512,
        emb_size_trip_in: int = 64,
        emb_size_trip_out: int = 64,
        emb_size_quad_in: int = 32,
        emb_size_quad_out: int = 32,
        emb_size_aint_in: int = 64,
        emb_size_aint_out: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_sbf: int = 32,
        num_before_skip: int = 2,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        num_atom_emb_layers: int = 2,
        num_output_afteratom: int = 3,
        cutoff: Optional[float] = None,
        cutoff_qint: Optional[float] = None,
        cutoff_aeaint: Optional[float] = None,
        cutoff_aint: Optional[float] = None,
        max_neighbors: int = 30,
        max_neighbors_qint: int = 8,
        max_neighbors_aeaint: int = 20,
        max_neighbors_aint: int = 1000,
        quad_interaction: bool = True,
        atom_edge_interaction: bool = True,
        edge_atom_interaction: bool = True,
        atom_interaction: bool = True,
        envelope_exponent: int = 5,
        output_mode: Literal["feature", "last"] = "feature",
    ) -> None:
        dim_embedding = _positive("dim_embedding", dim_embedding)
        num_rbf = _positive("num_rbf", num_rbf)
        num_blocks = _positive("num_blocks", num_blocks)
        num_spherical = _positive("num_spherical", num_spherical)
        cutoff_sr_v = _cutoff("cutoff_sr", cutoff_sr)
        cutoff_aint_v = cutoff_sr_v if cutoff_aint is None else _cutoff("cutoff_aint", cutoff_aint)
        if cutoff_aint_v > cutoff_sr_v + 1e-6:
            raise ValueError(
                f"cutoff_aint ({cutoff_aint_v}) cannot exceed cutoff_sr ({cutoff_sr_v})"
            )
        cutoff_v = cutoff_aint_v if cutoff is None else _cutoff("cutoff", cutoff)
        cutoff_q_v = cutoff_v if cutoff_qint is None else _cutoff("cutoff_qint", cutoff_qint)
        cutoff_ae_v = (
            cutoff_v if cutoff_aeaint is None else _cutoff("cutoff_aeaint", cutoff_aeaint)
        )
        if cutoff_v > cutoff_aint_v + 1e-6:
            raise ValueError(
                f"cutoff ({cutoff_v}) cannot exceed cutoff_aint ({cutoff_aint_v})"
            )
        if cutoff_q_v > cutoff_aint_v + 1e-6:
            raise ValueError(
                f"cutoff_qint ({cutoff_q_v}) cannot exceed cutoff_aint ({cutoff_aint_v})"
            )
        if cutoff_ae_v > cutoff_aint_v + 1e-6:
            raise ValueError(
                f"cutoff_aeaint ({cutoff_ae_v}) cannot exceed cutoff_aint ({cutoff_aint_v})"
            )
        if output_mode not in ("feature", "last"):
            raise ValueError(f"output_mode must be 'feature' or 'last', got {output_mode!r}")
        if int(envelope_exponent) < 1:
            raise ValueError(f"envelope_exponent must be >= 1, got {envelope_exponent}")

        self.output_mode = output_mode
        self.quad_interaction = bool(quad_interaction)
        self.atom_edge_interaction = bool(atom_edge_interaction)
        self.edge_atom_interaction = bool(edge_atom_interaction)
        self.atom_interaction = bool(atom_interaction)
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "Dij_sr",
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
            },
            output_fields={"atom_feature"},
        )
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.cutoff_sr = cutoff_sr_v
        self.cutoff = cutoff_v
        self.cutoff_qint = cutoff_q_v
        self.cutoff_aeaint = cutoff_ae_v
        self.cutoff_aint = cutoff_aint_v
        self.max_neighbors = _positive("max_neighbors", max_neighbors)
        self.max_neighbors_qint = _positive("max_neighbors_qint", max_neighbors_qint)
        self.max_neighbors_aeaint = _positive("max_neighbors_aeaint", max_neighbors_aeaint)
        self.max_neighbors_aint = _positive("max_neighbors_aint", max_neighbors_aint)
        self.num_blocks = num_blocks
        self.num_spherical = num_spherical
        self.emb_size_edge = _positive("emb_size_edge", emb_size_edge)
        self.emb_size_rbf = _positive("emb_size_rbf", emb_size_rbf)
        self.emb_size_cbf = _positive("emb_size_cbf", emb_size_cbf)
        self.emb_size_sbf = _positive("emb_size_sbf", emb_size_sbf)
        self.envelope_exponent = int(envelope_exponent)
        self.num_output_blocks = num_blocks + 1
        self.dim_feature_out = dim_embedding
        self.feature_irreps = f"{dim_embedding}x0e"
        self.radial_basis: Optional[GaussianRBFLayer] = None

        self.edge_emb = EdgeEmbedding(dim_embedding, num_rbf, self.emb_size_edge)
        self.mlp_rbf_tint = GemNetDense(num_rbf, self.emb_size_rbf, activation=False)
        self.mlp_rbf_h = GemNetDense(num_rbf, self.emb_size_rbf, activation=False)
        self.mlp_rbf_out = GemNetDense(num_rbf, self.emb_size_rbf, activation=False)
        self.mlp_cbf_tint = BasisDownProjection(
            num_spherical, num_rbf, self.emb_size_cbf
        )
        if self.quad_interaction:
            self.mlp_rbf_qint = GemNetDense(num_rbf, self.emb_size_rbf, activation=False)
            self.mlp_cbf_qint = BasisDownProjection(
                num_spherical, num_rbf, self.emb_size_cbf
            )
            self.mlp_sbf_qint = BasisDownProjection(
                num_spherical * num_spherical, num_rbf, self.emb_size_sbf
            )
        else:
            self.mlp_rbf_qint = None
            self.mlp_cbf_qint = None
            self.mlp_sbf_qint = None
        if self.atom_edge_interaction:
            self.mlp_rbf_aeint = GemNetDense(num_rbf, self.emb_size_rbf, activation=False)
            self.mlp_cbf_aeint = BasisDownProjection(
                num_spherical, num_rbf, self.emb_size_cbf
            )
        else:
            self.mlp_rbf_aeint = None
            self.mlp_cbf_aeint = None
        if self.edge_atom_interaction:
            self.mlp_rbf_eaint = GemNetDense(num_rbf, self.emb_size_rbf, activation=False)
            self.mlp_cbf_eaint = BasisDownProjection(
                num_spherical, num_rbf, self.emb_size_cbf
            )
        else:
            self.mlp_rbf_eaint = None
            self.mlp_cbf_eaint = None
        if self.atom_interaction:
            self.mlp_rbf_aint = GemNetDense(num_rbf, self.emb_size_rbf, activation=False)
        else:
            self.mlp_rbf_aint = None

        block_kwargs = dict(
            emb_atom=dim_embedding,
            emb_edge=self.emb_size_edge,
            emb_trip_in=_positive("emb_size_trip_in", emb_size_trip_in),
            emb_trip_out=_positive("emb_size_trip_out", emb_size_trip_out),
            emb_quad_in=_positive("emb_size_quad_in", emb_size_quad_in),
            emb_quad_out=_positive("emb_size_quad_out", emb_size_quad_out),
            emb_aint_in=_positive("emb_size_aint_in", emb_size_aint_in),
            emb_aint_out=_positive("emb_size_aint_out", emb_size_aint_out),
            emb_rbf=self.emb_size_rbf,
            emb_cbf=self.emb_size_cbf,
            emb_sbf=self.emb_size_sbf,
            num_before_skip=_nonnegative("num_before_skip", num_before_skip),
            num_after_skip=_nonnegative("num_after_skip", num_after_skip),
            num_concat=_nonnegative("num_concat", num_concat),
            num_atom=_nonnegative("num_atom", num_atom),
            num_atom_emb_layers=_nonnegative("num_atom_emb_layers", num_atom_emb_layers),
            quad_interaction=self.quad_interaction,
            atom_edge_interaction=self.atom_edge_interaction,
            edge_atom_interaction=self.edge_atom_interaction,
            atom_interaction=self.atom_interaction,
        )
        self.interactions = ModuleList(
            [GemNetOCBlock(**block_kwargs) for _ in range(num_blocks)]
        )
        n_after = _nonnegative("num_output_afteratom", num_output_afteratom)
        self.readouts = ModuleList(
            [
                GemNetOCOutput(
                    dim_embedding,
                    self.emb_size_edge,
                    self.emb_size_rbf,
                    block_kwargs["num_atom"],
                    n_after,
                )
                for _ in range(self.num_output_blocks)
            ]
        )

    def __str__(self) -> str:
        return """
#########################################################################
# GemNet-OC (TMLR 2022) — interaction hierarchy                        #
# Gaussian basis and atom embedding stay outside this Core.            #
# Energy and conservative forces stay outside this Core.               #
#########################################################################
"""

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)
        if self.output_mode == "last":
            from ..layers.readout import HierachicalNSEReadout, HierachicalReadout

            hierarchical = any(
                isinstance(layer, (HierachicalReadout, HierachicalNSEReadout))
                for layer in built_layers
            )
            if hierarchical:
                raise ValueError(
                    "GemNet-OC output_mode='last' returns atom_feature of shape "
                    "(N, F) for SimpleReadout. HierachicalReadout reads "
                    "(N, F, num_blocks+1). Use output_mode='feature'."
                )
        found_atom = False
        gaussians: List[GaussianRBFLayer] = []
        for layer in built_layers:
            if isinstance(layer, BaseAtomEmbedding):
                if int(layer.dim_embedding) != self.dim_embedding:
                    raise ValueError(
                        "Atom embedding width "
                        f"({layer.dim_embedding}) must equal dim_embedding "
                        f"({self.dim_embedding})"
                    )
                found_atom = True
            elif isinstance(layer, RangeSeparationLayer):
                if abs(float(layer.cutoff_sr) - self.cutoff_sr) > 1e-4:
                    raise ValueError(
                        "RangeSeparation cutoff_sr "
                        f"({layer.cutoff_sr}) must equal GemNet-OC cutoff_sr "
                        f"({self.cutoff_sr})"
                    )
            elif (
                isinstance(layer, GaussianRBFLayer)
                and getattr(layer, "flavor", None) == "gemnet-oc"
            ):
                gaussians.append(layer)
        if not found_atom:
            raise ValueError(
                "GemNetOCCore requires an atom embedding before the Core"
            )
        if len(gaussians) != 1:
            raise ValueError(
                "GemNetOCCore requires exactly one GaussianRBF with flavor "
                f"'gemnet-oc', found {len(gaussians)}"
            )
        radial = gaussians[0]
        if int(radial.num_rbf) != self.num_rbf:
            raise ValueError(
                f"GaussianRBF num_rbf ({radial.num_rbf}) must equal num_rbf "
                f"({self.num_rbf})"
            )
        if abs(float(radial.cutoff_sr) - self.cutoff_sr) > 1e-4:
            raise ValueError(
                "GaussianRBF cutoff_sr "
                f"({radial.cutoff_sr}) must equal GemNet-OC cutoff_sr "
                f"({self.cutoff_sr}). Shorter interaction cutoffs reuse this "
                "expansion."
            )
        if int(getattr(radial, "envelope_exponent", -1)) != self.envelope_exponent:
            raise ValueError(
                "GaussianRBF envelope_exponent "
                f"({getattr(radial, 'envelope_exponent', None)}) must equal "
                f"the Core envelope_exponent ({self.envelope_exponent})"
            )
        if radial.apply_cutoff_fn:
            raise ValueError(
                "GaussianRBF flavor 'gemnet-oc' must set apply_cutoff_fn: false"
            )
        # The layer already lives in pre_sequence. A second registration would
        # parent it twice.
        object.__setattr__(self, "radial_basis", radial)

    def _basis(self, distance: Tensor, cutoff: float) -> Tensor:
        radial = self.radial_basis
        if radial is None:
            raise RuntimeError(
                "GemNetOCCore.build has not adopted a gemnet-oc GaussianRBF"
            )
        return radial.expansion(distance, cutoff)

    def _down(self, layer: BasisDownProjection, raw: Tensor) -> Tensor:
        n_angular = int(layer.weight.shape[0])
        return layer(raw.unsqueeze(1).expand(-1, n_angular, -1).contiguous())

    def _circular(
        self,
        layer: BasisDownProjection,
        distance: Tensor,
        angle: Tensor,
        cutoff: float,
    ) -> Tensor:
        if angle.numel() == 0:
            return distance.new_zeros(0, int(layer.weight.shape[-1]))
        projected = self._down(layer, self._basis(distance, cutoff))
        return (projected * y_lm0(angle, self.num_spherical).unsqueeze(1)).sum(dim=-1)

    def _select(
        self,
        idx_i: Tensor,
        idx_j: Tensor,
        dist: Tensor,
        vij: Tensor,
        num_nodes: int,
        cutoff: float,
        max_neighbors: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mask = cap_incoming_mask(idx_i, dist, num_nodes, max_neighbors, cutoff)
        return idx_i[mask], idx_j[mask], dist[mask], vij[mask]

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> dict:
        self._check_edge_tensors(rbf, Dij_sr, vij_sr, idx_i_sr, idx_j_sr)
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                "atom_embedding must be "
                f"(N, {self.dim_embedding}), got {tuple(atom_embedding.shape)}"
            )
        num_nodes = int(atom_embedding.shape[0])
        if idx_i_sr.numel() and (
            int(idx_i_sr.min()) < 0
            or int(idx_j_sr.min()) < 0
            or int(idx_i_sr.max()) >= num_nodes
            or int(idx_j_sr.max()) >= num_nodes
        ):
            raise ValueError("edge index is outside atom_embedding")

        main_i, main_j, main_d, main_v = self._select(
            idx_i_sr, idx_j_sr, Dij_sr, vij_sr,
            num_nodes, self.cutoff, self.max_neighbors,
        )
        main_i, main_j, main_d, main_v, swap = symmetrize_edges(
            main_i, main_j, main_d, main_v
        )
        use_ae = self.atom_edge_interaction or self.edge_atom_interaction
        if use_ae:
            ae_i, ae_j, ae_d, ae_v = self._select(
                idx_i_sr, idx_j_sr, Dij_sr, vij_sr,
                num_nodes, self.cutoff_aeaint, self.max_neighbors_aeaint,
            )
        else:
            ae_i = ae_j = idx_i_sr.new_empty(0)
            ae_d = Dij_sr.new_empty(0)
            ae_v = vij_sr.new_empty(0, 3)
        if self.atom_interaction:
            a2a_i, a2a_j, a2a_d, _a2a_v = self._select(
                idx_i_sr, idx_j_sr, Dij_sr, vij_sr,
                num_nodes, self.cutoff_aint, self.max_neighbors_aint,
            )
        else:
            a2a_i = a2a_j = idx_i_sr.new_empty(0)
            a2a_d = Dij_sr.new_empty(0)

        rbf_main = self._basis(main_d, self.cutoff)
        rbf_e2e = self.mlp_rbf_tint(rbf_main)
        rbf_h = self.mlp_rbf_h(rbf_main)
        rbf_out = self.mlp_rbf_out(rbf_main)
        radial_w3 = self._down(self.mlp_cbf_tint, rbf_main)
        trips = mixed_triplets(main_i, main_j, main_v, main_i, main_j, main_v, num_nodes)
        angular3 = y_lm0(trips.angle, self.num_spherical)

        quad_kwargs: dict = {}
        if self.quad_interaction:
            assert self.mlp_rbf_qint is not None
            assert self.mlp_cbf_qint is not None
            assert self.mlp_sbf_qint is not None
            q_i, q_j, q_d, q_v = self._select(
                idx_i_sr, idx_j_sr, Dij_sr, vij_sr,
                num_nodes, self.cutoff_qint, self.max_neighbors_qint,
            )
            quad_kwargs = self._quadruplet_basis(
                main_i, main_j, main_d, main_v, rbf_main,
                q_i, q_j, q_d, q_v, num_nodes,
            )
        ae_kwargs: dict = {}
        if self.atom_edge_interaction:
            assert self.mlp_rbf_aeint is not None and self.mlp_cbf_aeint is not None
            a2e = mixed_triplets(main_i, main_j, main_v, ae_i, ae_j, ae_v, num_nodes)
            ae_kwargs.update(
                rbf_a2e=self.mlp_rbf_aeint(self._basis(ae_d, self.cutoff_aeaint)),
                radial_a2e=self._down(self.mlp_cbf_aeint, rbf_main),
                angular_a2e=y_lm0(a2e.angle, self.num_spherical),
                a2e_sender=ae_j,
                a2e_other=a2e.idx_other,
                a2e_edge=a2e.idx_edge,
            )
        if self.edge_atom_interaction:
            assert self.mlp_rbf_eaint is not None and self.mlp_cbf_eaint is not None
            e2a = mixed_triplets(ae_i, ae_j, ae_v, main_i, main_j, main_v, num_nodes)
            raw_ae = self._basis(ae_d, self.cutoff_aeaint)
            ae_kwargs.update(
                rbf_e2a=self.mlp_rbf_eaint(rbf_main),
                radial_e2a=self._down(self.mlp_cbf_eaint, raw_ae),
                angular_e2a=y_lm0(e2a.angle, self.num_spherical),
                e2a_other=e2a.idx_other,
                e2a_edge=e2a.idx_edge,
                e2a_receiver=ae_i,
            )
        if self.atom_interaction:
            assert self.mlp_rbf_aint is not None
            ae_kwargs.update(
                rbf_a2a=self.mlp_rbf_aint(self._basis(a2a_d, self.cutoff_aint)),
                a2a_i=a2a_i,
                a2a_j=a2a_j,
            )

        h = atom_embedding
        m = self.edge_emb(h, rbf_main, main_i, main_j)
        features = [self.readouts[0](h, m, rbf_out, main_i)]
        for block, readout in zip(self.interactions, self.readouts[1:]):
            h, m = block(
                h, m, rbf_e2e, rbf_h, radial_w3, angular3,
                trips.idx_edge, trips.idx_other, swap, main_i, main_j,
                **quad_kwargs,
                **ae_kwargs,
            )
            features.append(readout(h, m, rbf_out, main_i))
        atom_feature = torch.stack(features, dim=-1)
        if self.output_mode == "last":
            return {"atom_feature": atom_feature[:, :, -1]}
        return {"atom_feature": atom_feature}

    def _quadruplet_basis(
        self,
        main_i: Tensor,
        main_j: Tensor,
        main_d: Tensor,
        main_v: Tensor,
        rbf_main: Tensor,
        q_i: Tensor,
        q_j: Tensor,
        q_d: Tensor,
        q_v: Tensor,
        num_nodes: int,
    ) -> dict:
        assert self.mlp_rbf_qint is not None
        assert self.mlp_cbf_qint is not None
        assert self.mlp_sbf_qint is not None
        n_main = int(main_i.shape[0])
        n_sph = self.num_spherical * self.num_spherical
        radial_w4 = self._down(self.mlp_sbf_qint, rbf_main)
        rbf_q = self.mlp_rbf_qint(rbf_main)
        empty_e = main_i.new_empty(0)
        if n_main == 0 or q_i.numel() == 0:
            return {
                "rbf_q": rbf_q,
                "cbf_q": rbf_main.new_zeros(0, self.emb_size_cbf),
                "radial_w4": radial_w4,
                "angular4": rbf_main.new_zeros(0, n_sph),
                "edge_db": empty_e,
                "edge_ca": empty_e,
            }
        offset = n_main
        quads = quadruplet_graph(
            torch.cat([main_i, q_i]),
            torch.cat([main_j, q_j]),
            torch.cat([main_d, q_d]),
            torch.cat([main_v, q_v]),
            torch.arange(n_main, device=main_i.device),
            torch.arange(offset, offset + int(q_i.shape[0]), device=main_i.device),
            num_nodes,
            images=True,
        )
        if quads.edge_ca.numel() == 0:
            cbf_q = rbf_main.new_zeros(0, self.emb_size_cbf)
            angular4 = rbf_main.new_zeros(0, n_sph)
        else:
            cbf_q = self._circular(
                self.mlp_cbf_qint,
                q_d[quads.int_local],
                quads.angle_abd,
                self.cutoff_qint,
            )
            angular4 = legendre_outer(
                quads.angle_cab, quads.dihedral, self.num_spherical
            )
        return {
            "rbf_q": rbf_q,
            "cbf_q": cbf_q,
            "radial_w4": radial_w4,
            "angular4": angular4,
            "edge_db": quads.edge_db,
            "edge_ca": quads.edge_ca,
        }

    def _check_edge_tensors(
        self,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> None:
        shapes = {
            "rbf": tuple(rbf.shape),
            "Dij_sr": tuple(Dij_sr.shape),
            "vij_sr": tuple(vij_sr.shape),
            "idx_i_sr": tuple(idx_i_sr.shape),
            "idx_j_sr": tuple(idx_j_sr.shape),
        }
        lengths = {name: shape[0] if shape else -1 for name, shape in shapes.items()}
        aligned = (
            rbf.ndim == 2
            and rbf.shape[-1] == self.num_rbf
            and Dij_sr.ndim == 1
            and vij_sr.ndim == 2
            and vij_sr.shape[-1] == 3
            and idx_i_sr.ndim == 1
            and idx_j_sr.ndim == 1
            and len(set(lengths.values())) == 1
        )
        if aligned:
            return
        detail = ", ".join(f"{name} {shape}" for name, shape in shapes.items())
        raise ValueError(
            "rbf, Dij_sr, vij_sr, idx_i_sr, and idx_j_sr must describe the same "
            f"edges (rbf width {self.num_rbf}, Dij_sr rank 1, vij_sr shape (E, 3)); "
            f"got {detail}"
        )
