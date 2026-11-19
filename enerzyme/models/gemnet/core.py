"""GemNet Core — directional message passing with triplets and dihedrals.

Gasteiger, Becker, and Günnemann, NeurIPS 2021. ``triplets_only=True`` is
GemNet-T (angles at the receiver). ``triplets_only=False`` is GemNet-Q
(those triplets plus dihedral quadruplets). The reference repository is
Hippocratic-licensed and is not imported; this module is an original
implementation of the published block structure.

The Core owns edge states, Fourier–Bessel angular bases, and the interaction
stack. Species embeddings and the radial Bessel basis are pre-core layers
(``RandomAtomEmbedding``, ``BesselRBF`` with ``flavor: gemnet``). Each block
emits an atom latent for :class:`~enerzyme.models.layers.readout.HierachicalReadout`;
the bias-free linear map from that latent to energy stays outside the Core.
Forces of the conservative model are ``Fa = -∇E``.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers import BaseAtomEmbedding, BaseFFCore, BaseRBF
from .basis import SphericalBesselRadial, joint_zonal, real_spherical_harmonics, y_lm0
from .geometry import quadruplet_graph, reverse_edge_index, split_cutoff_edges, triplet_graph
from .layers import AtomUpdate, BasisDownProjection, EdgeEmbedding, GemNetDense, GemNetInteraction


def _positive(name: str, value: int) -> int:
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _nonnegative(name: str, value: int) -> int:
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


class GemNetCore(BaseFFCore):
    """Directional GemNet interaction stack.

    ``atom_feature`` is ``(N, emb_atom, num_blocks + 1)`` in
    ``output_mode="feature"`` and ``(N, emb_atom)`` in ``output_mode="last"``.
    ``feature`` is the input of :class:`~enerzyme.models.layers.readout.HierachicalReadout`.
    ``last`` is the input of :class:`~enerzyme.models.layers.readout.SimpleReadout`.
    ``emb_atom`` is ``dim_embedding``. GemNet-Q reads edges out to
    ``cutoff_sr`` and treats distances below ``emb_cutoff`` as the embedding
    graph. ``int_cutoff`` is the central-bond cutoff and is ignored for
    GemNet-T.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        cutoff_sr: float,
        num_blocks: int = 4,
        num_spherical: int = 7,
        emb_size_edge: Optional[int] = None,
        emb_size_trip: int = 64,
        emb_size_quad: int = 32,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_sbf: int = 32,
        emb_size_bil_trip: int = 64,
        emb_size_bil_quad: int = 32,
        num_before_skip: int = 1,
        num_after_skip: int = 1,
        num_concat: int = 1,
        num_atom: int = 2,
        triplets_only: bool = True,
        emb_cutoff: Optional[float] = None,
        int_cutoff: Optional[float] = None,
        envelope_exponent: int = 5,
        output_mode: Literal["feature", "last"] = "feature",
    ) -> None:
        dim_embedding = _positive("dim_embedding", dim_embedding)
        num_rbf = _positive("num_rbf", num_rbf)
        num_blocks = _positive("num_blocks", num_blocks)
        num_spherical = _positive("num_spherical", num_spherical)
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if output_mode not in ("feature", "last"):
            raise ValueError(f"output_mode must be 'feature' or 'last', got {output_mode!r}")
        emb_edge = dim_embedding if emb_size_edge is None else _positive("emb_size_edge", emb_size_edge)
        emb_cutoff_v = float(cutoff_sr if emb_cutoff is None else emb_cutoff)
        if triplets_only:
            int_cutoff_v = emb_cutoff_v
        else:
            int_cutoff_v = float(emb_cutoff_v if int_cutoff is None else int_cutoff)
        if emb_cutoff_v <= 0 or int_cutoff_v <= 0:
            raise ValueError("emb_cutoff and int_cutoff must be positive")
        if emb_cutoff_v > float(cutoff_sr) + 1e-6:
            raise ValueError(
                f"emb_cutoff ({emb_cutoff_v}) cannot exceed cutoff_sr ({cutoff_sr})"
            )
        if int_cutoff_v > float(cutoff_sr) + 1e-6:
            raise ValueError(
                f"int_cutoff ({int_cutoff_v}) cannot exceed cutoff_sr ({cutoff_sr})"
            )
        if (not triplets_only) and int_cutoff_v + 1e-6 < emb_cutoff_v:
            raise ValueError(
                f"int_cutoff ({int_cutoff_v}) must be >= emb_cutoff ({emb_cutoff_v})"
            )

        self.output_mode = output_mode
        self.triplets_only = bool(triplets_only)
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
        self.cutoff_sr = float(cutoff_sr)
        self.emb_cutoff = emb_cutoff_v
        self.int_cutoff = int_cutoff_v
        self.num_blocks = num_blocks
        self.num_spherical = num_spherical
        self.emb_size_edge = emb_edge
        self.num_output_blocks = num_blocks + 1
        self.dim_feature_out = dim_embedding
        self.feature_irreps = f"{dim_embedding}x0e"
        self.envelope_exponent = int(envelope_exponent)

        self.edge_emb = EdgeEmbedding(dim_embedding, num_rbf, emb_edge)
        self.triplet_radial = SphericalBesselRadial(
            num_spherical, num_rbf, self.emb_cutoff, self.envelope_exponent, repeat_m=False
        )
        self.mlp_rbf3 = GemNetDense(num_rbf, _positive("emb_size_rbf", emb_size_rbf), activation=False)
        self.mlp_cbf3 = BasisDownProjection(
            num_spherical, num_rbf, _positive("emb_size_cbf", emb_size_cbf)
        )
        self.mlp_rbf_h = GemNetDense(num_rbf, emb_size_rbf, activation=False)
        self.mlp_rbf_out = GemNetDense(num_rbf, emb_size_rbf, activation=False)
        if self.triplets_only:
            self.quad_cbf_radial = None
            self.quad_sbf_radial = None
            self.mlp_rbf4 = None
            self.mlp_cbf4 = None
            self.mlp_sbf4 = None
        else:
            self.quad_cbf_radial = SphericalBesselRadial(
                num_spherical, num_rbf, self.int_cutoff, self.envelope_exponent, repeat_m=False
            )
            self.quad_sbf_radial = SphericalBesselRadial(
                num_spherical, num_rbf, self.emb_cutoff, self.envelope_exponent, repeat_m=True
            )
            self.mlp_rbf4 = GemNetDense(num_rbf, emb_size_rbf, activation=False)
            self.mlp_cbf4 = GemNetDense(
                num_spherical * num_rbf, emb_size_cbf, activation=False
            )
            self.mlp_sbf4 = BasisDownProjection(
                self.quad_sbf_radial.angular_dim,
                num_rbf,
                _positive("emb_size_sbf", emb_size_sbf),
            )

        block_kwargs = dict(
            emb_atom=dim_embedding,
            emb_edge=emb_edge,
            emb_trip=_positive("emb_size_trip", emb_size_trip),
            emb_quad=_positive("emb_size_quad", emb_size_quad) if not self.triplets_only else 1,
            emb_rbf=emb_size_rbf,
            emb_cbf=emb_size_cbf,
            emb_sbf=_positive("emb_size_sbf", emb_size_sbf) if not self.triplets_only else 1,
            emb_bil_trip=_positive("emb_size_bil_trip", emb_size_bil_trip),
            emb_bil_quad=_positive("emb_size_bil_quad", emb_size_bil_quad) if not self.triplets_only else 1,
            num_before_skip=_nonnegative("num_before_skip", num_before_skip),
            num_after_skip=_nonnegative("num_after_skip", num_after_skip),
            num_concat=_nonnegative("num_concat", num_concat),
            num_atom=_nonnegative("num_atom", num_atom),
            triplets_only=self.triplets_only,
        )
        self.interactions = ModuleList(
            [GemNetInteraction(**block_kwargs) for _ in range(num_blocks)]
        )
        self.readouts = ModuleList(
            [
                AtomUpdate(dim_embedding, emb_edge, emb_size_rbf, block_kwargs["num_atom"])
                for _ in range(self.num_output_blocks)
            ]
        )

    def __str__(self) -> str:
        variant = "GemNet-T" if self.triplets_only else "GemNet-Q"
        return f"""
#########################################################################
# {variant} (NeurIPS 2021) — directional message passing               #
# Atom embedding and radial Bessel stay outside this Core.            #
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
                    "GemNet output_mode='last' returns atom_feature of shape (N, F) "
                    "for SimpleReadout. HierachicalReadout reads "
                    "(N, F, num_blocks+1). Use output_mode='feature'."
                )
        for layer in built_layers:
            if isinstance(layer, BaseAtomEmbedding):
                self.atom_embedding = layer
            elif isinstance(layer, BaseRBF):
                flavor = getattr(layer, "flavor", None)
                if flavor != "gemnet":
                    raise ValueError(
                        "GemNetCore requires BesselRBF with flavor 'gemnet', "
                        f"found {flavor!r}"
                    )
                basis_cutoff = float(getattr(layer, "basis_cutoff", layer.cutoff_sr))
                if abs(basis_cutoff - self.emb_cutoff) > 1e-4:
                    raise ValueError(
                        "BesselRBF basis_cutoff "
                        f"({basis_cutoff}) must equal GemNet emb_cutoff "
                        f"({self.emb_cutoff}). Set basis_cutoff when the "
                        "neighbor list is longer than the embedding cutoff."
                    )
                self.radial_basis_function = layer

    def _quad_basis(
        self,
        dist: Tensor,
        vij: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        emb_index: Tensor,
        int_index: Tensor,
        num_nodes: int,
        rbf_emb: Tensor,
    ) -> Dict[str, Tensor]:
        assert self.quad_cbf_radial is not None
        assert self.quad_sbf_radial is not None
        assert self.mlp_rbf4 is not None
        assert self.mlp_cbf4 is not None
        assert self.mlp_sbf4 is not None
        quads = quadruplet_graph(
            idx_i, idx_j, dist, vij, emb_index, int_index, num_nodes
        )
        sbf_radial = self.quad_sbf_radial(dist[emb_index])
        if quads.edge_ca.numel() == 0:
            cbf = rbf_emb.new_empty(0, self.mlp_cbf4.linear.out_features)
            angular = rbf_emb.new_empty(0, self.quad_sbf_radial.angular_dim)
        else:
            radial_ab = self.quad_cbf_radial(dist[int_index])[quads.int_local]
            joint = joint_zonal(radial_ab, y_lm0(quads.angle_abd, self.num_spherical))
            cbf = self.mlp_cbf4(joint)
            angular = real_spherical_harmonics(
                quads.angle_cab, quads.dihedral, self.num_spherical
            )
        return {
            "rbf4": self.mlp_rbf4(rbf_emb),
            "cbf4": cbf,
            "radial_w4": self.mlp_sbf4(sbf_radial),
            "angular4": angular,
            "edge_db": quads.edge_db,
            "edge_ca": quads.edge_ca,
        }

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> Dict[str, Tensor]:
        self._check_edge_tensors(rbf, Dij_sr, vij_sr, idx_i_sr, idx_j_sr)
        num_nodes = int(atom_embedding.shape[0])
        emb_index, int_index = split_cutoff_edges(
            Dij_sr,
            self.emb_cutoff,
            self.int_cutoff,
            triplets_only=self.triplets_only,
        )
        idx_i = idx_i_sr[emb_index]
        idx_j = idx_j_sr[emb_index]
        vij = vij_sr[emb_index]
        rbf_emb = rbf[emb_index]
        swap = reverse_edge_index(idx_i, idx_j, vij, num_nodes)
        trips = triplet_graph(idx_i, idx_j, vij, num_nodes)
        angular3 = y_lm0(trips.angle, self.num_spherical)
        radial_w3 = self.mlp_cbf3(self.triplet_radial(Dij_sr[emb_index]))
        rbf3 = self.mlp_rbf3(rbf_emb)
        rbf_h = self.mlp_rbf_h(rbf_emb)
        rbf_out = self.mlp_rbf_out(rbf_emb)

        h = atom_embedding
        m = self.edge_emb(h, rbf_emb, idx_i, idx_j)
        quad: Dict[str, Tensor] = {}
        if not self.triplets_only:
            quad = self._quad_basis(
                Dij_sr, vij_sr, idx_i_sr, idx_j_sr, emb_index, int_index, num_nodes, rbf_emb
            )

        features = [self.readouts[0](m, rbf_out, idx_i, num_nodes)]
        for block, readout in zip(self.interactions, self.readouts[1:]):
            h, m = block(
                h,
                m,
                rbf3,
                rbf_h,
                radial_w3,
                angular3,
                trips.idx_edge,
                trips.idx_other,
                swap,
                idx_i,
                idx_j,
                rbf4=quad.get("rbf4"),
                cbf4=quad.get("cbf4"),
                radial_w4=quad.get("radial_w4"),
                angular4=quad.get("angular4"),
                edge_db=quad.get("edge_db"),
                edge_ca=quad.get("edge_ca"),
            )
            features.append(readout(m, rbf_out, idx_i, num_nodes))
        atom_feature = torch.stack(features, dim=-1)
        if self.output_mode == "last":
            return {"atom_feature": atom_feature[:, :, -1]}
        return {"atom_feature": atom_feature}

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
