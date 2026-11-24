"""YAML layer adapters for the tensor-valued radial expansions."""

from ._base_layer import BaseFFLayer
from ..mp.basis import radial
from ..mp.basis.radial import (
    alignn_gaussian_terms,
    gaussian_expansion,
    gemnet_oc_gaussian,
    gemnet_oc_gaussian_terms,
    spinconv_gaussian_terms,
)


class _RadialLayer(radial.DistanceBasis, BaseFFLayer):
    """Keep the pipeline field contract and the historical checkpoint keys."""

    def _init_radial_module(self, apply_cutoff_fn: bool) -> None:
        fields = {"Dij_sr"}
        if apply_cutoff_fn:
            fields.add("cutoff_values_sr")
        BaseFFLayer.__init__(self, input_fields=fields, output_fields={"rbf"})

    forward = BaseFFLayer.forward


class BaseRBF(_RadialLayer):
    """Base type used by pipeline builders to identify a short-range basis."""


class GaussianRBFLayer(radial.GaussianBasis, BaseRBF):
    pass


class BernsteinRBFLayer(radial.BernsteinBasis, BaseRBF):
    pass


class SincRBFLayer(radial.SincBasis, BaseRBF):
    pass


class ExponentialRBF(radial.ExponentialBasis, BaseRBF):
    pass


class ExponentialGaussianRBFLayer(radial.ExponentialGaussianBasis, ExponentialRBF):
    pass


class ExponentialBernsteinRBFLayer(radial.ExponentialBernsteinBasis, ExponentialRBF):
    pass


class BesselRBFLayer(radial.BesselBasis, BaseRBF):
    pass


class ExpNormalSmearing(radial.ExpNormalBasis, _RadialLayer):
    pass


class SphericalBesselRBFLayer(radial.SphericalBesselBasis, BaseRBF):
    pass
