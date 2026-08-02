"""Vendored Cartesian tensor helpers from tace v0.1.0 (MIT).

Eager exports are limited to modules needed by the Cartesian TACE Core path
(``ICTD``, ``CartesianHarmonics``, …). ``SymmetricContraction`` depends on
``opt_einsum_fx`` and is loaded lazily so a default Enerzyme install can use
``tensor_basis: cartesian`` without that extra dependency.
"""

from ._irreps import Irrep, Irreps
from ._ictd import ICTD
from ._cartesian_harmonics import CartesianHarmonics
from ._zemin import cartesian_3j
from ._utils import expand_dims_to

__all__ = [
    "Irrep",
    "Irreps",
    "CartesianHarmonics",
    "SymmetricContraction",
    "ICTD",
    "cartesian_3j",
    "expand_dims_to",
]


def __getattr__(name: str):
    if name == "SymmetricContraction":
        from ._product_basis import SymmetricContraction

        return SymmetricContraction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
