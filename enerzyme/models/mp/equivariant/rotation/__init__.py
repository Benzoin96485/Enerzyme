"""Wigner-D edge-frame rotations in the real e3nn basis."""

from .fused import EdgeRotation, OrderProjection, SO3RotationFused
from .so3_rotation import init_edge_rot_mat
from .wigner_cg import WignerD, init_edge_rot_mat_quaternion, wigner_from_cg
from .wigner_jd import wigner_from_rotation_matrix
from .wigner_quaternion import WignerDCalculator, build_edge_quaternion


def edge_rotation(unit, lmax, backend="cg", *, frame=None, mmax=None):
    """Return the forward and inverse rotations for an (E, 3) edge vector tensor.

    With an explicit Cartesian `frame`, all four backends use that frame in
    the e3nn basis. Otherwise JD and fused use the eSCN frame, CG uses the
    TECE y-axis quaternion frame, and quaternion uses DPA4's z-axis chart
    and signed basis change. Directions may have arbitrary nonzero lengths.

    By default both matrices are square and degree-major. Supplying `mmax`
    folds the m-primary permutation and truncation into the forward matrix
    and its rescaled inverse. Pass a configured module in place of a backend
    name to reuse coefficient buffers across forwards.
    """
    if not isinstance(lmax, int) or lmax < 0 or (mmax is not None and not 0 <= mmax <= lmax):
        raise ValueError(f"expected 0 <= mmax <= lmax, got {mmax=}, {lmax=}")
    if isinstance(backend, str):
        if backend == "fused" and mmax is not None:
            backend = SO3RotationFused(lmax, lmax if mmax is None else mmax).to(unit)
        elif backend == "quaternion":
            backend = WignerDCalculator(lmax)
        elif backend not in ("cg", "jd", "fused"):
            raise ValueError(f"unknown edge rotation backend {backend!r}")
    elif not isinstance(backend, (WignerDCalculator, WignerD, SO3RotationFused)):
        raise TypeError(f"unsupported edge rotation module {type(backend).__name__}")
    elif backend.lmax != lmax:
        raise ValueError(f"rotation module has lmax={backend.lmax}, requested {lmax}")

    if frame is None:
        if isinstance(backend, WignerDCalculator):
            frame = backend.cartesian_frame(build_edge_quaternion(unit))
        elif backend == "cg" or isinstance(backend, WignerD):
            frame = init_edge_rot_mat_quaternion(unit)
        else:
            frame = init_edge_rot_mat(unit)
    else:
        frame = frame.to(unit)
    if backend == "cg":
        wigner = wigner_from_cg(frame, lmax)
    elif isinstance(backend, (WignerD, SO3RotationFused)):
        wigner = backend._rotation_to_wigner_matrix(frame, 0, lmax)
    else:
        wigner = wigner_from_rotation_matrix(frame, lmax)
    if mmax is None:
        return EdgeRotation(wigner, wigner.transpose(-1, -2))
    if isinstance(backend, OrderProjection):
        if backend.mmax != mmax:
            raise ValueError(f"rotation module has mmax={backend.mmax}, requested {mmax}")
        projection = backend
    else:
        projection = OrderProjection(lmax, mmax).to(unit)
    return projection.project(wigner, autocast_half=isinstance(backend, SO3RotationFused))


__all__ = ["EdgeRotation", "edge_rotation"]
