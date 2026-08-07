"""GPU4PySCF finite-step DFT driver; returns a CPU PySCF SCF object for post-processing."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyscf.gto import Mole
    from pyscf.scf.hf import SCF

XcLike = str


def run_gpu_dft(
    mol: Mole,
    *,
    xc: XcLike,
    max_cycle: int,
    conv_tol: float,
    density_fit: bool,
    verbose: int,
) -> SCF:
    """Run RKS (singlet) or UKS (open-shell) on GPU; transfer the result to CPU."""
    from gpu4pyscf import dft

    spin = int(mol.spin)
    if spin == 0:
        mf = dft.rks.RKS(mol, xc=xc)
    else:
        mf = dft.uks.UKS(mol, xc=xc)

    if density_fit:
        mf = mf.density_fit()

    mf = mf.to_gpu()
    mf.max_cycle = int(max_cycle)
    mf.conv_tol = float(conv_tol)
    mf.verbose = int(verbose)
    mf.kernel()
    return mf.to_cpu()


def run_cpu_dft(
    mol: Mole,
    *,
    xc: XcLike,
    max_cycle: int,
    conv_tol: float,
    density_fit: bool,
    verbose: int,
) -> SCF:
    """CPU PySCF fallback when GPU is unavailable."""
    from pyscf import dft

    spin = int(mol.spin)
    if spin == 0:
        mf = dft.RKS(mol, xc=xc)
    else:
        mf = dft.UKS(mol, xc=xc)

    if density_fit:
        mf = mf.density_fit()

    mf.max_cycle = int(max_cycle)
    mf.conv_tol = float(conv_tol)
    mf.verbose = int(verbose)
    mf.kernel()
    return mf


def run_dft(
    mol: Mole,
    *,
    xc: XcLike,
    max_cycle: int,
    conv_tol: float,
    density_fit: bool,
    verbose: int,
    use_gpu: bool,
) -> SCF:
    if use_gpu:
        try:
            return run_gpu_dft(
                mol,
                xc=xc,
                max_cycle=max_cycle,
                conv_tol=conv_tol,
                density_fit=density_fit,
                verbose=verbose,
            )
        except Exception as exc:
            import warnings

            warnings.warn(
                f"GPU4PySCF failed ({exc!r}); falling back to CPU PySCF.",
                RuntimeWarning,
                stacklevel=2,
            )
    return run_cpu_dft(
        mol,
        xc=xc,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        density_fit=density_fit,
        verbose=verbose,
    )
