"""GPU4PySCF / PySCF NAO per-atom charge/spin priors for Enerzyme preprocessing."""

from .deps import check_pyscf_nao_dependencies

__all__ = ["check_pyscf_nao_dependencies", "atomic_Q_and_S_from_pyscf_nao"]


def atomic_Q_and_S_from_pyscf_nao(*args, **kwargs):
    """Run DFT + NAO populations; loads heavy deps on first use."""
    check_pyscf_nao_dependencies()
    from .atomic_populations import atomic_Q_and_S_from_pyscf_nao as _fn

    return _fn(*args, **kwargs)
