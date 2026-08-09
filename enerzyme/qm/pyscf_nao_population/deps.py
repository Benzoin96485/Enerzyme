"""Runtime dependency checks for GPU4PySCF / PySCF NAO atomic population helpers."""


def check_pyscf_nao_dependencies(*, use_gpu: bool = True) -> None:
    """Verify packages needed for ``pyscf_nao_qs_prior`` preprocessing."""
    try:
        import ase  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyscf_nao_qs_prior requires ASE. Install with: pip install ase"
        ) from exc
    try:
        import pyscf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyscf_nao_qs_prior requires PySCF. Install with: pip install pyscf"
        ) from exc
    if use_gpu:
        try:
            import gpu4pyscf  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pyscf_nao_qs_prior requires gpu4pyscf when use_gpu is true. "
                "Install gpu4pyscf for your CUDA stack or set use_gpu: false in YAML."
            ) from exc
