"""Runtime dependency checks for GFN2-xTB + xtbml atomic population helpers."""


def check_xtbml_dependencies() -> None:
    """Verify packages needed for ``xtb_qs_prior`` preprocessing. Raises ``ImportError`` with install hints."""
    try:
        import ase  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "xtb_qs_prior requires ASE. Install with: pip install ase"
        ) from exc
    try:
        from tblite.exceptions import TBLiteRuntimeError  # noqa: F401
        from tblite.interface import Calculator, Result  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "xtb_qs_prior requires tblite (GFN-xTB + xtbml post-processing). "
            "Install tblite >= 0.5 with xtbml-enabled libtblite (conda/pip per your platform)."
        ) from exc
