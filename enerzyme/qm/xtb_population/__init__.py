"""GFN2-xTB + xtbml per-atom charge/spin populations (Mulliken-type) for Enerzyme preprocessing."""

from .deps import check_xtbml_dependencies

__all__ = ["check_xtbml_dependencies", "atomic_Q_and_S_from_xtbml"]


def atomic_Q_and_S_from_xtbml(*args, **kwargs):
    """Run xTB + xtbml; loads tblite on first use after :func:`check_xtbml_dependencies`."""
    check_xtbml_dependencies()
    from .atomic_populations import atomic_Q_and_S_from_xtbml as _fn

    return _fn(*args, **kwargs)
