"""Architecture registry and public imports for building and training force fields."""

from typing import Any, Callable, Dict, List, Tuple

from . import layers as Layers
from .launcher import BaseFFLauncher, FFDataset, FF_committee, FF_single, MetaStateDict
from .loss import LOSS_REGISTER
from .pipeline import build_layer, build_model

__all__ = [
    "SEP", "BaseFFLauncher", "FFDataset", "FF_committee", "FF_single",
    "MetaStateDict", "build_layer", "build_model", "get_ff_core",
]


SEP = "-"

# Keys present before any architecture is loaded. ``get_ff_core`` may add a
# loss that belongs to one architecture; the next architecture drops those
# keys so they do not stay available for a different model.
_BASE_LOSS_KEYS = frozenset(LOSS_REGISTER)
_ARCH_LOSS_KEYS: set = set()


def _install_architecture_losses(special_loss: Dict[str, Any]) -> None:
    global _ARCH_LOSS_KEYS
    for key in _ARCH_LOSS_KEYS:
        LOSS_REGISTER.pop(key, None)
    LOSS_REGISTER.update(special_loss)
    _ARCH_LOSS_KEYS = {key for key in special_loss if key not in _BASE_LOSS_KEYS}


def get_ff_core(architecture: str) -> Tuple[Callable[..., Layers.BaseFFCore], Dict[str, Any], List[Dict[str, Any]]]:
    """Core factory, default ``build_params`` and default ``layers`` of ``architecture``.

    Names and aliases are resolved by :mod:`enerzyme.models.presets`.
    """
    from .presets import get_architecture

    core, build_params, layer_params, special_loss = get_architecture(architecture).resolve()
    _install_architecture_losses(special_loss)
    return core, build_params, layer_params
