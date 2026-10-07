"""Process-global Electrolites plugin for a GPU4PySCF calculation.

Electrolites replaces GPU4PySCF entry points in place and cannot be unloaded.
Stock and patched calculations must run in separate processes. This module
records the choice made in the current process and refuses a second, different
choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_OFF = frozenset({"", "0", "false", "off", "none", "stock", "no"})
_ALL = frozenset({"1", "true", "on", "yes", "all"})


class ElectrolitesPluginError(RuntimeError):
    """Invalid Electrolites selection, or a conflicting patch in this process."""


@dataclass(frozen=True)
class ElectrolitesPlugin:
    """``enabled=False`` is stock GPU4PySCF. ``modules is None`` means ``patch_all()``."""

    enabled: bool = False
    modules: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "modules": None if self.modules is None else list(self.modules),
        }


_PROCESS_PLUGIN: ElectrolitesPlugin | None = None


def parse_electrolites(value: Any) -> ElectrolitesPlugin:
    """Accept YAML/CLI forms: bool, ``all``, a module list, or ``{enabled, modules}``."""
    if value is None or value is False:
        return ElectrolitesPlugin(enabled=False)
    if value is True:
        return ElectrolitesPlugin(enabled=True, modules=None)
    if isinstance(value, str):
        text = value.strip()
        key = text.lower()
        if key in _OFF:
            return ElectrolitesPlugin(enabled=False)
        if key in _ALL:
            return ElectrolitesPlugin(enabled=True, modules=None)
        modules = tuple(part.strip() for part in text.split(",") if part.strip())
        if not modules:
            raise ElectrolitesPluginError("electrolites module list is empty")
        return ElectrolitesPlugin(enabled=True, modules=modules)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        modules = tuple(str(part).strip() for part in value if str(part).strip())
        if not modules:
            raise ElectrolitesPluginError("electrolites module list is empty")
        return ElectrolitesPlugin(enabled=True, modules=modules)
    if isinstance(value, Mapping):
        enabled = value.get("enabled", True)
        modules = value.get("modules", None)
        if not enabled or enabled is False:
            return ElectrolitesPlugin(enabled=False)
        if isinstance(enabled, str) and enabled.strip().lower() in _OFF:
            return ElectrolitesPlugin(enabled=False)
        if modules is None or (isinstance(modules, str) and modules.strip().lower() in _ALL):
            return ElectrolitesPlugin(enabled=True, modules=None)
        return parse_electrolites(modules)
    raise ElectrolitesPluginError(f"unrecognized electrolites spec: {value!r}")


def process_plugin() -> ElectrolitesPlugin | None:
    """The plugin applied in this process, if :func:`apply_electrolites` has run."""
    return _PROCESS_PLUGIN


def apply_electrolites(plugin: ElectrolitesPlugin) -> tuple[str, ...]:
    """Patch GPU4PySCF in this process, or require that it is still stock.

    Returns the modules Electrolites reports as patched. A later call with the
    same plugin is a no-op. A different plugin raises.
    """
    global _PROCESS_PLUGIN
    if _PROCESS_PLUGIN is not None and _PROCESS_PLUGIN != plugin:
        raise ElectrolitesPluginError(
            "Electrolites patches are process-global and cannot be changed. "
            f"This process already applied {_PROCESS_PLUGIN.to_dict()}; "
            f"start a new process for {plugin.to_dict()}."
        )
    if not plugin.enabled:
        try:
            import electrolites
        except ImportError:
            _PROCESS_PLUGIN = plugin
            return ()
        already = tuple(electrolites.patched())
        if already:
            raise ElectrolitesPluginError(
                "stock GPU4PySCF was requested, but this process already "
                f"patched {already}. Start a fresh process for the stock arm."
            )
        _PROCESS_PLUGIN = plugin
        return ()
    try:
        import electrolites
    except ImportError as exc:
        raise ElectrolitesPluginError(
            "Electrolites is enabled but the electrolites package is not installed"
        ) from exc
    if plugin.modules is None:
        patched = tuple(electrolites.patch_all())
    else:
        patched = tuple(electrolites.patch(*plugin.modules))
    _PROCESS_PLUGIN = plugin
    return patched
