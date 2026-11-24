"""Public name for the projected multi-head attention shared by AllScAIP paths."""

from ..mp.nonlocal_.global_attention import GlobalAttention as BaseAttention

__all__ = ["BaseAttention"]
