"""Coordinate model construction while e3nn changes its process-wide code generator."""
from contextlib import contextmanager
from threading import RLock

# Presets may temporarily change codegen even inside an ordinary registry build.
construction_lock = RLock()


@contextmanager
def traceable_e3nn_kernels():
    """Keep generated kernels as FX graphs during construction, then restore e3nn."""
    import e3nn

    with construction_lock:
        previous = e3nn.get_optimization_defaults()["jit_script_fx"]
        try:
            e3nn.set_optimization_defaults(jit_script_fx=False)
            yield
        finally:
            e3nn.set_optimization_defaults(jit_script_fx=previous)
