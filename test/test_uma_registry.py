"""Registry wiring for uma_qs without loading fairchem."""
import ast
from pathlib import Path


def test_ff_registers_uma_qs_string():
    src = Path("enerzyme/models/ff.py").read_text()
    assert 'architecture.lower() == "uma_qs"' in src
    assert "from .esen import UMAWrapperQS" in src


def test_esen_exports_uma_wrapper_symbol():
    init_src = Path("enerzyme/models/esen/__init__.py").read_text()
    assert "UMAWrapperQS" in init_src


def test_shared_readout_and_spin_layers_exported():
    init_src = Path("enerzyme/models/layers/__init__.py").read_text()
    assert "SimpleReadout" in init_src
    assert "SpinConservationLayer" in init_src
