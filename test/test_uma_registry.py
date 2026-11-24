"""Registry wiring for uma_qs without loading fairchem."""
from pathlib import Path

from enerzyme.models.presets import get_architecture


def test_ff_registers_uma_qs_string():
    spec = get_architecture("uma_qs")
    assert spec.core == ".esen:UMAWrapperQS"
    assert spec.family == "wrapper"


def test_esen_exports_uma_wrapper_symbol():
    init_src = Path("enerzyme/models/esen/__init__.py").read_text()
    assert "UMAWrapperQS" in init_src


def test_shared_readout_and_spin_layers_exported():
    init_src = Path("enerzyme/models/layers/__init__.py").read_text()
    assert "SimpleReadout" in init_src
    assert "SpinConservationLayer" in init_src


def test_ff_registers_nse_related_architectures():
    assert get_architecture("allscaip").name == "allscaip"
    assert get_architecture("uma_flow_qs").core == ".esen:UMAFlowWrapperQS"


def test_nse_layers_exported():
    init_src = Path("enerzyme/models/layers/__init__.py").read_text()
    assert "NSEReadout" in init_src
    assert "NeuralSpinChargeEquilibrationLayer" in init_src
