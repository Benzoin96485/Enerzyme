import numpy as np
import pytest
from enerzyme.tasks.calculator import ASECalculator


A = 0.08
NA = 5
NM = 4


def _calculator() -> ASECalculator:
    calc = ASECalculator.__new__(ASECalculator)
    calc.E_conversion_factor = 1.0
    return calc


def _shallow_output(R: np.ndarray, c: float) -> dict:
    return {
        "E": [np.array([[0.0]])],
        "Fa": [np.zeros_like(R)],
        "E_var": [c * np.sum(R ** 2)],
        "E_var_grad": [2 * c * R],
    }


def _ensemble_output(R: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    E = a * np.sum(R ** 2) + b * np.sum(R)
    Fa = -(2 * R[..., None] * a + b)
    return {"E": [E[None, :]], "Fa": [Fa]}


def _finite_difference_forces(calc, make_output, R, B, h=1e-6):
    forces = np.zeros_like(R)
    for i in range(R.shape[0]):
        for j in range(3):
            Rp, Rm = R.copy(), R.copy()
            Rp[i, j] += h
            Rm[i, j] -= h
            Ep = calc._calculate_UDD(make_output(Rp), A=A, B=B)[1]["energy"]
            Em = calc._calculate_UDD(make_output(Rm), A=A, B=B)[1]["energy"]
            forces[i, j] = -(Ep - Em) / (2 * h)
    return forces


@pytest.mark.parametrize("exponent", [0.1, 0.5, 2.0])
def test_udd_shallow_ensemble_force_is_negative_bias_gradient(exponent):
    calc = _calculator()
    R = np.random.default_rng(0).normal(size=(NA, 3))
    c = 1e-4
    B = c * np.sum(R ** 2) / (2 * NA * exponent)
    make_output = lambda R: _shallow_output(R, c)
    _, biases = calc._calculate_UDD(make_output(R), A=A, B=B)
    assert biases["energy"] == pytest.approx(A * (np.exp(-exponent) - 1))
    np.testing.assert_allclose(
        biases["forces"], _finite_difference_forces(calc, make_output, R, B), rtol=1e-5, atol=1e-10
    )


@pytest.mark.parametrize("exponent", [0.1, 0.5, 2.0])
def test_udd_explicit_ensemble_force_is_negative_bias_gradient(exponent):
    calc = _calculator()
    rng = np.random.default_rng(1)
    R = rng.normal(size=(NA, 3))
    a, b = rng.normal(scale=1e-2, size=NM), rng.normal(scale=1e-2, size=NM)
    make_output = lambda R: _ensemble_output(R, a, b)
    E = make_output(R)["E"][0][0]
    B = np.var(E, ddof=1) / (2 * NA * exponent)
    _, biases = calc._calculate_UDD(make_output(R), A=A, B=B)
    assert biases["energy"] == pytest.approx(A * (np.exp(-exponent) - 1))
    np.testing.assert_allclose(
        biases["forces"], _finite_difference_forces(calc, make_output, R, B), rtol=1e-5, atol=1e-10
    )
