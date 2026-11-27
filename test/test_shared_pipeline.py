import numpy as np
import pytest
import torch

from enerzyme.data.neighbor_list import full_neighbor_list
from enerzyme.models.layers.geometry import RangeSeparationLayer
from enerzyme.tasks.batch import _decorate_batch_input, _decorate_batch_output


@pytest.mark.parametrize("size", [0, 1, 2, 15, 128])
def test_full_pairs_in_receiver_order(size):
    receiver, sender = full_neighbor_list(np.int64(size))
    expected = [(i, j) for i in range(size) for j in range(size) if i != j]
    assert receiver.dtype == sender.dtype == np.int64
    assert list(zip(receiver.tolist(), sender.tolist())) == expected


def test_batch_results_and_input_ownership():
    # The assembled CPU batch owns its storage even if individual inputs are views.
    coords = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    rows = [({"N": n, "Ra": coords[:n], "Za": np.ones(n, dtype=np.int64)},
             {"E": 1., "Fa": coords[:n]}, str(i)) for i, n in enumerate((2, 0, 4))]
    features, targets = _decorate_batch_input(rows, torch.float64, torch.device("cpu"))
    original = coords.copy()
    coords[:] = -1
    assert features["Ra"].is_leaf and features["Ra"].requires_grad
    actual, truth = _decorate_batch_output({"Fa": features["Ra"]}, features, targets)
    assert actual["data_key"] == truth["data_key"] == ("0", "1", "2")
    for i, n in enumerate((2, 0, 4)):
        np.testing.assert_array_equal(actual["Fa"][i], original[:n])
        np.testing.assert_array_equal(truth["Fa"][i], original[:n])


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("cutoff", [0., 1.5, 10.])
def test_selected_edges_keep_order_and_coordinate_gradients(device, cutoff):
    layer = RangeSeparationLayer(cutoff)
    distance = torch.tensor([2., 0.5, 1.5, 1.], dtype=torch.float64, device=device, requires_grad=True)
    vec = distance[:, None].expand(-1, 3).square()
    src = torch.tensor([1, 1, 0, 2], device=device)
    dst = torch.tensor([0, 2, 1, 1], device=device)
    mask = distance < cutoff
    actual = layer.get_output(distance, src, dst, vec)
    expected = {"Dij_sr": distance[mask], "idx_i_sr": src[mask], "idx_j_sr": dst[mask], "vij_sr": vec[mask]}
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for order in (1, 2):
        def derivative(parts):
            energy = parts["Dij_sr"].square().sum() + parts["vij_sr"].sum()
            grad, = torch.autograd.grad(energy, distance, retain_graph=True, create_graph=True)
            if order == 2:
                grad, = torch.autograd.grad(grad.sum(), distance, retain_graph=True)
            return grad
        torch.testing.assert_close(derivative(actual), derivative(expected), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vector_distance_keeps_directed_epsilon_and_mixed_derivatives(dtype):
    from enerzyme.models.layers.geometry import DistanceLayer

    coords = torch.tensor([[0, 0, 0], [1., -2., 0.5]], dtype=dtype, requires_grad=True)
    offsets = torch.tensor([[0, 0, 0], [1e-15, 0, -1e-15], [1., 2., 0]], dtype=dtype, requires_grad=True)
    i, j = torch.tensor([0, 0, 1]), torch.tensor([1, 0, 0])
    layer = DistanceLayer()
    layer.with_vector_on()
    out = layer.get_output(coords, i, j, offsets)
    right = coords[j] + offsets
    expected = torch.nn.functional.pairwise_distance(coords[i], right, eps=1e-15)
    torch.testing.assert_close(out["Dij"], expected, rtol=0, atol=0)
    torch.testing.assert_close(out["vij"], right - coords[i], rtol=0, atol=0)

    def derivatives(d, v):
        energy = (d.square() + (v.sin() * d[:, None]).sum(dim=1)).sum()
        forces, = torch.autograd.grad(energy, coords, create_graph=True, retain_graph=True)
        mixed, = torch.autograd.grad(forces.square().sum(), offsets, retain_graph=True)
        return forces, mixed

    actual = derivatives(out["Dij"], out["vij"])
    reference = derivatives(expected, right - coords[i])
    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_vector_distance_preserves_autocast_for_low_precision_coordinates(device):
    from enerzyme.models.layers.geometry import DistanceLayer

    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    coords = torch.tensor([[0, 0, 0], [1.25, 0.5, 1.]], device=device, dtype=dtype,
                          requires_grad=True)
    i = torch.tensor([0, 0, 1], device=device)
    j = torch.tensor([0, 1, 0], device=device)
    layer = DistanceLayer()
    layer.with_vector_on()
    with torch.autocast(device_type=device, dtype=dtype):
        actual = layer.get_output(coords, i, j)
        expected = torch.nn.functional.pairwise_distance(coords[i], coords[j], eps=1e-15)
    assert list(actual) == ["Dij", "vij"]
    torch.testing.assert_close(actual["Dij"], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual["vij"], coords[j] - coords[i], rtol=0, atol=0)
