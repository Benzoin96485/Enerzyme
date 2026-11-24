import pytest
import torch

from enerzyme.models.mp.graph import (
    GraphContext,
    GraphRequires,
    path_pairs,
    receiver_csr,
    receiver_pairs,
    reverse_edge_index,
)


def _graph(n_nodes=6, seed=0, cutoff=2.2):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n_nodes, 3, generator=g, dtype=torch.float64) * 3.0
    i, j = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing="ij")
    keep = i != j
    receiver, sender = i[keep], j[keep]
    vec = pos[sender] - pos[receiver]
    dist = vec.norm(dim=-1)
    inside = dist < cutoff
    perm = torch.randperm(int(inside.sum()), generator=g)
    return receiver[inside][perm], sender[inside][perm], vec[inside][perm], dist[inside][perm], n_nodes


def _receiver_pairs_ref(receiver, subset, include_self):
    edges = list(range(len(receiver))) if subset is None else subset.tolist()
    by_node = {}
    for e in sorted(edges, key=lambda e: (int(receiver[e]), e)):
        by_node.setdefault(int(receiver[e]), []).append(e)
    first, second = [], []
    for e in edges:
        for o in by_node[int(receiver[e])]:
            if o == e and not include_self:
                continue
            first.append(e)
            second.append(o)
    return first, second


@pytest.mark.parametrize("include_self", [False, True])
@pytest.mark.parametrize("use_subset", [False, True])
def test_receiver_pairs_order(include_self, use_subset):
    receiver, sender, vec, dist, n = _graph()
    subset = torch.where(dist <= 1.6)[0] if use_subset else None
    pairs = receiver_pairs(receiver, n, subset, include_self)
    ref = _receiver_pairs_ref(receiver, subset, include_self)
    assert pairs.first.tolist() == ref[0]
    assert pairs.second.tolist() == ref[1]


def test_path_pairs_orders_cover_same_set():
    receiver, sender, vec, dist, n = _graph()
    out = path_pairs(receiver, sender, n, "outgoing")
    atom = path_pairs(receiver, sender, n, "atom")
    ref_out = []
    for e_out in range(len(receiver)):
        j = int(sender[e_out])
        for e_in in sorted(range(len(receiver)), key=lambda e: (int(receiver[e]), e)):
            if int(receiver[e_in]) == j:
                ref_out.append((e_in, e_out))
    assert list(zip(out.first.tolist(), out.second.tolist())) == ref_out
    assert sorted(zip(atom.first.tolist(), atom.second.tolist())) == sorted(ref_out)
    assert torch.equal(receiver[atom.first], sender[atom.second])
    assert torch.all(atom.first[1:] * 0 + receiver[atom.first][1:] >= receiver[atom.first][:-1])


def test_reverse_edge_index_is_involution():
    receiver, sender, vec, dist, n = _graph()
    rev = reverse_edge_index(receiver, sender, vec, n)
    assert torch.equal(receiver[rev], sender) and torch.equal(sender[rev], receiver)
    assert torch.allclose(vec[rev], -vec)
    assert torch.equal(rev[rev], torch.arange(len(receiver)))


def test_receiver_csr_groups_edges():
    receiver, *_ , n = _graph()
    counts, offsets, perm = receiver_csr(receiver, n)
    for a in range(n):
        group = perm[offsets[a]:offsets[a + 1]]
        assert torch.all(receiver[group] == a)
        assert len(group) == int(counts[a])


def test_context_swap_and_cache():
    receiver, sender, vec, dist, n = _graph()
    inputs = dict(idx_i_sr=receiver, idx_j_sr=sender, vij_sr=vec, Dij_sr=dist, rbf=torch.zeros(len(dist), 4))
    ctx = GraphContext.from_inputs(inputs, GraphRequires(), num_nodes=n)
    swapped = GraphContext.from_inputs(inputs, GraphRequires(edge_convention="swap"), num_nodes=n)
    assert torch.equal(swapped.receiver, sender) and torch.equal(swapped.sender, receiver)
    assert torch.equal(swapped.vec, -vec)
    assert ctx.receiver_pairs(1.6) is ctx.receiver_pairs(1.6)
    sh = ctx.spherical_harmonics(2)
    assert sh.shape == (len(dist), 9)
    assert ctx.spherical_harmonics(2) is sh
    assert GraphRequires().input_fields() >= {"idx_i_sr", "idx_j_sr", "vij_sr", "Dij_sr", "rbf"}


def test_cross_receiver_pairs_matches_bruteforce():
    from enerzyme.models.mp.graph import cross_receiver_pairs

    torch.manual_seed(3)
    n = 6
    query = torch.randint(0, n, (17,))
    key = torch.randint(0, n, (23,))
    first, second = cross_receiver_pairs(query, key, n)
    key_order = sorted(range(len(key)), key=lambda e: (int(key[e]), e))
    expected = [(q, k) for q in range(len(query)) for k in key_order if key[k] == query[q]]
    assert list(zip(first.tolist(), second.tolist())) == expected


def test_receiver_rank_is_stable_knn():
    from enerzyme.models.mp.graph import receiver_rank

    receiver = torch.tensor([0, 1, 0, 0, 1, 2, 0])
    key = torch.tensor([0.5, 0.2, 0.1, 0.5, 0.1, 9.0, 0.3])
    rank = receiver_rank(receiver, key, 3)
    assert rank.tolist() == [2, 1, 0, 3, 0, 0, 1]


def test_declared_topology_prepares_once_and_reuses_receiver_csr(monkeypatch):
    from enerzyme.models.mp import graph
    from enerzyme.models.mp.graph import QuadrupletRequest, ReverseEdgeRequest, TripletRequest

    receiver, sender, vec, dist, n = _graph(n_nodes=4, cutoff=10)
    vec.requires_grad_()
    inputs = {"idx_i_sr": receiver, "idx_j_sr": sender, "vij_sr": vec, "Dij_sr": dist}
    calls = []
    original = graph.receiver_csr

    def counted(receiver, num_nodes):
        calls.append(receiver)
        return original(receiver, num_nodes)

    monkeypatch.setattr(graph, "receiver_csr", counted)
    req = GraphRequires(rbf=False, topology=(
        ReverseEdgeRequest(10), TripletRequest("receiver", edge_cutoff=10), QuadrupletRequest(10, 10)))
    ctx = GraphContext.from_inputs(inputs, req, n)
    sub = ctx.within(10)
    pairs = sub.triplets("receiver")
    quads = sub.quadruplets(sub)
    assert sub.triplets("receiver") is pairs
    assert sub.quadruplets() is quads
    assert len(calls) == 1
    actual = set(zip(quads.atom_c.tolist(), quads.atom_a.tolist(), quads.atom_b.tolist(), quads.atom_d.tolist()))
    from itertools import permutations
    assert actual == set(permutations(range(n), 4))
    assert quads.edge_ca.numel() == len(actual)
    grad, = torch.autograd.grad(quads.dihedral.sum(), vec)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_prepared_quadruplets_recompute_angles_and_their_derivatives():
    from enerzyme.models.mp.graph import PreparedTopology, QuadrupletRequest

    receiver, sender, vec, dist, n = _graph(n_nodes=4, cutoff=10)
    req = GraphRequires(rbf=False, topology=(QuadrupletRequest(10., 10.),))
    original = GraphContext(receiver, sender, n, vec=vec, dist=dist)
    with torch.no_grad():
        topology = PreparedTopology.build(original, req.topology)
    vectors = (vec * vec.new_tensor([1.2, 0.8, 0.9])).requires_grad_()
    fields = dict(idx_i_sr=receiver, idx_j_sr=sender, vij_sr=vectors, Dij_sr=vectors.norm(dim=-1))

    def evaluate(inputs):
        view = GraphContext.from_inputs(inputs, req, n).within(10.)
        angles = view.quadruplets(view).dihedral
        grad, = torch.autograd.grad(angles.square().sum(), vectors, create_graph=True)
        hessian, = torch.autograd.grad(grad.square().sum(), vectors)
        return angles, grad, hessian

    reference = evaluate(fields)
    actual = evaluate({**fields, "_prepared_topology": topology})
    for before, after in zip(reference, actual):
        torch.testing.assert_close(after, before)


def test_periodic_line_graph_backtracking_and_duplicate_reverse_edges():
    r = torch.tensor([0, 1, 0, 1])
    s = torch.tensor([1, 0, 1, 0])
    x = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [2., 0., 0.], [-2., 0., 0.]])
    graph = GraphContext(r, s, 2, vec=x)
    assert graph.triplets("path", exclude_backtracking=True).first.numel() == 0
    first, second = graph.triplets("line_graph", exclude_backtracking=True)
    assert first.numel() == 4
    assert ((x[first] + x[second]).norm(dim=1) > 0).all()
    # The same bond image has been listed twice in both directions.
    dup = GraphContext(r, s, 2, vec=x[[0, 1, 0, 1]])
    reverse = dup.reverse_edge(distinct=True)
    assert torch.equal(reverse[reverse], torch.arange(4))
    assert torch.equal(reverse, torch.tensor([1, 0, 3, 2]))
