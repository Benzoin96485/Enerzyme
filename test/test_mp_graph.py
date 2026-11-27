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


def _cache_keys(ctx, seen=None):
    """Every cache key reachable from ``ctx``, including derived graphs stored as values."""
    seen = set() if seen is None else seen
    if id(ctx) in seen:
        return []
    seen.add(id(ctx))
    keys = []
    for key, value in ctx.cache.items():
        keys.append(key)
        for item in (value if isinstance(value, tuple) else (value,)):
            if isinstance(item, GraphContext):
                keys += _cache_keys(item, seen)
    return keys


def _holds_runtime_object(key):
    if isinstance(key, (torch.Tensor, GraphContext)):
        return True
    return isinstance(key, tuple) and any(_holds_runtime_object(k) for k in key)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("model", ["chgnet", "spinconv", "gemnet_oc"])
def test_merged_topology_requests_keep_their_own_configuration(model, prepared):
    from enerzyme.models.chgnet.geometry import BondGraphRequest
    from enerzyme.models.gemnet.stages_oc import GemNetOCGraphRequest
    from enerzyme.models.mp.graph import PreparedTopology
    from enerzyme.models.spinconv.geometry import SpinProjectionRequest

    requests = {
        "chgnet": (BondGraphRequest(1.0), BondGraphRequest(4.0)),
        "spinconv": (SpinProjectionRequest(1, 4.0, 4, 8), SpinProjectionRequest(3, 4.0, 6, 12)),
        "gemnet_oc": (
            GemNetOCGraphRequest((4.0, 1), None, None, None, False, False),
            GemNetOCGraphRequest((4.0, 3), (4.0, 2), None, (4.0, 2), True, True),
        ),
    }[model]
    receiver, sender, vec, dist, n = _graph(n_nodes=6, cutoff=10)
    fields = dict(idx_i_sr=receiver, idx_j_sr=sender, vij_sr=vec, Dij_sr=dist)
    base = GraphRequires(rbf=False)

    def fresh():
        return GraphContext.from_inputs(fields, base, num_nodes=n)

    expected = [request.prepare(fresh()) for request in requests]
    req = GraphRequires(rbf=False, topology=(requests[0],)).merge(
        GraphRequires(rbf=False, topology=(requests[1],))
    )
    inputs = fields.copy()
    if prepared:
        inputs["_prepared_topology"] = PreparedTopology.build(fresh(), req.topology)
    graph = GraphContext.from_inputs(inputs, req, num_nodes=n)
    for request, reference in zip(requests, expected):
        torch.testing.assert_close(request.prepare(graph), reference)
    assert not any(_holds_runtime_object(key) for key in _cache_keys(graph))


def test_graph_cache_keys_are_constants():
    # The context crosses compiler graph breaks and Dynamo guards on its cache
    # keys. A tensor or context in a key changes on every call and recompiles.
    from enerzyme.models.mp.graph import QuadrupletRequest, ReverseEdgeRequest, TripletRequest

    receiver, sender, vec, dist, n = _graph(n_nodes=6, cutoff=10)
    inputs = {"idx_i_sr": receiver, "idx_j_sr": sender, "vij_sr": vec, "Dij_sr": dist}
    req = GraphRequires(rbf=False, topology=(
        ReverseEdgeRequest(2.0), TripletRequest("receiver", edge_cutoff=2.0, cutoff=1.5, strict=True),
        QuadrupletRequest(2.0, 2.5)))
    ctx = GraphContext.from_inputs(inputs, req, n)
    near, far = ctx.within(2.0), ctx.within(2.5)
    near.cross_pairs(far)
    near.quadruplets(far)
    ctx.triplets("receiver", cutoff=1.2)
    capped = ctx.closest(2.5, 2)
    capped.triplets("receiver")
    named = ctx.subset(torch.arange(3), key="first")

    keys = _cache_keys(ctx)
    assert keys and not [k for k in keys if _holds_runtime_object(k)]
    assert near.key == (("within", 2.0),) and capped.key == (("closest", 2.5, 2),)
    assert ctx.within(2.0) is near and ctx.closest(2.5, 2) is capped
    assert ctx.subset(torch.arange(3), key="first") is named
    assert ctx.subset(torch.arange(3)) is not ctx.subset(torch.arange(3))
    assert near.quadruplets(far) is near.quadruplets(ctx.within(2.5))
    unnamed = GraphContext(receiver, sender, n, vec=vec)
    assert unnamed.key is None and unnamed.within(2.0).key is None


def test_graph_context_reused_after_a_graph_break_does_not_recompile():
    import torch._dynamo

    receiver, sender, vec, dist, n = _graph(n_nodes=6, cutoff=10)
    req = GraphRequires(rbf=False)

    def evaluate(receiver, sender, vec):
        ctx = GraphContext.from_inputs({"idx_i_sr": receiver, "idx_j_sr": sender, "vij_sr": vec}, req, n)
        near = ctx.within(2.0)
        pairs = near.triplets("receiver", cutoff=1.5)
        capped = ctx.closest(2.5, 3)
        torch._dynamo.graph_break()
        return near.vec.sum() + capped.vec.sum() + pairs.first.numel()

    torch._dynamo.reset()
    compiled = torch.compile(evaluate, backend="eager")
    expected = evaluate(receiver, sender, vec)
    compiled(receiver.clone(), sender.clone(), vec.clone())
    with torch._dynamo.config.patch(error_on_recompile=True):
        for _ in range(3):
            torch.testing.assert_close(compiled(receiver.clone(), sender.clone(), vec.clone()), expected)
    torch._dynamo.reset()


@pytest.mark.parametrize("prepared", [False, True])
def test_cross_graph_tables_distinguish_independent_roots(prepared):
    from enerzyme.models.mp.graph import PreparedTopology, QuadrupletRequest, cross_receiver_pairs
    from enerzyme.models.mp.quadruplets import dihedral_quadruplets

    receiver, sender, vec, dist, n = _graph(n_nodes=4, cutoff=10)
    request = QuadrupletRequest(9., 10.)
    requires = GraphRequires(rbf=False, topology=(request,))

    def root(selection):
        fields = {"idx_i_sr": receiver[selection], "idx_j_sr": sender[selection],
                  "vij_sr": vec[selection], "Dij_sr": dist[selection]}
        graph = GraphContext.from_inputs(fields, requires, n)
        if prepared:
            fields["_prepared_topology"] = PreparedTopology.build(graph, requires.topology)
            graph = GraphContext.from_inputs(fields, requires, n)
        return graph

    first = root(slice(None))
    query, local = first.within(9.), first.within(10.)
    foreign = root(slice(0, 3)).within(10.)
    assert local.key == foreign.key
    query.cross_pairs(local)
    local_quads = query.quadruplets(local)
    expected_pairs = cross_receiver_pairs(query.receiver, foreign.receiver, n)
    expected_quads = dihedral_quadruplets(query, foreign)
    assert len(local_quads.int_local) > len(expected_quads.int_local) > 0
    torch.testing.assert_close(query.cross_pairs(foreign), expected_pairs)
    torch.testing.assert_close(query.quadruplets(foreign), expected_quads)
    assert query.quadruplets(local) is local_quads


def test_fullgraph_distinguishes_roots_sharing_the_same_input_tensors():
    requires = GraphRequires(vectors=False, distances=False, rbf=False)

    @torch.compile(backend="eager", fullgraph=True)
    def roots(receiver, sender):
        inputs = {"idx_i_sr": receiver, "idx_j_sr": sender}
        first = GraphContext.from_inputs(inputs, requires, 3)
        second = GraphContext.from_inputs(inputs, requires, 3)
        child = first.subset(torch.tensor([0, 1]), key="first")
        assert first._relative(second) is None
        assert first._relative(child) == child.key
        return child.receiver + child.sender

    receiver, sender = torch.tensor([0, 1, 0]), torch.tensor([1, 2, 2])
    torch.testing.assert_close(roots(receiver, sender), torch.tensor([1, 3]))
    with torch._dynamo.config.patch(error_on_recompile=True):
        torch.testing.assert_close(roots(receiver.clone(), sender.clone()), torch.tensor([1, 3]))


def test_fullgraph_external_contexts_can_alternate_between_related_and_independent():
    requires = GraphRequires(vectors=False, distances=False, rbf=False)
    fields = {"idx_i_sr": torch.tensor([0, 1]), "idx_j_sr": torch.tensor([1, 0])}
    first = GraphContext.from_inputs(fields, requires, 2)
    other = GraphContext.from_inputs(fields, requires, 2)
    local = first.subset(torch.tensor([0]), key="edge")
    foreign = other.subset(torch.tensor([0]), key="edge")

    @torch.compile(backend="eager", fullgraph=True)
    def cacheable(graph, view, value):
        return value if graph._relative(view) is None else value + 1

    value = torch.tensor(0)
    for view, expected in ((local, 1), (foreign, 0), (local, 1)):
        assert cacheable(first, view, value).item() == expected
