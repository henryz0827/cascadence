"""Flow solver checks against analytic circuits and conservation laws."""

from __future__ import annotations

import numpy as np
import pytest

from cascadence.network import FlowNetwork, lattice_network


def test_series_conductances():
    """Two conductances in series: same flow, effective conductance g1g2/(g1+g2)."""
    g1, g2 = 2.0, 3.0
    net = FlowNetwork(
        edges=[[0, 1], [1, 2]],
        conductance=[g1, g2],
        volume=[1.0, 1.0],
        inlets=[0],
        outlets=[2],
    )
    state = net.solve()
    g_eff = g1 * g2 / (g1 + g2)

    assert state.flow[0] == pytest.approx(g_eff)
    assert state.flow[1] == pytest.approx(g_eff)
    assert net.total_flow(state) == pytest.approx(g_eff)
    # Pressure at the midpoint follows from current continuity, not from g_eff:
    # g1 (1 - p) = g2 p  =>  p = g1 / (g1 + g2).
    assert state.pressure[1] == pytest.approx(g1 / (g1 + g2))


def test_parallel_conductances():
    g1, g2 = 2.0, 3.0
    net = FlowNetwork(
        edges=[[0, 1], [0, 1]],
        conductance=[g1, g2],
        volume=[1.0, 1.0],
        inlets=[0],
        outlets=[1],
    )
    state = net.solve()
    assert state.flow[0] == pytest.approx(g1)
    assert state.flow[1] == pytest.approx(g2)
    assert net.total_flow(state) == pytest.approx(g1 + g2)


def test_transit_time_is_volume_over_flow():
    net = FlowNetwork(
        edges=[[0, 1]],
        conductance=[2.0],
        volume=[5.0],
        inlets=[0],
        outlets=[1],
        p_in=3.0,
        p_out=1.0,
    )
    state = net.solve()
    assert state.flow[0] == pytest.approx(4.0)  # g * dp
    assert state.transit_time[0] == pytest.approx(5.0 / 4.0)


def test_current_conservation_on_lattice():
    rng = np.random.default_rng(0)
    net = lattice_network((12, 10), rng)
    state = net.solve()

    divergence = np.zeros(net.n_nodes)
    np.add.at(divergence, net.edges[:, 0], state.flow)
    np.add.at(divergence, net.edges[:, 1], -state.flow)

    interior = np.ones(net.n_nodes, dtype=bool)
    interior[net.inlets] = False
    interior[net.outlets] = False
    assert np.abs(divergence[interior]).max() < 1e-10
    # What leaves the inlets must arrive at the outlets.
    assert net.total_flow(state) == pytest.approx(-divergence[net.outlets].sum())
    assert net.total_flow(state) > 0.0


def test_boundary_face_edges_are_never_perfused():
    """Edges joining two same-face boundary nodes sit at zero pressure drop.

    Both endpoints are pinned to the same Dirichlet value, so these segments
    carry no flow by construction and can never take part in a cascade. Not a
    defect, but it means ``n_perfused < n_edges`` even in a pristine network.
    """
    rng = np.random.default_rng(0)
    net = lattice_network((12, 10), rng)
    state = net.solve()

    on_inlet_face = np.isin(net.edges, net.inlets).all(axis=1)
    on_outlet_face = np.isin(net.edges, net.outlets).all(axis=1)
    same_face = on_inlet_face | on_outlet_face

    assert same_face.any()
    assert not state.perfused[same_face].any()
    assert state.n_perfused == net.n_edges - int(np.count_nonzero(same_face))


def test_disconnection_yields_zero_flow_not_a_singular_solve():
    """Cutting the inlet face must give zero flow everywhere, without blowing up."""
    rng = np.random.default_rng(0)
    net = lattice_network((10, 8), rng)
    blocked = np.isin(net.edges[:, 0], net.inlets) | np.isin(
        net.edges[:, 1], net.inlets
    )
    state = net.solve(blocked)

    assert state.n_perfused == 0
    assert net.total_flow(state) == pytest.approx(0.0)
    assert np.all(np.isinf(state.transit_time))
    assert np.all(np.isfinite(state.pressure))


def test_partial_blocking_reduces_throughput_monotonically():
    rng = np.random.default_rng(5)
    net = lattice_network((14, 12), rng)
    baseline = net.solve()
    throughput = net.total_flow(baseline)

    blocked = np.zeros(net.n_edges, dtype=bool)
    order = rng.permutation(np.flatnonzero(baseline.perfused))[:25]
    previous = throughput
    for edge in order:
        blocked[edge] = True
        current = net.total_flow(net.solve(blocked))
        assert current <= previous + 1e-12
        previous = current
    assert previous < throughput


def test_input_validation():
    with pytest.raises(ValueError, match="self-loops"):
        FlowNetwork(
            edges=[[0, 0]], conductance=[1.0], volume=[1.0], inlets=[0], outlets=[1]
        )
    with pytest.raises(ValueError, match="strictly positive"):
        FlowNetwork(
            edges=[[0, 1]], conductance=[0.0], volume=[1.0], inlets=[0], outlets=[1]
        )
    with pytest.raises(ValueError, match="disjoint"):
        FlowNetwork(
            edges=[[0, 1]], conductance=[1.0], volume=[1.0], inlets=[0], outlets=[0]
        )
    with pytest.raises(ValueError, match="p_in must exceed"):
        FlowNetwork(
            edges=[[0, 1]],
            conductance=[1.0],
            volume=[1.0],
            inlets=[0],
            outlets=[1],
            p_in=0.0,
            p_out=1.0,
        )
    with pytest.raises(ValueError, match="shape"):
        FlowNetwork(
            edges=[[0, 1]],
            conductance=[1.0, 2.0],
            volume=[1.0],
            inlets=[0],
            outlets=[1],
        )


def test_lattice_three_dimensional():
    rng = np.random.default_rng(4)
    net = lattice_network((6, 5, 5), rng)
    state = net.solve()
    assert net.n_nodes == 150
    assert net.total_flow(state) > 0.0
    assert state.n_perfused > 0
