"""Conductance networks under a fixed pressure drop, with edge removal.

A :class:`FlowNetwork` is a weighted graph carrying Poiseuille flow between
Dirichlet pressure boundaries. Blocking an edge sets its conductance to zero and
the whole pressure field is re-solved, which is what makes the redistribution in
:mod:`cascadence.cascade` *long-ranged*: removing one edge changes flows
everywhere, not just at its neighbours.

Two transport quantities matter downstream:

``flows``
    Volumetric flow ``q_e = g_e * (p_u - p_v)`` on every edge.
``transit_times``
    ``T_e = V_e / |q_e|``, the time a fluid element spends in segment ``e``.
    Volume over volumetric flow rate -- not length over velocity -- because it
    stays dimensionally correct without committing to a cross-sectional area
    convention.

Removing edges can disconnect parts of the graph, which makes the graph
Laplacian singular. Rather than regularising it, :meth:`FlowNetwork.solve`
restricts the linear solve to the free nodes that still have a path to a
pressure boundary; everything cut off carries exactly zero flow, which is the
physically right answer and keeps the system non-singular.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

__all__ = ["FlowNetwork", "FlowState", "lattice_network"]

# Flows below this fraction of the baseline mean are treated as stagnant. A
# disconnected subtree solves to exactly 0.0, but a subtree hanging off a
# near-blocked path solves to a denormal-ish value that is physically stasis.
_STAGNANT_FRACTION = 1e-9


@dataclass(frozen=True)
class FlowState:
    """Solved pressures, flows and transit times for one blocking configuration."""

    pressure: np.ndarray
    flow: np.ndarray
    transit_time: np.ndarray
    perfused: np.ndarray

    @property
    def n_perfused(self) -> int:
        return int(np.count_nonzero(self.perfused))


@dataclass
class FlowNetwork:
    """Undirected conductance network with fixed inlet and outlet pressures.

    Parameters
    ----------
    edges
        ``(E, 2)`` integer array of node index pairs. Parallel edges are
        allowed; self-loops are not.
    conductance
        ``(E,)`` positive conductances ``g_e``.
    volume
        ``(E,)`` positive segment volumes ``V_e``, used for transit times.
    inlets, outlets
        Node indices held at ``p_in`` and ``p_out``. Must be disjoint.
    """

    edges: np.ndarray
    conductance: np.ndarray
    volume: np.ndarray
    inlets: np.ndarray
    outlets: np.ndarray
    p_in: float = 1.0
    p_out: float = 0.0
    n_nodes: int = field(default=-1)

    def __post_init__(self) -> None:
        self.edges = np.asarray(self.edges, dtype=np.int64)
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must have shape (E, 2)")
        self.conductance = np.asarray(self.conductance, dtype=float)
        self.volume = np.asarray(self.volume, dtype=float)
        self.inlets = np.atleast_1d(np.asarray(self.inlets, dtype=np.int64))
        self.outlets = np.atleast_1d(np.asarray(self.outlets, dtype=np.int64))

        if self.conductance.shape != (self.n_edges,):
            raise ValueError("conductance must have shape (E,)")
        if self.volume.shape != (self.n_edges,):
            raise ValueError("volume must have shape (E,)")
        if np.any(self.conductance <= 0.0):
            raise ValueError("conductances must be strictly positive")
        if np.any(self.volume <= 0.0):
            raise ValueError("volumes must be strictly positive")
        if np.any(self.edges[:, 0] == self.edges[:, 1]):
            raise ValueError("self-loops are not allowed")
        if np.intersect1d(self.inlets, self.outlets).size:
            raise ValueError("inlets and outlets must be disjoint")
        if self.p_in <= self.p_out:
            raise ValueError("p_in must exceed p_out")

        implied = int(self.edges.max()) + 1 if self.n_edges else 0
        if self.n_nodes < 0:
            self.n_nodes = max(
                implied, int(self.inlets.max()) + 1, int(self.outlets.max()) + 1
            )
        elif self.n_nodes < implied:
            raise ValueError("n_nodes is smaller than the largest node index in edges")

        self._boundary = np.zeros(self.n_nodes, dtype=bool)
        self._boundary[self.inlets] = True
        self._boundary[self.outlets] = True

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def _laplacian(self, g: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
        """Weighted adjacency and Laplacian over the edges with ``g > 0``."""
        active = g > 0.0
        u = self.edges[active, 0]
        v = self.edges[active, 1]
        w = g[active]
        n = self.n_nodes
        upper = coo_matrix((w, (u, v)), shape=(n, n))
        adjacency = (upper + upper.T).tocsr()
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        return adjacency, (diags(degree) - adjacency).tocsr()

    def solve(self, blocked: np.ndarray | None = None) -> FlowState:
        """Solve for pressures and flows with ``blocked`` edges removed."""
        g = self.conductance.copy()
        if blocked is not None:
            blocked = np.asarray(blocked, dtype=bool)
            if blocked.shape != (self.n_edges,):
                raise ValueError("blocked must have shape (E,)")
            g[blocked] = 0.0

        adjacency, laplacian = self._laplacian(g)

        pressure = np.zeros(self.n_nodes, dtype=float)
        pressure[self.inlets] = self.p_in
        pressure[self.outlets] = self.p_out

        # Only free nodes still joined to a pressure boundary have a determined
        # pressure; the rest form floating islands that carry no flow.
        _, labels = connected_components(adjacency, directed=False)
        anchored = np.isin(labels, np.unique(labels[self._boundary]))
        free = (~self._boundary) & anchored

        if free.any():
            lff = laplacian[free][:, free].tocsc()
            lfb = laplacian[free][:, self._boundary]
            rhs = -(lfb @ pressure[self._boundary])
            pressure[free] = spsolve(lff, rhs)

        flow = np.zeros(self.n_edges, dtype=float)
        active = g > 0.0
        flow[active] = g[active] * (
            pressure[self.edges[active, 0]] - pressure[self.edges[active, 1]]
        )

        magnitude = np.abs(flow)
        scale = magnitude.max()
        threshold = _STAGNANT_FRACTION * scale if scale > 0.0 else 0.0
        perfused = magnitude > threshold

        with np.errstate(divide="ignore", invalid="ignore"):
            transit = np.where(perfused, self.volume / magnitude, np.inf)

        return FlowState(
            pressure=pressure, flow=flow, transit_time=transit, perfused=perfused
        )

    def total_flow(self, state: FlowState) -> float:
        """Net volumetric flow out of the inlets -- the network's throughput."""
        incident_out = np.isin(self.edges[:, 0], self.inlets)
        incident_in = np.isin(self.edges[:, 1], self.inlets)
        return float(
            np.sum(state.flow[incident_out]) - np.sum(state.flow[incident_in])
        )


def lattice_network(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    *,
    radius_median: float = 1.0,
    radius_log_sd: float = 0.3,
    viscosity: float = 1.0,
    segment_length: float = 1.0,
    periodic_transverse: bool = True,
) -> FlowNetwork:
    """Synthetic lattice with lognormal radii, a stand-in for a capillary bed.

    Flow is driven along axis 0: the ``x = 0`` face is the inlet, the
    ``x = shape[0] - 1`` face the outlet. Transverse axes are periodic by
    default, which removes the side-wall artefacts that would otherwise show up
    as a spurious length scale in the avalanche statistics.

    Conductances follow Poiseuille, ``g = pi r**4 / (8 mu L)``, and volumes
    ``V = pi r**2 L``. The ``r**4`` law is what makes the radius distribution
    matter so much: a lognormal spread of ``radius_log_sd`` in ``r`` becomes a
    spread of ``4 * radius_log_sd`` in ``g``.

    This geometry is a placeholder. It is deliberately *not* claimed to
    resemble any real microvascular bed -- swap in measured geometry before
    reading anything physiological off the results.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        raise ValueError("shape must have at least 2 dimensions")
    if any(s < 2 for s in shape):
        raise ValueError("every axis must have length >= 2")

    index = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    edge_list: list[np.ndarray] = []
    for axis in range(len(shape)):
        head = index
        tail = np.roll(index, -1, axis=axis)
        wrap_last = axis > 0 and periodic_transverse
        if not wrap_last:
            # Drop the wrap-around slice on non-periodic axes.
            slicer = [slice(None)] * len(shape)
            slicer[axis] = slice(0, shape[axis] - 1)
            head = head[tuple(slicer)]
            tail = tail[tuple(slicer)]
        elif shape[axis] == 2:
            # With only two sites, wrapping would duplicate the single edge.
            slicer = [slice(None)] * len(shape)
            slicer[axis] = slice(0, 1)
            head = head[tuple(slicer)]
            tail = tail[tuple(slicer)]
        edge_list.append(np.stack([head.ravel(), tail.ravel()], axis=1))

    edges = np.concatenate(edge_list, axis=0)
    radii = radius_median * np.exp(radius_log_sd * rng.standard_normal(edges.shape[0]))
    conductance = np.pi * radii**4 / (8.0 * viscosity * segment_length)
    volume = np.pi * radii**2 * segment_length

    inlets = index[0].ravel()
    outlets = index[-1].ravel()
    return FlowNetwork(
        edges=edges,
        conductance=conductance,
        volume=volume,
        inlets=inlets,
        outlets=outlets,
        n_nodes=int(index.size),
    )
