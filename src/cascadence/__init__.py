"""cascadence -- critical cascades on flow networks with load redistribution.

A library for the coupled *timing* and *size* statistics of avalanches on
networks where removing an edge redistributes flow globally. The biological
application that motivated it (vaso-occlusion) is one instance and lives in
``examples/``; nothing in the core modules is specific to it.

Layout
------
:mod:`cascadence.network`
    Conductance networks under a pressure drop; edge removal and transit times.
:mod:`cascadence.kernel`
    Blocking kernels -- the probability a segment occludes at a given transit
    time. The only place microscopic kinetics enters.
:mod:`cascadence.cascade`
    Avalanche dynamics and measurement of the branching ratio.
:mod:`cascadence.scaling`
    Heavy-tail estimation: discrete power-law MLE, cutoff tests, exponent
    profiles.
:mod:`cascadence.fiberbundle`
    Equal-load-sharing fiber bundle, whose exponents are known analytically.
    The validation fixture for everything in :mod:`cascadence.scaling`.

Read ``cascadence.fiberbundle`` before trusting any exponent this library
reports: the same fiber bundle realisations yield fitted exponents anywhere from
1.54 to 2.72 depending only on the sampling window and the fitting range, so an
exponent quoted without both is not a measurement.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from . import cascade, fiberbundle, kernel, network, scaling

__all__ = ["cascade", "fiberbundle", "kernel", "network", "scaling", "__version__"]
