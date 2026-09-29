"""Equal-load-sharing (ELS) fiber bundle model -- the validation fixture.

This module exists to *validate the rest of the library*, not to model anything
biological. The ELS fiber bundle is the standard reference system for avalanche
statistics under load redistribution because its avalanche-size exponent is
known analytically:

    P(Delta) ~ Delta**(-5/2)        (Hemmer & Hansen, full loading history)

and, restricted to a window just below global failure, crosses over to

    P(Delta) ~ Delta**(-3/2)        (near-critical window)

Both results are independent of the threshold distribution. So if
``cascadence.scaling`` cannot recover 5/2 here, any exponent it reports on a
vascular network is an artefact and must not be believed. Everything
downstream of this file rests on that check.

The crossover matters for interpreting results on other systems too: 5/2 and
3/2 are *the same mechanism sampled over different windows*, not two different
mechanisms. Reporting a fitted exponent without stating the sampling window is
therefore meaningless.

Model
-----
N fibers in parallel share a total load F equally. Fiber i fails once its own
share exceeds a threshold x_i, drawn i.i.d. from some distribution. Under
quasi-static (force-controlled) loading, sort the thresholds ascending; the
external force required to break the (k+1)-th weakest fiber is

    F_k = (N - k) * x_k,      k = 0, ..., N-1

because k fibers are already broken and the remaining N-k share F equally.
As F is ramped monotonically upward, one fiber breaks each time F reaches a new
*record maximum* of the sequence F_k; every subsequent fiber with F_k below
that record breaks immediately, without any further increase in external load.
An avalanche is therefore exactly the gap between successive record maxima of
F_k, which is what :func:`avalanche_sizes` computes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["avalanche_sizes", "critical_force", "uniform_bundle"]


def uniform_bundle(n_fibers: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_fibers`` failure thresholds uniformly on (0, 1).

    The uniform distribution is the conventional choice for this benchmark. The
    asymptotic exponents do not depend on it, which is part of why the model is
    a good validation target.
    """
    return rng.random(n_fibers)


def critical_force(thresholds: np.ndarray) -> float:
    """Peak of the load curve, i.e. the force at which the bundle fails globally.

    Equals ``max_k (N - k) * x_k`` over the ascending-sorted thresholds.
    """
    x = np.sort(np.asarray(thresholds, dtype=float))
    n = x.size
    return float(np.max((n - np.arange(n)) * x))


def avalanche_sizes(
    thresholds: np.ndarray,
    *,
    include_final: bool = False,
    force_window: float | None = None,
) -> np.ndarray:
    """Avalanche sizes over a quasi-static loading history.

    Parameters
    ----------
    thresholds
        Fiber failure thresholds, any order. Must be strictly positive.
    include_final
        Whether to keep the catastrophic final avalanche, i.e. everything that
        breaks once the external force passes the global maximum of the load
        curve. It is *not* an avalanche in the scaling sense -- it is O(N) and
        a single realisation contributes one enormous outlier that visibly
        corrupts the fitted tail. Default ``False``, which is what the 5/2 law
        refers to.
    force_window
        If given, keep only avalanches that *start* at a force of at least
        ``force_window * F_c``, with ``F_c`` the global maximum of the load
        curve. Use this to sample the near-critical window, where the exponent
        crosses over from 5/2 to 3/2. Values around 0.9-0.95 are the usual
        choice; smaller values interpolate back toward 5/2.

    Returns
    -------
    Integer array of avalanche sizes, in loading order.
    """
    x = np.sort(np.asarray(thresholds, dtype=float))
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any(x <= 0.0):
        raise ValueError("fiber thresholds must be strictly positive")

    # Force needed to break the (k+1)-th weakest fiber.
    force = (n - np.arange(n)) * x

    # A new record maximum of `force` means the external load had to increase,
    # which starts a fresh avalanche. Position 0 is always a record.
    running_max = np.maximum.accumulate(force)
    is_record = np.empty(n, dtype=bool)
    is_record[0] = True
    is_record[1:] = force[1:] > running_max[:-1]

    starts = np.flatnonzero(is_record)
    sizes = np.diff(np.append(starts, n))
    start_force = force[starts]

    # The last record is by construction the global maximum of `force`, so the
    # last "avalanche" is the catastrophic final failure.
    if not include_final:
        sizes = sizes[:-1]
        start_force = start_force[:-1]

    if force_window is not None:
        if not 0.0 <= force_window <= 1.0:
            raise ValueError("force_window must lie in [0, 1]")
        keep = start_force >= force_window * force.max()
        sizes = sizes[keep]

    return sizes.astype(np.int64)
