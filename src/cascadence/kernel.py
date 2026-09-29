"""Blocking kernels: the probability that a segment occludes at transit time T.

The kernel is the one place where microscopic kinetics enters. Everything else
in the library is agnostic to it, so alternative kinetics can be substituted by
implementing :meth:`BlockingKernel.p_block`.

The race construction
---------------------
A segment occludes when polymerisation outruns transport: a cell blocks if its
delay time ``tau`` is shorter than its transit time ``T`` through the segment.
So the per-segment blocking probability is the probability that one random time
beats another,

    p_block(T) = P(tau < T)

with ``tau`` distributed across cells because the cells are heterogeneous.

A correction worth stating explicitly
-------------------------------------
It is tempting to argue that because the delay time depends on concentration
with a very large exponent ``n``,

    tau = tau_ref * (c / c_ref) ** -n,       n of order 10-30

the kernel must be a near-step function of ``T``, and that this steepness is
what drives the system critical. **That argument is wrong**, and the sign of the
effect is the opposite of what it claims. If ``log c`` is Gaussian with standard
deviation ``s``, then

    log tau ~ Normal(log tau_ref - n * log(c/c_ref),  (n * s) ** 2)

so the log-scale spread of ``tau`` across the population is ``n * s``. A large
``n`` therefore makes ``tau`` span *many decades*, and

    p_block(T) = Phi((log T - m) / (n * s))

is correspondingly *flat* in ``log T``. Large ``n`` smears the transition out;
it does not sharpen it.

What large ``n`` does buy is leverage over *composition*: shifting the median
concentration by a factor ``f`` translates ``log tau`` by ``n * log f``, i.e.
many spread-widths for a modest change in ``f``. So the kernel is a slowly
varying function of transit time but an extremely sensitive function of the
control parameter.

The consequence for the model is structural, and it is a feature: criticality is
*not* automatic here. The branching ratio must be tuned to reach 1, so "does the
operating point sit near criticality?" stays an empirical question with a
computable answer, rather than an assumption smuggled in through the kinetics.
This is why :mod:`cascadence.cascade` always *measures* the branching ratio and
never assumes a value for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import norm

__all__ = ["BlockingKernel", "LogisticKernel", "RaceKernel"]


@runtime_checkable
class BlockingKernel(Protocol):
    """Anything that maps transit times to per-segment blocking probabilities."""

    def p_block(self, transit_time: np.ndarray) -> np.ndarray:
        """Probability of occlusion at each transit time, in ``[0, 1]``."""
        ...


@dataclass(frozen=True)
class RaceKernel:
    """``p_block(T) = P(tau < T)`` with a lognormal delay time.

    Parameters
    ----------
    log_tau_median
        Median of ``log tau``, i.e. ``log`` of the median delay time. Shifting
        this is how the control parameter enters: lower it and segments block at
        shorter transit times.
    log_tau_sd
        Standard deviation of ``log tau`` across the cell population. Under the
        concentration scaling above this equals ``n * s``, so it is *large* for
        realistic kinetics -- typically several natural-log units, meaning the
        delay time spans decades.
    """

    log_tau_median: float
    log_tau_sd: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.log_tau_median):
            raise ValueError("log_tau_median must be finite")
        if not (self.log_tau_sd > 0.0 and np.isfinite(self.log_tau_sd)):
            raise ValueError("log_tau_sd must be finite and positive")

    @classmethod
    def from_polymerization(
        cls,
        *,
        tau_ref: float,
        exponent_n: float,
        conc_ratio_median: float = 1.0,
        conc_log_sd: float = 0.1,
    ) -> RaceKernel:
        """Build the kernel from the concentration scaling ``tau ~ c ** -n``.

        Parameters
        ----------
        tau_ref
            Delay time at the reference concentration ``c_ref``.
        exponent_n
            The exponent ``n`` in ``tau = tau_ref * (c / c_ref) ** -n``.
        conc_ratio_median
            Median of ``c / c_ref`` in the population. Values above 1 shorten
            the delay time steeply, which is the control-parameter knob.
        conc_log_sd
            Standard deviation of ``log(c / c_ref)`` across cells.

        Note the resulting ``log_tau_sd = exponent_n * conc_log_sd``: the
        population spread in delay time is amplified by ``n``, not suppressed.
        """
        if tau_ref <= 0.0:
            raise ValueError("tau_ref must be positive")
        if exponent_n <= 0.0:
            raise ValueError("exponent_n must be positive")
        if conc_ratio_median <= 0.0:
            raise ValueError("conc_ratio_median must be positive")
        if conc_log_sd <= 0.0:
            raise ValueError("conc_log_sd must be positive")
        median = float(np.log(tau_ref) - exponent_n * np.log(conc_ratio_median))
        return cls(
            log_tau_median=median,
            log_tau_sd=float(exponent_n * conc_log_sd),
        )

    def p_block(self, transit_time: np.ndarray) -> np.ndarray:
        t = np.asarray(transit_time, dtype=float)
        out = np.zeros(t.shape, dtype=float)
        # A stagnant segment (T = inf) has already lost perfusion; the cascade
        # never evaluates it, but return the limit rather than a nan.
        out[np.isposinf(t)] = 1.0
        finite = np.isfinite(t) & (t > 0.0)
        if np.any(finite):
            z = (np.log(t[finite]) - self.log_tau_median) / self.log_tau_sd
            out[finite] = norm.cdf(z)
        return out

    def shifted(self, delta_log_tau: float) -> RaceKernel:
        """Same kernel with the median delay time translated in log space.

        This is the control-parameter sweep: negative ``delta_log_tau`` makes
        occlusion easier. Because of the ``n`` amplification above, a shift of
        one unit here corresponds to a concentration change of only
        ``exp(1 / n)`` -- a percent or two.
        """
        return RaceKernel(
            log_tau_median=self.log_tau_median + float(delta_log_tau),
            log_tau_sd=self.log_tau_sd,
        )


@dataclass(frozen=True)
class LogisticKernel:
    """Logistic kernel in ``log T``, for probing sensitivity to kernel shape.

    Exists so that conclusions can be checked against a kernel with the same
    location and slope but different tails. If a reported exponent moves when
    the kernel is swapped for this one at matched slope, the result is a
    property of the kernel, not of the network.
    """

    log_t_half: float
    slope: float

    def __post_init__(self) -> None:
        if not (self.slope > 0.0 and np.isfinite(self.slope)):
            raise ValueError("slope must be finite and positive")

    def p_block(self, transit_time: np.ndarray) -> np.ndarray:
        t = np.asarray(transit_time, dtype=float)
        out = np.zeros(t.shape, dtype=float)
        out[np.isposinf(t)] = 1.0
        finite = np.isfinite(t) & (t > 0.0)
        if np.any(finite):
            z = self.slope * (np.log(t[finite]) - self.log_t_half)
            out[finite] = 1.0 / (1.0 + np.exp(-z))
        return out
