"""Heavy-tail estimation for avalanche-size distributions.

Implements the Clauset-Shalizi-Newman (CSN) recipe for *discrete* data, which
is what avalanche sizes are: counts of blocked segments or broken fibers. Using
the continuous approximation on small integer counts biases the exponent, and
the bias is largest exactly where it matters -- near the lower cutoff, where
most of the data live -- so everything here works on the integer support.

Estimators provided
-------------------
``fit_powerlaw``
    Discrete power-law MLE with KS-selected lower cutoff ``xmin``.
``bootstrap_alpha``
    Non-parametric bootstrap CI for the exponent, optionally re-selecting
    ``xmin`` on each replicate (which is the honest thing to do -- holding it
    fixed understates the uncertainty).
``fit_powerlaw_cutoff`` / ``lr_test_cutoff``
    Power law with exponential cutoff, parametrised by ``rate = 1/xi >= 0`` so
    that the pure power law is the *nested* case ``rate == 0``. That makes the
    likelihood-ratio test a clean chi-squared(1) test rather than a Vuong test.
``fit_lognormal`` / ``vuong_test``
    Discrete lognormal on the same tail, and the non-nested comparison against
    it. A heavy tail that a lognormal fits just as well is not evidence for
    scale invariance, so this test gates any claim about a critical exponent.

A fitted exponent is meaningless without the sampling window it came from --
see :mod:`cascadence.fiberbundle` for why (the same mechanism yields 5/2 over a
full loading history and 3/2 restricted to the near-critical window).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.special import logsumexp, zeta
from scipy.stats import chi2, norm

__all__ = [
    "PowerLawFit",
    "CutoffFit",
    "LognormalFit",
    "bootstrap_alpha",
    "exponent_profile",
    "fit_lognormal",
    "fit_powerlaw",
    "fit_powerlaw_cutoff",
    "lr_test_cutoff",
    "vuong_test",
]

# The discrete cutoff model needs its normalising sum truncated somewhere. Terms
# beyond xmin + 40/rate contribute < 1e-17 of the total, so the cap only bites
# when `rate` is so small that the model is indistinguishable from a pure power
# law over the observed range -- the regime where the LR test correctly reports
# no cutoff either way.
_SUPPORT_CAP = 1_000_000
_ALPHA_BOUNDS = (1.01, 20.0)
_MIN_TAIL = 15


def _as_counts(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data)
    if x.ndim != 1:
        x = x.ravel()
    if x.size == 0:
        raise ValueError("no data")
    if not np.all(np.isfinite(x)):
        raise ValueError("data contain non-finite values")
    rounded = np.rint(x).astype(np.int64)
    if not np.allclose(x, rounded):
        raise ValueError(
            "fit_powerlaw expects integer-valued data (avalanche sizes); "
            "got non-integer values"
        )
    if np.any(rounded < 1):
        raise ValueError("avalanche sizes must be >= 1")
    return rounded


# --------------------------------------------------------------------------
# pure discrete power law
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerLawFit:
    """Result of a discrete power-law fit on the tail ``x >= xmin``."""

    alpha: float
    xmin: int
    ks: float
    n_tail: int
    n_total: int
    loglik: float

    @property
    def tail_fraction(self) -> float:
        return self.n_tail / self.n_total


def _powerlaw_nll(alpha: float, log_sum: float, n: int, xmin: int) -> float:
    if not _ALPHA_BOUNDS[0] <= alpha <= _ALPHA_BOUNDS[1]:
        return np.inf
    return alpha * log_sum + n * np.log(zeta(alpha, xmin))


def _fit_alpha(tail: np.ndarray, xmin: int) -> tuple[float, float]:
    """MLE of the discrete power-law exponent on a given tail.

    Returns ``(alpha, loglik)``.
    """
    n = tail.size
    log_sum = float(np.sum(np.log(tail)))
    # CSN's continuous-approximation estimator, used as a starting bracket.
    guess = 1.0 + n / max(float(np.sum(np.log(tail / (xmin - 0.5)))), 1e-12)
    guess = float(np.clip(guess, _ALPHA_BOUNDS[0] + 1e-6, _ALPHA_BOUNDS[1] - 1e-6))
    result = optimize.minimize_scalar(
        _powerlaw_nll,
        args=(log_sum, n, xmin),
        bounds=_ALPHA_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8},
    )
    alpha = float(result.x) if result.success else guess
    return alpha, -_powerlaw_nll(alpha, log_sum, n, xmin)


def _ks_distance(tail: np.ndarray, alpha: float, xmin: int) -> float:
    """KS distance between the empirical and fitted discrete CDFs."""
    values, counts = np.unique(tail, return_counts=True)
    empirical = np.cumsum(counts) / tail.size
    # P(X <= x) = 1 - zeta(alpha, x + 1) / zeta(alpha, xmin) on integer support.
    fitted = 1.0 - zeta(alpha, values + 1) / zeta(alpha, xmin)
    return float(np.max(np.abs(empirical - fitted)))


def fit_powerlaw(
    data: np.ndarray,
    *,
    xmin: int | None = None,
    xmin_candidates: int = 60,
    min_tail: int = _MIN_TAIL,
) -> PowerLawFit:
    """Fit a discrete power law, selecting ``xmin`` by KS minimisation.

    Parameters
    ----------
    data
        Integer-valued observations, all >= 1.
    xmin
        Fix the lower cutoff instead of selecting it. Use this only when the
        cutoff is known on independent grounds; otherwise let KS choose, and
        propagate the resulting uncertainty with :func:`bootstrap_alpha`.
    xmin_candidates
        Cap on how many distinct candidate cutoffs to scan. The candidates are
        the distinct observed values, thinned geometrically when there are more
        than this many, which keeps the scan O(1) in the data range without
        materially changing the selected cutoff.
    min_tail
        Refuse to fit a tail shorter than this. Exponents from a handful of
        points are not estimates.
    """
    x = _as_counts(data)
    n_total = x.size

    if xmin is not None:
        xmin = int(xmin)
        tail = x[x >= xmin]
        if tail.size < min_tail:
            raise ValueError(
                f"tail above xmin={xmin} has {tail.size} points, need >= {min_tail}"
            )
        alpha, loglik = _fit_alpha(tail, xmin)
        return PowerLawFit(
            alpha=alpha,
            xmin=xmin,
            ks=_ks_distance(tail, alpha, xmin),
            n_tail=tail.size,
            n_total=n_total,
            loglik=loglik,
        )

    candidates = np.unique(x)
    # Only cutoffs that leave a usable tail are worth scanning.
    usable = np.searchsorted(np.sort(x), candidates, side="left")
    candidates = candidates[(n_total - usable) >= min_tail]
    if candidates.size == 0:
        raise ValueError(f"no candidate xmin leaves a tail of >= {min_tail} points")
    if candidates.size > xmin_candidates:
        idx = np.unique(
            np.geomspace(1, candidates.size, xmin_candidates).astype(int) - 1
        )
        candidates = candidates[idx]

    best: PowerLawFit | None = None
    for candidate in candidates:
        tail = x[x >= candidate]
        alpha, loglik = _fit_alpha(tail, int(candidate))
        ks = _ks_distance(tail, alpha, int(candidate))
        if best is None or ks < best.ks:
            best = PowerLawFit(
                alpha=alpha,
                xmin=int(candidate),
                ks=ks,
                n_tail=tail.size,
                n_total=n_total,
                loglik=loglik,
            )
    assert best is not None
    return best


def bootstrap_alpha(
    data: np.ndarray,
    *,
    n_boot: int = 200,
    refit_xmin: bool = True,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
    **fit_kwargs,
) -> tuple[float, float, np.ndarray]:
    """Bootstrap CI for the power-law exponent.

    Resamples the full dataset with replacement. With ``refit_xmin=True`` the
    lower cutoff is re-selected on every replicate, so the interval includes
    the uncertainty in choosing it -- typically the dominant term, and the one
    that makes naive intervals far too narrow.

    Returns ``(lo, hi, replicates)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = _as_counts(data)
    point = fit_powerlaw(x, **fit_kwargs)
    kwargs = dict(fit_kwargs)
    if not refit_xmin:
        kwargs["xmin"] = point.xmin

    replicates = []
    for _ in range(n_boot):
        sample = rng.choice(x, size=x.size, replace=True)
        try:
            replicates.append(fit_powerlaw(sample, **kwargs).alpha)
        except ValueError:
            continue  # degenerate resample, no usable tail
    if len(replicates) < max(20, n_boot // 5):
        raise RuntimeError(
            f"only {len(replicates)}/{n_boot} bootstrap replicates converged; "
            "data are too sparse for a meaningful interval"
        )
    reps = np.asarray(replicates)
    tail_prob = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [tail_prob, 1.0 - tail_prob])
    return float(lo), float(hi), reps


def exponent_profile(
    data: np.ndarray,
    *,
    xmins: np.ndarray | None = None,
    min_tail: int = 200,
) -> dict[str, np.ndarray]:
    """Fitted exponent as a function of the lower cutoff -- report this, not a number.

    A single fitted exponent is not an observable. On the ELS fiber bundle,
    whose asymptotics are known exactly, the *same* realisations yield anywhere
    from alpha ~= 1.54 to alpha ~= 2.72 depending only on the sampling window
    and the fitting range (see ``tests/test_fiberbundle.py``). Quoting one
    number without both of those is therefore uninterpretable, and comparing
    one number against another study's is worse.

    What carries information is the *profile*: whether alpha(xmin) plateaus,
    and at what value. A genuine asymptotic exponent shows up as a flat stretch
    that persists until the statistics run out; a curved profile with no plateau
    means the range is pre-asymptotic and no exponent should be quoted at all.

    Returns a dict with keys ``xmin``, ``alpha``, ``ks``, ``n_tail``.
    """
    x = _as_counts(data)
    if xmins is None:
        # Extend the grid to the largest cutoff that still leaves `min_tail`
        # points. Anything beyond that is not an estimate, and stopping short of
        # it hides the plateau the profile exists to reveal.
        ordered = np.sort(x)
        top = int(ordered[-min_tail]) if ordered.size >= min_tail else int(ordered[-1])
        xmins = np.unique(np.geomspace(1, max(top, 2), 24).astype(np.int64))

    rows = []
    for candidate in xmins:
        candidate = int(candidate)
        tail = x[x >= candidate]
        if tail.size < min_tail:
            continue
        alpha, _ = _fit_alpha(tail, candidate)
        rows.append((candidate, alpha, _ks_distance(tail, alpha, candidate), tail.size))
    if not rows:
        raise ValueError(f"no candidate xmin leaves a tail of >= {min_tail} points")

    arr = np.asarray(rows, dtype=float)
    return {
        "xmin": arr[:, 0].astype(np.int64),
        "alpha": arr[:, 1],
        "ks": arr[:, 2],
        "n_tail": arr[:, 3].astype(np.int64),
    }


# --------------------------------------------------------------------------
# power law with exponential cutoff (nested)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CutoffFit:
    """Discrete power law with exponential cutoff, ``p(x) ~ x**-alpha e**-rate*x``."""

    alpha: float
    rate: float
    xmin: int
    n_tail: int
    loglik: float
    support_capped: bool

    @property
    def xi(self) -> float:
        """Cutoff scale ``1 / rate``; ``inf`` when no cutoff was detected."""
        return float("inf") if self.rate <= 0.0 else 1.0 / self.rate


def _cutoff_log_norm(alpha: float, rate: float, xmin: int) -> tuple[float, bool]:
    """log of sum_{x>=xmin} x**-alpha exp(-rate x), and whether the cap bit."""
    if rate <= 0.0:
        return float(np.log(zeta(alpha, xmin))), False
    span = int(np.ceil(40.0 / rate))
    capped = xmin + span > _SUPPORT_CAP
    upper = min(xmin + span, _SUPPORT_CAP)
    grid = np.arange(xmin, upper + 1, dtype=float)
    return float(logsumexp(-alpha * np.log(grid) - rate * grid)), capped


def fit_powerlaw_cutoff(data: np.ndarray, xmin: int) -> CutoffFit:
    """MLE for a discrete power law with exponential cutoff on ``x >= xmin``.

    Parametrised by ``rate = 1/xi`` with ``rate >= 0`` so that the pure power
    law sits on the boundary at ``rate == 0``. That nesting is what makes
    :func:`lr_test_cutoff` a chi-squared test.
    """
    x = _as_counts(data)
    xmin = int(xmin)
    tail = x[x >= xmin]
    if tail.size < _MIN_TAIL:
        raise ValueError(f"tail above xmin={xmin} has only {tail.size} points")

    log_sum = float(np.sum(np.log(tail)))
    total = float(np.sum(tail))
    n = tail.size
    capped_flag = {"hit": False}

    def nll(theta: np.ndarray) -> float:
        alpha, rate = float(theta[0]), float(theta[1])
        if not _ALPHA_BOUNDS[0] <= alpha <= _ALPHA_BOUNDS[1] or rate < 0.0:
            return np.inf
        log_norm, capped = _cutoff_log_norm(alpha, rate, xmin)
        capped_flag["hit"] = capped
        return alpha * log_sum + rate * total + n * log_norm

    start_alpha, _ = _fit_alpha(tail, xmin)
    # Seed the cutoff scale at the observed tail mean: if a cutoff exists it is
    # of that order, and if it does not the optimiser walks `rate` down to 0.
    start_rate = 1.0 / max(float(np.mean(tail)), 1.0)
    result = optimize.minimize(
        nll,
        x0=np.array([start_alpha, start_rate]),
        bounds=[_ALPHA_BOUNDS, (0.0, 1.0)],
        method="L-BFGS-B",
    )
    alpha, rate = float(result.x[0]), float(result.x[1])
    log_norm, capped = _cutoff_log_norm(alpha, rate, xmin)
    loglik = -(alpha * log_sum + rate * total + n * log_norm)
    return CutoffFit(
        alpha=alpha,
        rate=rate,
        xmin=xmin,
        n_tail=n,
        loglik=loglik,
        support_capped=capped or capped_flag["hit"],
    )


def lr_test_cutoff(data: np.ndarray, xmin: int) -> tuple[float, float, CutoffFit]:
    """Nested LR test of pure power law against power law with cutoff.

    Returns ``(lr_statistic, p_value, cutoff_fit)``. Because the null sits on
    the boundary of the parameter space (``rate == 0``), the asymptotic null
    distribution is the 50:50 mixture of a point mass at 0 and chi-squared(1),
    so the p-value is half the naive chi-squared(1) tail. A small p-value means
    the tail is *not* a pure power law over this range.
    """
    x = _as_counts(data)
    pure = fit_powerlaw(x, xmin=int(xmin))
    cut = fit_powerlaw_cutoff(x, int(xmin))
    lr = 2.0 * (cut.loglik - pure.loglik)
    lr = max(lr, 0.0)
    p = 0.5 * float(chi2.sf(lr, df=1)) if lr > 0.0 else 1.0
    return float(lr), p, cut


# --------------------------------------------------------------------------
# discrete lognormal + non-nested comparison
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LognormalFit:
    mu: float
    sigma: float
    xmin: int
    n_tail: int
    loglik: float


def _lognormal_log_pmf(
    values: np.ndarray, mu: float, sigma: float, xmin: int
) -> np.ndarray:
    grid = np.arange(xmin, min(xmin + _SUPPORT_CAP, 10_000_000), dtype=float)
    log_kernel = -np.log(grid) - 0.5 * ((np.log(grid) - mu) / sigma) ** 2
    log_norm = logsumexp(log_kernel)
    v = np.asarray(values, dtype=float)
    return -np.log(v) - 0.5 * ((np.log(v) - mu) / sigma) ** 2 - log_norm


def fit_lognormal(data: np.ndarray, xmin: int) -> LognormalFit:
    """MLE for a discrete lognormal truncated to ``x >= xmin``."""
    x = _as_counts(data)
    xmin = int(xmin)
    tail = x[x >= xmin]
    if tail.size < _MIN_TAIL:
        raise ValueError(f"tail above xmin={xmin} has only {tail.size} points")
    log_tail = np.log(tail.astype(float))

    def nll(theta: np.ndarray) -> float:
        mu, sigma = float(theta[0]), float(theta[1])
        if sigma <= 1e-3:
            return np.inf
        return -float(np.sum(_lognormal_log_pmf(tail, mu, sigma, xmin)))

    result = optimize.minimize(
        nll,
        x0=np.array([float(np.mean(log_tail)), max(float(np.std(log_tail)), 0.5)]),
        bounds=[(-50.0, 50.0), (1e-3, 50.0)],
        method="L-BFGS-B",
    )
    mu, sigma = float(result.x[0]), float(result.x[1])
    return LognormalFit(
        mu=mu, sigma=sigma, xmin=xmin, n_tail=tail.size, loglik=-nll(result.x)
    )


def vuong_test(data: np.ndarray, xmin: int) -> tuple[float, float]:
    """Vuong test: power law vs lognormal on the same tail.

    Returns ``(statistic, p_value)``. A positive statistic favours the power
    law, negative favours the lognormal; a p-value above ~0.1 means the data
    cannot tell them apart, which is the usual outcome for short tails and is
    the honest thing to report when it happens.
    """
    x = _as_counts(data)
    xmin = int(xmin)
    tail = x[x >= xmin]
    pl = fit_powerlaw(x, xmin=xmin)
    ln = fit_lognormal(x, xmin)

    ll_pl = -pl.alpha * np.log(tail.astype(float)) - np.log(zeta(pl.alpha, xmin))
    ll_ln = _lognormal_log_pmf(tail, ln.mu, ln.sigma, xmin)
    diff = ll_pl - ll_ln
    n = diff.size
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return 0.0, 1.0
    statistic = float(np.sum(diff)) / (np.sqrt(n) * sd)
    p = 2.0 * float(norm.sf(abs(statistic)))
    return statistic, p
