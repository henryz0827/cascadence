"""Finite-size scaling: does this network have a critical point at all?

The prerequisite question, and the one left open by ``examples/validation.py``.
There, avalanche-size distributions on a single lattice showed no plateau in
``alpha(x_min)``, and the largest avalanches reached 90% of the perfused
network -- so the distribution was truncated by the box and nothing could be
concluded about the dynamics.

The test
--------
At a genuine critical point the correlation length diverges, so the mean
avalanche size and the fitted cutoff must **grow** as a power of the linear
size ``L``. If they saturate or shrink with ``L``, the large avalanches seen on
small lattices were a finite-size artefact -- the small box was simply easy to
span -- and there is no diverging length scale to support scale-free
statistics. That decides whether a critical-cascade framing is available for
this model at all, which is why it comes before any exponent.

Matching across sizes
---------------------
System sizes are compared at **matched branching ratio**, not at matched
control shift. The same shift produces different ratios on different lattices,
so a shift-matched comparison comes out of different physical states and any
trend in ``L`` it shows is an artefact of the mismatch. Each measurement point
therefore bisects for the shift that puts ``R`` at its target.

Three targets are used: ``R = 1``, where a branching process would be critical,
and two supercritical values, which is where the small lattices produced their
system-spanning avalanches.

Each size is run over several independent disorder realisations, because the
critical shift of a single lattice fluctuates with its particular draw of radii
and one realisation cannot separate that scatter from a trend in ``L``.

Run ``python examples/finite_size.py --quick`` for a coarse version.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cascadence.cascade import avalanche_ensemble, shift_for_ratio
from cascadence.kernel import RaceKernel
from cascadence.network import lattice_network
from cascadence.scaling import fit_powerlaw_cutoff

# Ordinal ramp (single hue, monotone lightness) for the ordered branching-ratio
# targets; validated against the data-viz palette checker in --ordinal mode.
# Light to dark follows small to large R, so the ramp reads with the magnitude.
RAMP = ["#86b6ef", "#2a78d6", "#184f95"]
MARKERS = ["o", "s", "^"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
REFERENCE = "#8a8983"

TARGETS = (1.0, 1.5, 2.0)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "font.family": "sans-serif",
        }
    )


def build(size: int, disorder_seed: int):
    net = lattice_network((size, size), np.random.default_rng(disorder_seed))
    baseline = net.solve()
    transit = baseline.transit_time[baseline.perfused]
    kernel = RaceKernel.from_polymerization(
        tau_ref=float(np.median(transit)), exponent_n=20.0, conc_log_sd=0.1
    )
    return net, baseline, kernel


def measure(net, baseline, kernel, shift, *, n_realizations, seed):
    ensemble = avalanche_ensemble(
        net,
        kernel.shifted(shift),
        np.random.default_rng(seed),
        n_realizations=n_realizations,
    )
    censored = float(np.mean([av.censored for av in ensemble]))
    sizes = np.array(
        [av.deperfused for av in ensemble if not av.censored], dtype=np.int64
    )
    row = {
        "censored": censored,
        "n": int(sizes.size),
        "mean": float(sizes.mean()) if sizes.size else float("nan"),
        "max": int(sizes.max()) if sizes.size else 0,
        "max_over_perfused": (
            float(sizes.max() / baseline.n_perfused) if sizes.size else 0.0
        ),
        "xi": float("nan"),
    }
    with contextlib.suppress(ValueError, RuntimeError):
        row["xi"] = fit_powerlaw_cutoff(sizes, 1).xi
    return row


_NUMERIC = {"size", "rep", "perfused", "target_R", "shift", "censored", "n",
            "mean", "max", "max_over_perfused", "xi"}


def load_rows(path: Path) -> list[dict]:
    """Read a previous run's CSV back, so the figure can be redrawn cheaply.

    The simulation costs tens of minutes and the figure gets restyled far more
    often than the numbers change, so plotting is kept separate from computing
    and ``--replot`` re-renders from this file alone.
    """
    with path.open(newline="") as handle:
        return [
            {key: float(value) if key in _NUMERIC else value
             for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="redraw the figure from an existing finite_size.csv, no simulation",
    )
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    if args.quick:
        sizes = [12, 16, 20, 24]
        n_disorder, n_aval, n_seeds = 2, 600, 200
    else:
        sizes = [12, 16, 20, 24, 28, 32]
        n_disorder, n_aval, n_seeds = 3, 2500, 400

    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    started = time.time()
    rows = []

    if args.replot:
        rows = load_rows(args.out / "finite_size.csv")
        sizes = sorted({int(row["size"]) for row in rows})
        print(f"replotting {len(rows)} rows from {args.out}/finite_size.csv")

    for size in [] if args.replot else sizes:
        for rep in range(n_disorder):
            net, baseline, kernel = build(size, disorder_seed=1000 * rep + size)
            for target in TARGETS:
                try:
                    shift = shift_for_ratio(
                        net,
                        kernel,
                        target,
                        n_seeds=n_seeds,
                        rng=np.random.default_rng(7 + rep),
                    )
                except ValueError as exc:
                    print(f"L={size} rep={rep} R={target}: {exc}", flush=True)
                    continue
                stats = measure(
                    net,
                    baseline,
                    kernel,
                    shift,
                    n_realizations=n_aval,
                    seed=5000 + 17 * rep + size,
                )
                rows.append(
                    {
                        "size": size,
                        "rep": rep,
                        "perfused": baseline.n_perfused,
                        "target_R": target,
                        "shift": shift,
                        **stats,
                    }
                )
                print(
                    f"L={size:>3} rep={rep} R={target:.1f} shift={shift:.3f}  "
                    f"<S>={stats['mean']:.2f} max={stats['max']:>5} "
                    f"max/perf={stats['max_over_perfused']:.3f} "
                    f"xi={stats['xi']:.1f} censored={stats['censored']:.3f}",
                    flush=True,
                )

    if not args.replot:
        with (args.out / "finite_size.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    arr_size = np.array([r["size"] for r in rows])
    arr_target = np.array([r["target_R"] for r in rows])

    def aggregate(target, key):
        mean, err = [], []
        for size in sizes:
            sel = (arr_size == size) & (arr_target == target)
            vals = np.array([rows[i][key] for i in np.flatnonzero(sel)], dtype=float)
            vals = vals[np.isfinite(vals)]
            mean.append(vals.mean() if vals.size else np.nan)
            err.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
        return np.array(mean), np.array(err)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.5, 4.2))
    lengths = np.array(sizes, dtype=float)
    slopes = {}

    for colour, marker, target in zip(RAMP, MARKERS, TARGETS, strict=True):
        label = f"$R={target:g}$"
        mean, err = aggregate(target, "mean")
        ax_a.errorbar(lengths, mean, yerr=err, color=colour, marker=marker, ms=5,
                      capsize=3, label=label)
        finite = np.isfinite(mean)
        if finite.sum() >= 2:
            slopes[target] = float(
                np.polyfit(np.log(lengths[finite]), np.log(mean[finite]), 1)[0]
            )

        xi_mean, xi_err = aggregate(target, "xi")
        ax_b.errorbar(lengths, xi_mean, yerr=xi_err, color=colour, marker=marker, ms=5,
                      capsize=3, label=label)

        frac_mean, frac_err = aggregate(target, "max_over_perfused")
        ax_c.errorbar(lengths, frac_mean, yerr=frac_err, color=colour, marker=marker,
                      ms=5, capsize=3, label=label)

    # A reference slope of 1 shows what "grows with L" would look like.
    guide = lengths / lengths[0] * 3.0
    ax_a.plot(lengths, guide, color=REFERENCE, ls="--", lw=1.0, zorder=1)
    ax_a.annotate(r"slope 1", xy=(lengths[-1], guide[-1]), xytext=(-42, 2),
                  textcoords="offset points", color=MUTED, fontsize=8)

    def _size_axis(ax) -> None:
        # Label the lattices actually simulated: default log ticks fall between
        # them and leave the reader unable to place a point on a size.
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xticks([], minor=True)
        ax.set_xlabel("linear size  $L$")
        ax.grid(True, alpha=0.5, which="major")

    _size_axis(ax_a)
    # Headroom so the legend clears the R=2 error bars rather than sitting on them.
    ax_a.set_ylim(top=ax_a.get_ylim()[1] * 3.0)
    ax_a.set_ylabel(r"mean avalanche size  $\langle S \rangle$")
    ax_a.set_title(r"A  $\langle S\rangle$ is flat in $L$", loc="left")
    # Upper right: every series falls to the right, so that corner stays clear.
    ax_a.legend(loc="upper right", title="matched branching ratio")

    _size_axis(ax_b)
    ax_b.set_ylabel(r"fitted cutoff  $\xi$")
    ax_b.set_title(r"B  the cutoff does not diverge", loc="left")
    ax_b.legend(loc="center right")

    _size_axis(ax_c)
    ax_c.set_ylabel("largest avalanche / perfused edges")
    ax_c.set_title("C  spanning is a small-lattice artefact", loc="left")
    ax_c.legend(loc="lower left")

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"finite_size.{suffix}", dpi=200, bbox_inches="tight")

    elapsed = time.time() - started
    written = "png,pdf" if args.replot else "png,pdf and .csv"
    print(f"\nwrote {args.out}/finite_size.{{{written}}} in {elapsed:.0f}s")
    print("log-log slope of <S> against L, at matched branching ratio:")
    for target, slope in slopes.items():
        print(f"   R={target:g}:  {slope:+.3f}")
    print("  a clearly positive slope would indicate a diverging correlation")
    print("  length; zero or negative means no critical point here, and so no")
    print("  scale-free regime to read an exponent from.")


if __name__ == "__main__":
    main()
