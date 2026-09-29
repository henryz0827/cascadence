"""The prediction: event frequency and event size are set by one parameter.

What survives after criticality is ruled out
--------------------------------------------
``examples/finite_size.py`` establishes that this model has no critical point,
so there is no scale-free avalanche regime and no critical exponent to read.
The original question does not need one. It was: predict the *joint*
distribution of when occlusive events happen and how large they are.

A single control parameter -- the median delay time, which physiologically
tracks cell composition -- sets both halves:

* the rate at which the quiescent network throws off a first occlusion
  (``cascade.seeding_rate``), and
* the size scale of the avalanche that follows.

Eliminating that parameter between them leaves a relation between frequency
and size with nothing free in it. That is the same coupling a critical-scaling
argument would have provided, obtained without assuming criticality -- which
matters, because here criticality is not available.

The observable model
--------------------
Not every occlusion becomes a clinically visible event; only avalanches above
some size do. So the observables are

    rate      =  lambda * P(S > S_c)
    mean size =  E[S | S > S_c]

with the detection threshold ``S_c`` the only free parameter. Panel C traces
(rate, mean size) as the control parameter varies, at three thresholds, so the
prediction can be read off and its threshold-sensitivity seen at the same time.

Both axes are in model units: ``lambda`` is a rate up to the cell volume, and
size is in occluded segments. Only the *shape* of the curve is a prediction --
comparing it to data means matching that shape, not the absolute scales.

The exponent is not yet a prediction -- measured, not assumed
------------------------------------------------------------
Panel C's slope ``b`` in ``xi ~ lambda**b`` is **not stable on this synthetic
lattice**, and must not be quoted as a prediction. Measured across sizes and
disorder realisations:

===========  ==================  ================
lattice      sweeping shift      sweeping matched R
===========  ==================  ================
16x16        0.42, 0.39          0.79, 0.60
24x24        0.59, 0.62          0.43, 0.46
32x32        0.86, 0.88          --
===========  ==================  ================

The two protocols drift with system size in *opposite* directions, and two
disorder realisations of the same 16x16 lattice differ by 0.19. Nothing that
moves like that is a property of the mechanism.

This is a statement about the placeholder geometry, not necessarily about the
coupling. A periodic lattice has no intrinsic size -- its path lengths are set
by the box -- so asking what ``b`` tends to as ``L`` grows is asking about an
artefact. A real capillary bed has one architecture and one size, and for a
clinical prediction no limit is needed. Which is to say: settling ``b`` requires
measured microvascular geometry, and until then this script reports a coupling
whose existence is clear and whose exponent is not.

Run ``python examples/coupling.py --quick`` for a coarse version.
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

from cascadence.cascade import (
    avalanche_ensemble,
    expected_branching_ratio,
    seeding_rate,
)
from cascadence.kernel import RaceKernel
from cascadence.network import lattice_network
from cascadence.scaling import fit_powerlaw_cutoff

# Ordinal ramp over the detection thresholds; validated in --ordinal mode.
RAMP = ["#86b6ef", "#2a78d6", "#184f95"]
MARKERS = ["o", "s", "^"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

THRESHOLDS = (2, 5, 10)
_NUMERIC_PREFIXES = ("shift", "ratio", "lambda", "xi", "censored", "rate_", "size_")


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


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return [
            {
                key: float(value) if key.startswith(_NUMERIC_PREFIXES) else value
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="redraw from an existing coupling.csv, no simulation",
    )
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    shape = (16, 16) if args.quick else (24, 24)
    n_aval = 600 if args.quick else 4000
    shifts = np.arange(4.0, 9.01, 0.5) if args.quick else np.arange(4.0, 9.01, 0.25)

    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    started = time.time()

    if args.replot:
        rows = load_rows(args.out / "coupling.csv")
        print(f"replotting {len(rows)} rows from {args.out}/coupling.csv")
    else:
        net = lattice_network(shape, np.random.default_rng(42))
        baseline = net.solve()
        transit = baseline.transit_time[baseline.perfused]
        kernel = RaceKernel.from_polymerization(
            tau_ref=float(np.median(transit)), exponent_n=20.0, conc_log_sd=0.1
        )

        rows = []
        for shift in shifts:
            shifted = kernel.shifted(float(shift))
            rate = seeding_rate(net, shifted, baseline=baseline)
            ratio = expected_branching_ratio(
                net, shifted, n_seeds=300, rng=np.random.default_rng(3),
                baseline=baseline,
            )
            ensemble = avalanche_ensemble(
                net, shifted, np.random.default_rng(5), n_realizations=n_aval
            )
            censored = float(np.mean([av.censored for av in ensemble]))
            sizes = np.array(
                [av.deperfused for av in ensemble if not av.censored], dtype=np.int64
            )
            row = {
                "shift": float(shift),
                "ratio_R": ratio,
                "lambda": rate,
                "censored": censored,
                "xi": float("nan"),
            }
            with contextlib.suppress(ValueError, RuntimeError):
                row["xi"] = fit_powerlaw_cutoff(sizes, 1).xi
            for threshold in THRESHOLDS:
                above = sizes[sizes > threshold]
                detected = above.size / sizes.size if sizes.size else 0.0
                row[f"rate_{threshold}"] = rate * detected
                row[f"size_{threshold}"] = (
                    float(above.mean()) if above.size else float("nan")
                )
            rows.append(row)
            print(
                f"shift={shift:.2f} R={ratio:.3f} lambda={rate:.4g} xi={row['xi']:.1f} "
                f"censored={censored:.3f}  "
                + "  ".join(
                    f"S_c={t}: rate={row[f'rate_{t}']:.3g} size={row[f'size_{t}']:.1f}"
                    for t in THRESHOLDS
                ),
                flush=True,
            )

        with (args.out / "coupling.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    shift_axis = np.array([r["shift"] for r in rows])
    lam = np.array([r["lambda"] for r in rows])
    xi = np.array([r["xi"] for r in rows])
    intrinsic = np.isfinite(lam) & np.isfinite(xi) & (lam > 0) & (xi > 0)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    primary = RAMP[1]

    # A/B: the two halves, each against the control parameter.
    ax_a.plot(shift_axis[intrinsic], lam[intrinsic], color=primary, marker="o", ms=4)
    ax_a.set_yscale("log")
    ax_a.invert_xaxis()
    ax_a.set_xlabel(r"control shift  $\Delta\log\tau$   (susceptibility $\rightarrow$)")
    ax_a.set_ylabel(r"seeding rate  $\lambda$  (model units)")
    ax_a.set_title("A  one parameter sets the event rate", loc="left")
    ax_a.grid(True, alpha=0.5, which="major")

    ax_b.plot(shift_axis[intrinsic], xi[intrinsic], color=primary, marker="s", ms=4)
    ax_b.set_yscale("log")
    ax_b.invert_xaxis()
    ax_b.set_xlabel(r"control shift  $\Delta\log\tau$   (susceptibility $\rightarrow$)")
    ax_b.set_ylabel(r"size scale  $\xi$  (segments)")
    ax_b.set_title("B  ...and the size scale, together", loc="left")
    ax_b.grid(True, alpha=0.5, which="major")

    # C: the prediction proper -- the control parameter eliminated between them.
    slope_intrinsic = float("nan")
    if intrinsic.sum() >= 2:
        slope_intrinsic, intercept = np.polyfit(
            np.log(lam[intrinsic]), np.log(xi[intrinsic]), 1
        )
        guide = np.exp(intercept) * lam[intrinsic] ** slope_intrinsic
        ax_c.plot(lam[intrinsic], guide, color=MUTED, ls="--", lw=1.0, zorder=1)
        ax_c.annotate(
            rf"slope ${slope_intrinsic:.2f}$",
            xy=(lam[intrinsic][-1], guide[-1]), xytext=(6, -10),
            textcoords="offset points", color=MUTED, fontsize=8,
        )
    ax_c.plot(lam[intrinsic], xi[intrinsic], color=primary, marker="o", ms=4, zorder=3)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"seeding rate  $\lambda$  (model units)")
    ax_c.set_ylabel(r"size scale  $\xi$  (segments)")
    ax_c.set_title(r"C  the prediction: $\xi$ against $\lambda$", loc="left")
    ax_c.grid(True, alpha=0.5, which="major")

    # D: what a study would actually see, once detection is imposed.
    for colour, marker, threshold in zip(RAMP, MARKERS, THRESHOLDS, strict=True):
        rate = np.array([r[f"rate_{threshold}"] for r in rows])
        size = np.array([r[f"size_{threshold}"] for r in rows])
        good = np.isfinite(rate) & np.isfinite(size) & (rate > 0)
        ax_d.plot(rate[good], size[good], color=colour, marker=marker, ms=4,
                  label=f"$S_c={threshold}$")
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("observed event rate  (model units)")
    ax_d.set_ylabel("mean size above threshold")
    ax_d.set_title("D  with detection imposed", loc="left")
    ax_d.grid(True, alpha=0.5, which="major")
    ax_d.legend(loc="upper left", title="detection threshold")

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"coupling.{suffix}", dpi=200, bbox_inches="tight")

    elapsed = time.time() - started
    written = "png,pdf" if args.replot else "png,pdf and .csv"
    print(f"\nwrote {args.out}/coupling.{{{written}}} in {elapsed:.0f}s")
    print(f"intrinsic coupling on THIS lattice: xi ~ lambda ** {slope_intrinsic:.3f}")
    print("  NOT a prediction: the exponent drifts with system size in opposite")
    print("  directions under the two sweep protocols, and varies by ~0.2 between")
    print("  disorder realisations. See this file's docstring. Settling it needs")
    print("  measured microvascular geometry, not a larger lattice.")
    for threshold in THRESHOLDS:
        rate = np.array([r[f"rate_{threshold}"] for r in rows])
        size = np.array([r[f"size_{threshold}"] for r in rows])
        good = np.isfinite(rate) & np.isfinite(size) & (rate > 0)
        if good.sum() >= 2:
            slope = np.polyfit(np.log(rate[good]), np.log(size[good]), 1)[0]
            print(f"  with S_c={threshold}: apparent slope {slope:+.3f}")
    print("  the apparent slopes are shallower and threshold-dependent, because a")
    print("  mean above S_c is pinned near S_c wherever xi is small: panel D is what")
    print("  a study sees, panel C is what the mechanism says.")


if __name__ == "__main__":
    main()
