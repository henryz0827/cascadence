"""Why tuning the branching ratio to 1 does not make this system critical.

``examples/finite_size.py`` establishes that there is no critical point: at
matched branching ratio the mean avalanche size does not grow with the system
size. This script explains it.

The branching ratio is normally one number, and ``R = 1`` is normally the
critical condition. Both assume the cascade is a tree, so that every generation
branches at the same rate. On a flow network with global pressure
redistribution it does not. Resolving the ratio generation by generation shows
the first generation branching at 1 by construction, the **second at roughly
half that**, and the rest plateauing well below 1.

The mechanism is depletion: a generation's blocks are clustered around the ones
that produced them, so the next generation re-attacks a neighbourhood already
stripped of its susceptible segments. Whether that depletion weakens as the
lattice grows -- in which case it would be a finite-size artefact -- is the
question this script answers, by measuring the profile across sizes.

Three earlier observations follow from this one measurement:

* Avalanches stay small at ``R_1 = 1``, because that is not the critical
  condition. Driving later generations to 1 would need ``R_1`` near 2, and by
  then avalanches span the system outright.
* ``<S> = 1/(1-R)`` is invalid, since it assumes a single rate. Hence the two
  estimators in ``cascade.branching_ratio`` disagreeing near criticality.
* The system passes from self-limiting to system-spanning with no scale-free
  regime in between.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cascadence.cascade import avalanche_ensemble, critical_shift, generation_profile
from cascadence.kernel import RaceKernel
from cascadence.network import lattice_network

# Ordinal ramp over system size; validated in --ordinal mode.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
REFERENCE = "#8a8983"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    sizes = [12, 16, 24, 32]
    n_aval = 800 if args.quick else 4000
    n_seeds = 200 if args.quick else 400

    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    started = time.time()

    profiles, rows = {}, []
    for size in sizes:
        net = lattice_network((size, size), np.random.default_rng(42))
        baseline = net.solve()
        transit = baseline.transit_time[baseline.perfused]
        kernel = RaceKernel.from_polymerization(
            tau_ref=float(np.median(transit)), exponent_n=20.0, conc_log_sd=0.1
        )
        shift = critical_shift(
            net, kernel, n_seeds=n_seeds, rng=np.random.default_rng(7)
        )
        ensemble = avalanche_ensemble(
            net, kernel.shifted(shift), np.random.default_rng(5),
            n_realizations=n_aval,
        )
        profile = generation_profile(ensemble)
        profiles[size] = profile
        for gen, ratio, total in zip(
            profile["generation"], profile["ratio"], profile["total"], strict=True
        ):
            rows.append(
                {
                    "size": size,
                    "shift_c": shift,
                    "generation": int(gen),
                    "ratio": float(ratio),
                    "n_in_generation": int(total),
                }
            )
        print(
            f"L={size:>3} s_c={shift:.3f}  "
            f"R_k = {np.round(profile['ratio'][:6], 3).tolist()}",
            flush=True,
        )

    with (args.out / "generations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax_a.axhline(1.0, color=REFERENCE, lw=1.0, ls="--", zorder=1)
    ax_a.annotate("$R_k=1$", xy=(6, 1.0), xytext=(0, 5), textcoords="offset points",
                  color=MUTED, fontsize=8)
    for colour, size in zip(RAMP, sizes, strict=True):
        profile = profiles[size]
        keep = profile["generation"] <= 8
        ax_a.plot(profile["generation"][keep], profile["ratio"][keep], color=colour,
                  marker="o", ms=4, label=f"$L={size}$", zorder=3)
    ax_a.set_xlabel("generation  $k$")
    ax_a.set_ylabel("branching ratio  $R_k$")
    ax_a.set_title("A  the ratio halves after generation 1", loc="left")
    ax_a.set_ylim(0.0, 1.35)
    ax_a.grid(True, alpha=0.5)
    ax_a.legend(loc="upper right", ncol=2)

    # B: the drop itself, against system size -- is it a finite-size artefact?
    lengths = np.array(sizes, dtype=float)
    drop = np.array([profiles[s]["ratio"][1] / profiles[s]["ratio"][0] for s in sizes])
    ax_b.axhline(1.0, color=REFERENCE, lw=1.0, ls="--", zorder=1)
    ax_b.annotate("no drop", xy=(lengths[0], 1.0), xytext=(2, 5),
                  textcoords="offset points", color=MUTED, fontsize=8)
    ax_b.plot(lengths, drop, color=RAMP[2], marker="o", ms=6, zorder=3)
    ax_b.set_xscale("log")
    # Label the sizes actually simulated; the default log ticks land between
    # them and leave the reader unable to tell which point is which lattice.
    ax_b.set_xticks(lengths)
    ax_b.set_xticklabels([str(s) for s in sizes])
    ax_b.set_xticks([], minor=True)
    ax_b.set_xlabel("linear size  $L$")
    ax_b.set_ylabel(r"$R_2 / R_1$")
    ax_b.set_title("B  the drop does not soften with size", loc="left")
    ax_b.set_ylim(0.0, 1.15)
    ax_b.grid(True, alpha=0.5)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"generations.{suffix}", dpi=200, bbox_inches="tight")

    elapsed = time.time() - started
    print(f"\nwrote {args.out}/generations.{{png,pdf}} and .csv in {elapsed:.0f}s")
    print(f"R_2/R_1 across L = {np.round(drop, 3).tolist()}")
    print("  a drop that persists as L grows is structural, not a finite-size artefact:")
    print("  no single branching ratio describes this cascade, so R = 1 is not")
    print("  a critical condition and <S> = 1/(1-R) does not apply.")


if __name__ == "__main__":
    main()
