"""Validation run: what this library has established, and what it has not.

Produces ``figures/validation.{png,pdf}`` plus the CSVs behind every panel, so
the numbers can be read without the picture.

The four panels, and the claim each one supports:

A  Exponent profiles, fiber bundle against the synthetic vascular lattice.
   The fiber bundle's profile *plateaus* at 5/2, its analytically known value.
   The lattice's does not plateau at all. A plateau is what licenses quoting an
   exponent; this panel is the difference between a measurement and a number.

B  Fiber bundle crossover. Tightening the sampling window onto the critical
   point drives the small-size slope from ~2 down to 3/2, on the *same*
   realisations. So 5/2 and 3/2 do not distinguish mechanisms -- they
   distinguish sampling windows.

C  Lattice profiles across control settings. None plateau, so no exponent is
   reported for this network at any setting. The largest avalanches press
   against the system size, which is the diagnosis: the distribution is
   truncated by the box, not by the dynamics. Finite-size scaling across
   several lattice sizes is the prerequisite for any exponent claim here, and
   it is not done yet.

D  Branching ratio against the control parameter, by two independent
   estimators. They cross 1 in the same place but disagree by tens of percent
   approaching it: global pressure redistribution correlates offspring across
   generations, so the cascade is not a tree and ``<S> = 1/(1-R)`` fails
   exactly where it would be wanted. A branching ratio must therefore not be
   inferred from a mean avalanche size in this regime.

Run ``python examples/validation.py --quick`` for a coarse version in well
under a minute.
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

from cascadence.cascade import avalanche_ensemble, sweep_control
from cascadence.fiberbundle import avalanche_sizes, uniform_bundle
from cascadence.kernel import RaceKernel
from cascadence.network import lattice_network
from cascadence.scaling import exponent_profile, fit_powerlaw, lr_test_cutoff

# Validated against the data-viz palette checker (see README): categorical
# slots 1 and 2 for the two-series panels, and a single-hue blue ordinal ramp
# for the control-parameter sequence in panel C.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_RAMP = ["#86b6ef", "#2a78d6", "#184f95"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
REFERENCE = "#8a8983"

LATTICE_SHIFTS = (4.0, 4.5, 5.0)


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


def _write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def fiber_bundle_profile(rng, *, n_fibers: int, reps: int):
    sizes = np.concatenate(
        [avalanche_sizes(uniform_bundle(n_fibers, rng)) for _ in range(reps)]
    )
    return sizes, exponent_profile(sizes, min_tail=250)


def fiber_bundle_crossover(rng, *, n_fibers: int, reps: int, windows):
    pooled = {w: [] for w in windows}
    for _ in range(reps):
        thresholds = uniform_bundle(n_fibers, rng)
        for window in windows:
            pooled[window].append(avalanche_sizes(thresholds, force_window=window))
    return [fit_powerlaw(np.concatenate(pooled[w]), xmin=1).alpha for w in windows]


def lattice_profiles(net, kernel, *, n_realizations: int, shifts=LATTICE_SHIFTS):
    baseline = net.solve()
    out = {}
    for shift in shifts:
        ensemble = avalanche_ensemble(
            net,
            kernel.shifted(shift),
            np.random.default_rng(101),
            n_realizations=n_realizations,
        )
        censored = float(np.mean([av.censored for av in ensemble]))
        sizes = np.array(
            [av.deperfused for av in ensemble if not av.censored], dtype=np.int64
        )
        record = {
            "sizes": sizes,
            "censored": censored,
            "max_fraction": float(sizes.max() / baseline.n_perfused)
            if sizes.size
            else 0.0,
        }
        try:
            record["profile"] = exponent_profile(sizes, min_tail=150)
        except ValueError:
            record["profile"] = None
        try:
            fit = fit_powerlaw(sizes)
            lr, p, cut = lr_test_cutoff(sizes, fit.xmin)
            record["fit"] = (fit, lr, p, cut)
        except ValueError:
            record["fit"] = None
        out[shift] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="coarse run, seconds not minutes"
    )
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    quick = args.quick
    n_fibers = 60_000 if quick else 300_000
    fb_reps = 4 if quick else 12
    cross_reps = 6 if quick else 20
    n_aval = 400 if quick else 2500
    sweep_n = 80 if quick else 250
    lattice_shape = (16, 12) if quick else (24, 20)

    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    started = time.time()

    print("[1/4] fiber bundle exponent profile")
    fb_sizes, fb_profile = fiber_bundle_profile(
        np.random.default_rng(11), n_fibers=n_fibers, reps=fb_reps
    )
    fb_deep = float(fb_profile["alpha"][-5:].mean())
    print(
        f"      {fb_sizes.size} avalanches; deep-end alpha = {fb_deep:.3f} (expect 2.5)"
    )

    print("[2/4] fiber bundle near-critical crossover")
    windows = [0.70, 0.80, 0.90, 0.95, 0.99, 0.999]
    cross_alpha = fiber_bundle_crossover(
        np.random.default_rng(3), n_fibers=n_fibers, reps=cross_reps, windows=windows
    )
    print(f"      alpha(xmin=1): {[round(a, 3) for a in cross_alpha]} (expect -> 1.5)")

    print("[3/4] lattice avalanche ensembles")
    net = lattice_network(lattice_shape, np.random.default_rng(42))
    baseline = net.solve()
    transit = baseline.transit_time[baseline.perfused]
    kernel = RaceKernel.from_polymerization(
        tau_ref=float(np.median(transit)), exponent_n=20.0, conc_log_sd=0.1
    )
    lattice = lattice_profiles(net, kernel, n_realizations=n_aval)
    for shift, record in lattice.items():
        note = (
            "no fit"
            if record["fit"] is None
            else (
                f"KS alpha={record['fit'][0].alpha:.3f} xmin={record['fit'][0].xmin} "
                f"cutoff p={record['fit'][2]:.1e}"
            )
        )
        print(
            f"      shift={shift}: n={record['sizes'].size} "
            f"censored={record['censored']:.3f} "
            f"max/perfused={record['max_fraction']:.2f} {note}"
        )

    print("[4/4] branching ratio sweep")
    shifts = np.arange(3.0, 11.0, 1.0)
    sweep = sweep_control(
        net, kernel, shifts, np.random.default_rng(7), n_realizations=sweep_n
    )

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A: plateau vs no plateau
    ax_a.axhline(2.5, color=REFERENCE, lw=1.0, ls="--", zorder=1)
    ax_a.annotate(
        "5/2",
        xy=(fb_profile["xmin"][-1], 2.5),
        xytext=(4, 5),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
    )
    ax_a.plot(
        fb_profile["xmin"],
        fb_profile["alpha"],
        color=ORANGE,
        marker="o",
        ms=4,
        label="fiber bundle (known 5/2)",
        zorder=3,
    )
    reference_shift = LATTICE_SHIFTS[0]
    lattice_profile = lattice[reference_shift]["profile"]
    if lattice_profile is not None:
        ax_a.plot(
            lattice_profile["xmin"],
            lattice_profile["alpha"],
            color=BLUE,
            marker="s",
            ms=4,
            label=f"lattice (shift {reference_shift})",
            zorder=2,
        )
    ax_a.set_xscale("log")
    ax_a.set_xlabel("lower cutoff  $x_{\\min}$")
    ax_a.set_ylabel(r"fitted exponent  $\alpha$")
    ax_a.set_title("A  a plateau is what licenses an exponent", loc="left")
    ax_a.grid(True, alpha=0.5)
    # Lower right is the only quadrant both curves leave clear.
    ax_a.legend(loc="lower right")

    # B: crossover
    ax_b.axhline(1.5, color=REFERENCE, lw=1.0, ls="--", zorder=1)
    ax_b.annotate(
        "3/2",
        xy=(windows[0], 1.5),
        xytext=(4, 5),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
    )
    ax_b.plot(windows, cross_alpha, color=BLUE, marker="o", ms=5, zorder=3)
    ax_b.set_xlabel("sampling window  $F/F_c$ lower bound")
    ax_b.set_ylabel(r"$\alpha$ fitted from $x_{\min}=1$")
    ax_b.set_title("B  same data, window sets the exponent", loc="left")
    ax_b.grid(True, alpha=0.5)

    # C: lattice profiles across control settings (ordinal ramp)
    for colour, shift in zip(BLUE_RAMP, LATTICE_SHIFTS, strict=False):
        profile = lattice[shift]["profile"]
        if profile is None:
            continue
        ax_c.plot(
            profile["xmin"],
            profile["alpha"],
            color=colour,
            marker="o",
            ms=4,
            label=f"shift {shift}",
        )
    ax_c.set_xscale("log")
    ax_c.set_xlabel("lower cutoff  $x_{\\min}$")
    ax_c.set_ylabel(r"fitted exponent  $\alpha$")
    ax_c.set_title("C  lattice: no plateau at any setting", loc="left")
    ax_c.grid(True, alpha=0.5)
    ax_c.legend(loc="upper left", title="control shift")

    # D: branching ratio, two estimators
    ax_d.axhline(1.0, color=REFERENCE, lw=1.0, ls="--", zorder=1)
    ax_d.annotate(
        "$R=1$",
        xy=(shifts[-1], 1.0),
        xytext=(-30, 5),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
    )
    usable = sweep["censored_fraction"] == 0.0
    ax_d.plot(
        sweep["shift"][usable],
        sweep["r_offspring"][usable],
        color=BLUE,
        marker="o",
        ms=5,
        label="mean first-generation offspring",
    )
    ax_d.plot(
        sweep["shift"][usable],
        sweep["r_mean_size"][usable],
        color=ORANGE,
        marker="s",
        ms=5,
        label=r"$1 - 1/\langle S\rangle$ (mean-field)",
    )
    ax_d.set_yscale("log")
    ax_d.set_xlabel("control shift  $\\Delta \\log \\tau$")
    ax_d.set_ylabel("branching ratio  $R$")
    ax_d.set_title("D  the two estimators part company near $R=1$", loc="left")
    ax_d.grid(True, alpha=0.5)
    ax_d.legend(loc="lower left")

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"validation.{suffix}", dpi=200, bbox_inches="tight")

    # ---------------- tables ----------------
    _write_csv(
        args.out / "fiber_bundle_profile.csv",
        ["xmin", "alpha", "ks", "n_tail"],
        zip(
            fb_profile["xmin"],
            fb_profile["alpha"],
            fb_profile["ks"],
            fb_profile["n_tail"],
            strict=True,
        ),
    )
    _write_csv(
        args.out / "fiber_bundle_crossover.csv",
        ["force_window", "alpha_at_xmin_1"],
        zip(windows, cross_alpha, strict=True),
    )
    _write_csv(
        args.out / "branching_ratio_sweep.csv",
        [
            "shift",
            "r_offspring",
            "r_mean_size",
            "mean_size",
            "mean_deperfused",
            "censored_fraction",
        ],
        zip(
            sweep["shift"],
            sweep["r_offspring"],
            sweep["r_mean_size"],
            sweep["mean_size"],
            sweep["mean_deperfused"],
            sweep["censored_fraction"],
            strict=True,
        ),
    )
    lattice_rows = []
    for shift, record in lattice.items():
        profile = record["profile"]
        if profile is None:
            continue
        for xmin, alpha, n_tail in zip(
            profile["xmin"], profile["alpha"], profile["n_tail"], strict=True
        ):
            lattice_rows.append(
                [shift, xmin, alpha, n_tail, record["censored"], record["max_fraction"]]
            )
    _write_csv(
        args.out / "lattice_profiles.csv",
        ["shift", "xmin", "alpha", "n_tail", "censored_fraction", "max_over_perfused"],
        lattice_rows,
    )

    elapsed = time.time() - started
    print(f"\nwrote {args.out}/validation.png, .pdf and 4 CSVs in {elapsed:.0f}s")
    print(f"fiber bundle deep-end alpha = {fb_deep:.3f}  (analytic 5/2)")
    print(f"crossover alpha at window 0.999 = {cross_alpha[-1]:.3f}  (analytic 3/2)")
    print("lattice: no plateau at any control setting -> no exponent reported")


if __name__ == "__main__":
    main()
