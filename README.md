# cascadence

Critical cascades on flow networks with load redistribution: the coupled
*timing* and *size* statistics of avalanches on networks where removing one
edge redistributes flow everywhere.

The motivating application is vaso-occlusion, but nothing in the core modules is
specific to it — the same machinery describes any system where a transport
element fails, its load moves to its neighbours, and that can push them over
too. The biological instance lives in `examples/`.

---

## Status

This repository is at the stage where the **tooling is validated and the first
substantive question is open**. That distinction is the point of the sections
below, and it is kept deliberately sharp: an exponent quoted from unvalidated
tooling is not a measurement, and a plateau that is not there cannot be
asserted into existence.

### Established

**The avalanche extraction is exact.** The equal-load-sharing fiber bundle
implementation reproduces, avalanche for avalanche, an independently written
brute-force simulation that ramps the external force and finds the fixed point
of the broken set at each level (`tests/test_fiberbundle.py`).

**The exponent estimator recovers known analytic values.** On fiber-bundle data
the fitted exponent converges to 5/2 over the full loading history, and to 3/2
when the sampling window is tightened onto the critical point — both are
analytically known, and both come out (panels A and B of the validation figure).

**The flow solver is correct.** Checked against series and parallel conductances,
current conservation on a lattice, and the disconnection case, where cutting the
inlet face gives exactly zero flow everywhere rather than a singular solve
(`tests/test_network.py`).

**The cascade measures response, not baseline.** With a kernel whose blocking
probability does not depend on transit time, no segment ever blocks beyond the
seed — confirming the dynamics samples the response to redistribution rather
than the standing hazard (see *The conditional construction* below).

### Open

**No exponent is claimed for the vascular lattice.** Its `alpha(x_min)` profile
does not plateau at any control setting tested (panel C), and where there is
enough data the nested likelihood-ratio test rejects a pure power law. The
diagnosis is visible in the numbers: at the most permissive setting the largest
avalanche reaches 90% of the perfused network, so the distribution is truncated
by the box, not by the dynamics. **Finite-size scaling across several lattice
sizes is the prerequisite for any exponent claim here, and it has not been
done.** Until it is, the honest output is a profile, not a number.

**The lattice geometry is a placeholder.** A periodic lattice with lognormal
radii is not claimed to resemble any real microvascular bed.

**No clinical data is connected.** See `data/README.md`; the access question is
recorded there as unresolved rather than assumed.

---

## Two findings that constrain how results may be read

### A large kinetic exponent flattens the kernel, it does not sharpen it

If the delay time scales as `tau = tau_ref * (c/c_ref)**-n` with `n` large, and
`log c` is Gaussian with standard deviation `s`, then `log tau` has standard
deviation `n * s`. So the blocking kernel

```
p_block(T) = Phi((log T - m) / (n * s))
```

is *flat* in `log T` for large `n` — the delay time spans decades across the
population. The intuition that a steep concentration dependence makes the
kernel a near-step function in transit time, and that this is what drives the
system critical, has the sign backwards.

What large `n` does buy is leverage over composition: a shift in median
concentration by a factor `f` translates `log tau` by `n * log f`.

The structural consequence is a feature. Criticality is not automatic, so
"does the operating point sit near criticality?" stays an empirical question
with a computable answer. Accordingly `cascadence.cascade` always *measures*
the branching ratio and never assumes one.

### A fitted exponent is not an observable unless the protocol is fixed

On fiber-bundle data, whose asymptotics are known exactly, the *same*
realisations yield fitted exponents anywhere from **1.54 to 2.72**, depending
only on two choices that are easy to leave unstated: the sampling window
relative to the critical point, and the lower cutoff of the fit.

This rules out a tempting research design — reading a single fitted exponent as
a signature that discriminates between mechanisms (say, long-range against
local redistribution). Without the window and the fitting range pinned down, and
without a demonstrated plateau, such a comparison is underdetermined. Hence
`scaling.exponent_profile`, which returns the whole `alpha(x_min)` curve, and
the validation figure, whose first panel is nothing but the contrast between a
profile that plateaus and one that does not.

### The mean-field branching relation fails where it would be wanted

Two independent estimators of the branching ratio — mean first-generation
offspring, and `1 - 1/<S>` inverted from the mean total progeny — agree deep in
the subcritical regime and **diverge by tens of percent approaching `R = 1`**
(panel D, pinned in `tests/test_cascade.py`). Global pressure redistribution
correlates offspring across generations, so the cascade is not a tree.

Practical consequence: a branching ratio must not be inferred from a mean
avalanche size in the near-critical regime, which is the only regime of
interest. Any inference scheme resting on mean-field branching relations needs
rebuilding on measured quantities.

---

## The model

**Layer 1 — the kernel.** A segment occludes when polymerisation outruns
transport: the cell's delay time `tau` beats its transit time `T`. So
`p_block(T) = P(tau < T)`, with `tau` distributed across cells.

**Layer 2 — the network.** Conductance network under a fixed pressure drop.
Poiseuille conductances `g = pi r**4 / (8 mu L)`, segment volumes
`V = pi r**2 L`, and transit time `T = V / |q|` — volume over volumetric flow
rate, which stays dimensionally correct without committing to an area
convention.

**Layer 3 — the cascade.** Block a seed segment, re-solve the pressure field,
let segments whose transit time grew block, repeat until quiescent.

### The conditional construction

Every perfused segment carries a standing blocking probability
`p_base = p_block(T_base)`, and it is not small. Sampling it directly each round
would occlude the network immediately with or without any redistribution: the
"quiescent" state would not be quiescent and no branching ratio would be
measurable. An avalanche is the response to redistribution, so the probability
that applies is conditional on having survived the baseline hazard:

```
p_cond = (p_block(T_now) - p_block(T_base)) / (1 - p_block(T_base))
```

which is zero when the transit time has not changed. The slow baseline
occlusion this factors out is the *driving*; it belongs to a separate, slower
process.

### Two sizes, and censoring

`size` counts segments actively occluded; `deperfused` also counts those
stranded behind them, and is the quantity with a physical referent. Every
avalanche records *why* it stopped. Only `terminated == "quiescent"` avalanches
are self-limiting; `"exhausted"` ones ate the network and their size measures
`n_edges` rather than the dynamics. Mixing the latter into a size distribution
manufactures a cutoff at the system size — `Avalanche.censored` exists to make
that impossible to do by accident.

---

## Install and run

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pytest              # ~2 min, includes the analytic checks
.venv/bin/python examples/validation.py # ~2 min -> figures/validation.{png,pdf} + CSVs
.venv/bin/python examples/validation.py --quick   # ~15 s
```

`figures/` is generated and git-ignored; every panel is reproducible from the
script, and each one also writes the CSV behind it so the numbers can be read
without the picture.

## Layout

```
src/cascadence/
├── network.py      # conductance networks, edge removal, transit times
├── kernel.py       # blocking kernels; the only place kinetics enters
├── cascade.py      # avalanche dynamics, branching-ratio measurement
├── scaling.py      # discrete power-law MLE, cutoff tests, exponent profiles
└── fiberbundle.py  # ELS fiber bundle: the analytic validation fixture
```

Read `fiberbundle.py` before trusting any exponent this library reports.

## License

Code: MIT, see `LICENSE`. Manuscript text and figures under `paper/`, when that
exists: CC-BY-4.0.
