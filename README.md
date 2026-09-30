# cascadence

Critical cascades on flow networks with load redistribution: the coupled
*timing* and *size* statistics of avalanches on networks where removing one
edge redistributes flow everywhere.

The motivating application is vaso-occlusion, but nothing in the core modules is
specific to it — the same machinery describes any system where a transport
element fails, its load moves to its neighbours, and that can push them over
too. The biological instance lives in `examples/`.

---

## The main result so far is negative, and it is about the framing

Cascades on a flow network with global pressure redistribution are **not a
branching process with a single rate**, and the model has **no critical point**.
Both are measured, not argued, and together they rule out the scale-free
avalanche statistics that motivated the project.

**The branching ratio collapses after the first generation.** Resolved
generation by generation at the setting where `R_1 = 1`, the second generation
branches at roughly *half* that rate and the rest plateau near 0.5:

| L | R₁ | R₂ | R₃ | R₄ | R₂/R₁ |
|---:|---:|---:|---:|---:|---:|
| 12 | 1.006 | 0.495 | 0.539 | 0.482 | 0.49 |
| 16 | 1.026 | 0.521 | 0.574 | 0.623 | 0.51 |
| 24 | 0.967 | 0.466 | 0.445 | 0.441 | 0.48 |
| 32 | 1.032 | 0.540 | 0.539 | 0.625 | 0.52 |

`R_2/R_1` is flat across a factor of nearly three in linear size, so this is
structural, not a finite-size artefact. The mechanism is depletion: a
generation's blocks cluster around the ones that produced them, so the next
generation re-attacks a neighbourhood already stripped of its susceptible
segments. The cascade is not a tree.

**There is no critical point.** Comparing system sizes at *matched branching
ratio*, the mean avalanche size does not grow with the linear size — it
declines:

| matched R | log-log slope of ⟨S⟩ vs L |
|---:|---:|
| 1.0 | −0.086 |
| 1.5 | −0.443 |
| 2.0 | −0.995 |

The fitted cutoff behaves the same way, and the largest avalanche as a fraction
of the network falls from ~0.2 at L=12 to ~0.01 at L=32. The system-spanning
avalanches that show up on small lattices are the small box being easy to span,
not a diverging correlation length.

**Three earlier observations follow from the first result**, and are no longer
separate puzzles:

- Avalanches stay small at `R_1 = 1` because that is not the critical
  condition. Driving later generations to 1 would need `R_1` near 2, and by
  then avalanches span the system outright.
- `⟨S⟩ = 1/(1-R)` is not merely imprecise here, it is inapplicable: it assumes
  one rate for every generation. This is why the two estimators in
  `cascade.branching_ratio` disagree approaching criticality.
- The system passes from self-limiting to system-spanning with no scale-free
  regime in between.

### A mechanism that was checked and did not hold

The natural explanation for self-limitation — that under a fixed pressure drop
blocking diverts flow and *speeds up* the survivors — is wrong. Across 1 to 60
blocked segments, **62% of surviving segments slow down**, and the fraction is
independent of how many are blocked. Recorded here so it is not proposed again.

---

## Status

### Established

**The avalanche extraction is exact.** The equal-load-sharing fiber bundle
implementation reproduces, avalanche for avalanche, an independently written
brute-force simulation that ramps the external force and finds the fixed point
of the broken set at each level (`tests/test_fiberbundle.py`).

**The exponent estimator recovers known analytic values.** On fiber-bundle data
the fitted exponent converges to 5/2 over the full loading history, and to 3/2
when the sampling window is tightened onto the critical point — both are
analytically known, and both come out.

**The flow solver is correct.** Checked against series and parallel conductances,
current conservation on a lattice, and the disconnection case, where cutting the
inlet face gives exactly zero flow everywhere rather than a singular solve.

**The cascade measures response, not baseline.** With a kernel whose blocking
probability does not depend on transit time, no segment ever blocks beyond the
seed — confirming the dynamics samples the response to redistribution rather
than the standing hazard (see *The conditional construction* below).

### Open

**Boundary conditions.** Everything above is at a fixed pressure drop, and that
choice may be doing the work. Under fixed *total flow* — the autoregulated
case — blocking would force the same flow through fewer channels, which could
be self-amplifying rather than self-limiting. This is the one direction that
could plausibly recover criticality, and it is the first question a reader will
ask. Not implemented.

**The lattice geometry is a placeholder.** A periodic lattice with lognormal
radii is not claimed to resemble any real microvascular bed.

**Kernel-shape sensitivity.** `kernel.LogisticKernel` exists to re-run the
conclusions under a different kernel at matched slope. Not yet done.

**No clinical data is connected.** See `data/README.md`; the access question is
recorded there as unresolved rather than assumed.

---

## Two findings that constrain how any result here may be read

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
`scaling.exponent_profile`, which returns the whole `alpha(x_min)` curve.

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

### Measuring the branching ratio

`expected_branching_ratio` computes it in closed form: given the seed, the
offspring count is a sum of independent Bernoulli trials, so its expectation is
a sum of conditional blocking probabilities — one pressure solve per seed, no
sampling noise. `critical_shift` and `shift_for_ratio` bisect it.

Both hold the seed subsample **fixed** across evaluations. Re-drawing it leaves
each call unbiased but makes every bisection step probe a slightly different
function; the first version of this did re-draw, and the bracket collapsed onto
a shift wrong by up to 0.4. Pinned as a test.

---

## Install and run

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pytest                   # ~3 min, includes the analytic checks
.venv/bin/python examples/validation.py      # ~2 min   estimator validation
.venv/bin/python examples/generations.py     # ~3 min   why R=1 is not critical
.venv/bin/python examples/finite_size.py     # ~37 min  is there a critical point?
.venv/bin/python examples/finite_size.py --replot   # ~1 s, redraw from the CSV
```

Every script takes `--quick` for a coarse version in well under a minute.
`figures/` is generated and git-ignored; each script also writes the CSV behind
every panel, so the numbers can be read without the picture — and
`finite_size.py --replot` restyles the figure from that CSV rather than
re-running the simulation.

## Layout

```
src/cascadence/     # the model
├── network.py      # conductance networks, edge removal, transit times
├── kernel.py       # blocking kernels; the only place kinetics enters
├── cascade.py      # avalanche dynamics, branching-ratio measurement
├── scaling.py      # discrete power-law MLE, cutoff tests, exponent profiles
└── fiberbundle.py  # ELS fiber bundle: the analytic validation fixture

src/voc/            # the clinical data pipeline — a sibling, not a submodule
├── codes.py        # ICD sets for sickle cell disease; trait excluded, audited
├── episodes.py     # the 3-day collapsing rule, and gap times
├── cohort.py       # crisis-encounter assembly and the go/no-go count
└── sql/            # extraction queries for MIMIC-IV
```

Read `fiberbundle.py` before trusting any exponent this library reports.

---

## The clinical side

`voc` exists because the modelling had outrun the data: three negative results
in a row, none of which settle anything, because no empirical fact was
constraining the model. What the distribution of inter-crisis intervals
actually looks like, whether crises cluster, and whether frequency tracks
severity are empirical questions nobody here has answered yet — and they decide
whether a cascade model is needed at all.

### The rule that is not optional

Around **17% of vaso-occlusive crisis ED encounters are followed by a revisit
within 3 days** (Walsh et al., *Am J Hematol* 2023; 40 US emergency
departments, 13,847 index encounters), and a revisit that fast is usually the
same under-treated crisis. Vaso-occlusion trials count a new crisis only when
it starts at least 3 days after the previous resolved.

Skipping that does not add noise, it manufactures signal. On the synthetic
fixture, uncollapsed encounters give a median interval of 17 days with 50% of
intervals under a week; collapsed at 3 days, the median is 60 days and none are
under a week. The same shift on real data would read as temporal clustering.
`examples/voc_feasibility.py` prints that sensitivity table every run.

### Which dataset, and why not the obvious one

**MIMIC-III is ICU-only** and therefore wrong: most crises never reach an ICU,
so it samples a selected tail.

**MIMIC-IV is hospital-wide but still unlikely to carry the interval
analysis.** Its source hospital is not a sickle cell centre; adult sickle cell
care in Boston concentrates elsewhere, and the statewide population is a few
thousand. On top of that it admits patients only via an ED or ICU touch,
excludes under-18s at first visit, and has no day-hospital encounter type —
which is where milder crises are treated. The per-patient event series is
non-randomly incomplete.

No published count exists for how many MIMIC-IV patients have three or more
crisis episodes, so `voc.cohort.feasibility` settles it by counting, against a
threshold written down first.

The intended split, which is why `voc.episodes` accepts both timestamps and
integer day offsets:

| what | from | why |
|---|---|---|
| intervals, clustering | HCUP SID + SEDD | follows a patient across facilities *and* care settings within a state, over years, at day resolution |
| severity proxies | MIMIC-IV | labs, administered opioid doses, transfusion, oxygen — no claims source has these |
| a cheap pilot on frequency vs severity | HCUP NRD | large and inexpensive, but its patient linkage resets every calendar year, so it cannot give interval distributions |

None of this is settled: the dataset numbers above come from secondary sources
and want verifying at the primary documentation before anything is purchased.

### Running it

```bash
.venv/bin/python examples/voc_feasibility.py --demo    # synthetic, no access needed
.venv/bin/python examples/voc_feasibility.py --encounters crisis_encounters.csv
```

Once credentialed: run `src/voc/sql/01_audit_codes.sql` and reconcile anything
it lists that `voc.codes.classify` calls `unknown`; **write the threshold down**;
then run `02_crisis_encounters.sql` and point the script at the export.

The pipeline is tested against synthetic MIMIC-shaped fixtures with declared
ground truth (`tests/voc_fixture.py`), so it is checkable before access exists
and stays checkable afterwards without committing data that cannot be committed.

## License

Code: MIT, see `LICENSE`. Manuscript text and figures under `paper/`, when that
exists: CC-BY-4.0.
