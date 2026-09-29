# Data

**Nothing in this directory is tracked.** `data/.gitignore` ignores everything
except itself and this file, so restricted data cannot reach the repository by
accident. Do not relax that: a commit containing restricted data cannot be
fully undone — removing it needs a history rewrite *and* a request to GitHub to
purge cached views, and any clone taken in between keeps the data.

## Layout

```
data/
├── raw/        # exactly as obtained, never edited
├── interim/    # intermediate products of the pipeline
└── README.md   # this file, and the only tracked content besides .gitignore
```

## Obtaining the clinical data

Not yet resolved, and deliberately recorded as open rather than guessed at.
Candidate sources still to be verified — none of the following is confirmed:

- Whether the cohorts of interest are deposited with NHLBI BioLINCC, and under
  what access terms.
- Whether any daily-diary cohort with event-level timing is publicly deposited
  at all.
- What the data-use agreement permits regarding derived summary statistics,
  which decides whether fitted parameters may appear in a public repository.

Access is the long-lead item on this project: request it before writing
analysis code against it, because the terms may constrain what can be published.

When a source is settled, record here: the source, the access terms, the exact
extract used, and the date obtained. Someone reproducing this work needs to be
able to request the same extract — that is what makes the rest of the
repository meaningful without the data itself.

## What may be committed

Derived quantities only, and only if the data-use agreement allows:
aggregate summary statistics, fitted parameters, and figures. Never
record-level data, and never anything from which record-level data could be
reconstructed.
