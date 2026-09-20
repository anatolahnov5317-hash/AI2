# Block 1 quality gate

This document describes the independent quality gate for the experimental
real-data route. It is intentionally separate from the original GUM
news/interview pilot and from the legacy v6 dialogue regressions.

## Frozen data selection

The exact selection is stored in `GUM_BLOCK1_GATE_SELECTION.json`. It is fixed
before fitting or opening the new test results:

- train: 12 whole documents, six academic and six biography;
- validation: 4 whole documents, two academic and two biography;
- test: 4 whole documents, two academic and two biography;
- none of these documents occur in the original news/interview pilot.

The source revision is fixed to
`22fdf87f9c71c96bcc771461d06e689b1f90020d`. The freezer verifies the local
Git checkout, published split metadata and hashes every source file before the
normal GUM converter is allowed to build the learning corpus.

## Frozen model configuration

The first Block 1 run uses:

- seed 17;
- 6 epochs;
- 8192 hashed features;
- contiguous mention candidates up to 8 Unicode-token units;
- up to 64 previous mentions as antecedent candidates;
- negative ratio 3;
- deterministic hard-negative mining for coreference pairs.

No entity-name, pronoun or domain vocabulary is introduced by this change.
The model still receives only external mention/entity supervision in train.

## Gate criteria

The thresholds are stored with the selection before the test split is evaluated.
All criteria must pass:

1. unsupported gold mention boundaries <= 5%;
2. learned exact-span F1 is not below the train-surface control;
3. validation enables the link gate with >= 90% empirical precision and at
   least 10 evaluable choices;
4. held-out accepted-link precision >= 80% with at least 10 evaluable choices;
5. coreference F1 with oracle mention boundaries is not below the nearest
   identical-surface control.

These are research gates for the next architecture step. Passing them does not
establish production readiness or general language understanding.

## Reproducibility

The GitHub Actions workflow `block1-quality-gate.yml` checks out the exact GUM
revision, freezes checksums, prepares the corpus, trains only on train, calibrates
only on validation and evaluates test only after the model and policy are frozen.
The resulting model, manifest, progress checkpoint and evaluation report are
uploaded as workflow artifacts.

A failed gate is a result, not a CI defect: the report should be inspected and
the architecture changed without weakening thresholds after observing test.
