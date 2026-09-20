# Block 1 Quality Gate — TEST branch

Branch: `test/block1-quality-gate`

Status: experimental. This branch is derived from
`test/real-data-full-implementation` and exists to close the Block 1 stop
condition before further context/dynamics work.

## What this branch adds

1. A frozen, tamper-evident Block-1 evaluation gate.
   - The corpus fingerprint, train/validation/test fingerprints, test document
     IDs/groups, text fingerprints and mention fingerprints are frozen before
     evaluation.
   - Evaluation rejects a corpus or test split changed after the gate was frozen.
   - Test thresholds are never tuned from test predictions.

2. Explicit Block-1 slices.
   - Exact mention recovery on surfaces not seen in train.
   - Unsupported-gold span rate.
   - Same-surface/multiple-entity false merges.
   - Accepted-link decision count and empirical precision.
   - Delta from the simple memorization/same-surface baseline for mention F1 and
     end-to-end coreference F1.

3. Conservative identity-conflict training.
   - Distinct gold entities with the same case-folded surface are retained as
     negative training examples and cannot be sampled away.
   - Identifier-like strings with the same non-digit skeleton but conflicting
     digit runs are also retained as critical negatives.
   - Remaining negatives are still selected by bounded hard-negative mining.

4. CLI:
   - `text-factors mention-learning freeze-gate --corpus ... --output ...`
   - `text-factors mention-learning quality-gate --corpus ... --model ... --gate ... --output ...`

## Why this is separate from ordinary CI

Passing the unit/regression suite proves the implementation obeys its contracts.
It does not prove that mention extraction or coreference works on a new real
corpus. The frozen quality gate is intended to fail honestly until those
metrics satisfy the predeclared requirements.

The existing GUM pilot remains an already-open regression dataset. A new claim
of improvement requires a new pre-frozen held-out set that was not inspected or
selected using the predictions of the model being evaluated.

## Default experimental requirements

These are engineering gates for this experiment, not universal product claims:

- unsupported gold span rate <= 10%
- unseen-surface exact mention recall >= 25%
- zero same-surface/multiple-entity false merges
- accepted-link empirical precision >= 90%
- at least 10 evaluable accepted links
- mention F1 no worse than the simple baseline
- end-to-end coreference F1 no worse than the simple baseline

Changing these requirements after opening the held-out predictions creates a
new experiment and a new gate fingerprint.

## Next action after CI is green

Freeze a genuinely new held-out corpus, train only on its declared train split,
calibrate only on validation, and run the gate once. If it fails, keep the
failed report and improve the model from train/validation evidence only; do not
rewrite the held-out set around the failure.
