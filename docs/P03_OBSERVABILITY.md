# P03: public candidate diagnostics

`scripts/diagnose_open_mentions.py` analyzes frozen train/validation data without
access to held-out test documents. It does not fit the production model, change
thresholds, alter the archive or re-evaluate Gate 1/Gate 2. Prepare a separate
public-only JSON input with exactly three top-level fields:

```json
{
  "schema": "ai2-p03-public-splits-v1",
  "train": ["the original, unmodified train document objects"],
  "validation": ["the original, unmodified validation document objects"]
}
```

The two arrays must match the sorted train and validation arrays used by the
original training run. Their hashes are checked against the frozen fingerprints
in the model bundle. Never pass a file containing a `test` key or test document.
The tool rejects other top-level fields and non-public split labels.

```bash
PYTHONPATH=src python scripts/diagnose_open_mentions.py \
  --public-data artifacts/gum-public-only.json \
  --model artifacts/gum-block1-model/model.json \
  --output artifacts/p03-validation-diagnostics.json \
  --max-documents 3 --timeout 60
```

The report contains exact-boundary candidate recall **before** filtering by the
already frozen production mention threshold, plus the selected precision,
recall and F1 **after** filtering. Every annotated span remains in the
denominator, including spans that cannot be generated. It reports gold spans by
Unicode-token length (1, 2–4, 5–8, 9–16, 17+ or unaligned) and separately by
strict nesting. The length slices count false positives by their own span
length; nesting reports gold recall because there is no unique gold category for
an invented predicted span. Document-level counts help locate longer failures.

The existing `text_factors.evaluation.baselines.NGramMemory` scores entire
strings; it never proposed mention boundaries. The **new n-gram span adapter**
generates exactly the same contiguous Unicode-token candidates as the learned
model, trains that old n-gram scorer on *train mention surfaces only* and picks
an interval selection threshold using validation labels only. The report marks
that its validation score reuses calibration data and is **not an independent
test score**. The pre-existing Gate 1 comparison is an exact train-surface
lookup, which remains a separately named control. These controls must not be
reported as interchangeable.

For the longest public validation documents, one subprocess per document
measures wall time for `span_scores` and total peak worker RSS (Python, NumPy
and loaded model included). The worker sees text only; no gold labels are sent.
The process includes a timeout per document and the report retains timed-out
and failed documents in its requested/completed counts. The measurement is
operational context, not a fixed memory guarantee across machines. A future
closed gate can use these diagnostics after the model, inputs and comparisons
are frozen, without weakening its existing quality thresholds.
