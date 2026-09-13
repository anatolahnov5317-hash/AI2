# Direct sparse-factor recovery diagnostics

This protocol evaluates the contents of local clusters directly. The v0.2
string benchmark scored maximum context familiarity; that score does not by
itself measure which hidden factors the output code recovers.

The protocol and parameters below were fixed before comparing the two
consolidation methods. Every seed and condition is retained, including null
controls and failures. This is a developmental diagnostic, not a preregistered
external benchmark or evidence of general intelligence.

## Data and isolation of ground truth

Seeds are **7, 17, and 42**. Each seed supplies 192 training observations and
64 independently drawn test observations in a 128-bit space. Four disjoint
12-bit masks are drawn once; their indices and occurrence labels are available
only to the evaluator. The learner receives a boolean observation vector,
without factor IDs, labels, or supervised targets. It receives one pass through
training observations. Test observations never update its memory.

| Condition | Observation generator | Evaluator targets |
|---|---|---|
| `single_clean` | First factor occurs with probability 0.5; other rows are empty | One planted factor |
| `multiple_clean` | Four factors occur independently with probability 0.35 each | Four planted factors |
| `multiple_nuisance` | Same factor occurrences plus four random distinct bits outside all factors | Four planted factors |
| `marginal_shuffle_control` | Independently shuffle each noisy-data column, separately within train and test | No planted factor-occurrence labels |

Training and test can contain identical full vectors, especially in the clean
conditions with only 2 or 16 possible combinations. Counts of unique vectors
and overlap are recorded. Consequently, this protocol tests structural recovery
and new draws, **not transfer to unseen combinations**. Context-transformation
experiments must test that separate claim.

The shuffled control preserves the activation count of every input bit in each
split. It changes row densities and the histories selected by a cluster's
activation threshold; it is not a fully matched control for those properties.
Both split column counts, density summaries, and hashes are reported.

## Frozen memory configuration

Both methods receive exactly the same observations, random receptors and output
mapping for each seed. They use 64 points, 48 receptive bits per point, 64 output
bits, creation threshold 4, activation threshold 3, probation after 8 matching
observations, stability after 24, and at most 32 clusters per point. Pruning uses
ratio 0.75. Coactivation uses at most 32 remembered rows and three passes.

`frequency` retains bits whose accumulated frequency is at least 0.75.
`coactivation` uses the implementation's floating-point Hebbian kernel and
retains normalized weights strictly greater than 0.75. This is a comparison of
the actual bounded implementations, not a causal ablation isolating only their
weight equation: history horizon, threshold boundary and order dependence also
differ. No thresholds are selected using test outcomes.

## Metrics

Only **stable** clusters enter the primary structural recovery measures.
Temporary and probationary counts remain visible. For a cluster at point `p`,
the target for factor `F` is the **local projection** `F ∩ receptor[p]`.
Projections smaller than the activation threshold are unobservable and are
excluded as matching targets. A global factor must not be used as the local
recall denominator.

Each stable cluster is matched to the eligible projection with highest F1;
ties use the lowest factor index. This is evaluator-only, many-to-one matching
against the generator's known structure. No mapping is learned from test
outcomes. Multiple clusters can recover one factor, so cluster count does not
count distinct concepts.

- Local precision, recall and F1 measure cluster bits against that projection.
- Exact local recovery requires identical bit sets. Unmatched cluster count is
  `stable_clusters - exact_local_clusters`; it does not assert that every
  unmatched cluster is useless in every task.
- Approximate recovery uses the fixed F1 threshold 0.8.
- Global bit recall unions exact local recoveries of each observable factor,
  then divides by the complete factor size. Per-factor observable coverage is
  reported alongside recovery. Factors with no eligible projection are counted
  separately and excluded from the mean.
- Held-out selectivity compares threshold activation of approximately matched
  clusters with the matched factor's occurrence label on test observations.
  Its precision, recall and F1 are macro-averaged over those clusters; the
  contributing cluster count is always reported. With no qualifying cluster,
  this metric is undefined rather than a successful zero-error result.

The null control has no planted factors; all its stable clusters are unmatched
and latent-factor coverage is undefined. Stability on random observations is
not evidence of a meaningful concept. Reference masks from the unshuffled
generator remain in the manifest for auditing, without being treated as truth
labels after shuffling.

## Isolated coactivation and order diagnostic

A separate kernel-level diagnostic injects an eight-bit candidate composed of
two exclusive four-bit groups. Sixteen rows alternate the groups. The union
never occurs, so normal observation would not create this candidate. Results
are explicitly labelled **isolated filter**, not end-to-end recovery.

A control applies 512 seeded attempts at degree-preserving 2×2 switches. It
preserves every row count and every column count while changing coactivation.
The original, reversed and switched histories use the same candidate and
threshold. Both frequency weights and the actual coactivation kernel's weights
and retained bits are recorded, without selecting a winning order. This exposes
the distinction between first moments and joint activations and the limitations
of an uncentered, finite-pass, single-component rule. It does not promise to
separate equal-strength independent factors.

## Reproducibility and bounds

`run_factor_recovery()` returns a JSON-serializable dictionary with protocol
version, source hashes, Python/NumPy versions, full model configurations, data hashes and masks,
all per-seed results, unweighted seed means, and the isolated diagnostics.
Fit and evaluation timings are separate. A missing metric remains `null` and
its contributing-seed count is reported in aggregates.

The default elapsed-time budget is 180 seconds per seed, checked between
training observations and evaluation points. A timeout marks that seed and the
overall report incomplete. Completed method/condition runs remain available;
missing runs never contribute invented zero scores. Aggregate rows identify
both requested and completed seeds. The process may continue with other seeds.
Reported NumPy-array bytes are only a lower bound: Python container and object
overhead, transient allocations, and process memory are not included.

Smaller explicit sample/point arguments are intended for smoke tests and appear
in the report. Inputs are bounded to 256 training and 256 test observations,
128 points, and 16 distinct seeds. These limits do not replace a process timeout
when running a complete suite on an unknown machine.

The coactivation direction comes from Alexey Redozubov's explanation of joint
activation and consolidation in
[part 12 of Logic of Consciousness](https://habr.com/ru/articles/326334/)
and the author's [original experiment](https://github.com/aldrd/aboutbrain).
The synthetic generators and evaluation protocol in this repository are our
engineering interpretation, not an experiment or claim attributed to him.
