# Transparent active-learning microworld pilot

This pilot isolates one narrow question: can a learner use one informative action
to distinguish two possible interaction protocols and transfer that distinction to
new identifiers and attribute recombinations?

> **The complete hypothesis class is exactly two hand-specified rules.**
>
> 1. `color_match`: success when `object.color == lock.color`
> 2. `shape_match`: success when `object.shape == lock.shape`
>
> The rules and structured attributes are supplied by the experiment author. They
> are not learned by the text-factor SDR, induced as a DSL, or evidence that the
> system has learned causality or achieved AGI.

## Protocol

Each episode has one hidden rule. With two or more episodes, the runner alternates
the two rules so the report covers both. The authoritative environment computes an
outcome from color or shape only; object and lock identifiers never enter the
calculation.

Every strategy receives the same two initial observations. One has both color and
shape matching and succeeds; the other has neither matching and fails. Both
hypotheses therefore remain consistent. The unlabeled candidate pool contains
color-only matches, shape-only matches, and examples on which the rules agree.

The active learner evaluates candidates using expected information gain under a
uniform version-space prior and uses a deterministic semantic tie-break. It sees no
hidden rule, generator state, or candidate outcome. A real action is the only way
to expose a candidate outcome. Once the two hypotheses disagree, the learner
predicts probability `0.5` and abstains. One discriminating result retains exactly
one rule.

Held-out outcomes are computed only after every strategy finishes selecting and
observing its actions. Held-out objects and locks use fresh identifiers. Objects
use unseen attribute combinations; lock attribute combinations may already occur
in the candidate pool. The held-out interactions are never eligible queries.

## Baselines and metrics

All strategies share the initial observations, candidate pool, and maximum action
budget:

- `active` uses the two-rule learner and maximum information gain.
- `random` uses the same two-rule learner but a seeded random candidate ordering.
- `episodic` takes the exact same actions as `random` and memorizes full interactions,
  including identifiers. It abstains on a new interaction rather than generalizing.

`success` is correct held-out predictions divided by all held-out trials, so an
abstention is not counted as success. `coverage` is the predicted fraction,
`accuracy_on_covered` is conditional accuracy (defined as `0.0` at zero coverage),
and `query_cost` is real candidate actions per episode. All metrics are finite.

The action budget is a maximum. The active learner stops after its uncertainty is
resolved or when no allowed candidate has positive information gain. Random and
episodic baselines spend the requested budget.

## Contradictions

If an observation contradicts every candidate rule, the version space becomes
empty. The learner does **not** reset. It returns `state="unknown"`,
`probability=null`, and `abstained=true` until a caller explicitly constructs a new
learner. This prevents misspecification from being reported as false certainty.

## Python API and report

```python
from text_factors.microworld import run_microworld

report = run_microworld(seed=42, episodes=32, action_budget=1)
```

The result contains a `manifest`, per-episode traces, held-out outcomes and
predictions, and aggregate metrics for `active`, `random`, and `episodic`. It
contains only JSON-serializable values. The same validated inputs produce an
identical report.

This is deliberately a single-intervention/transfer demonstration. It is not a
multi-step planner, an open-ended causal learner, or a learned rule language.
