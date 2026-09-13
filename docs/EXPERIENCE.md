# Experimental learning from observed experience

`text_factors.experience` connects the existing supervised factor memory to
explicit observations, context pairs and actions. It is a bounded integration
layer, not a new rule engine. The learner receives sparse input/output examples;
it does not receive the demonstration world's rule function.

## Observation boundary

`SparseCode(encoding_id, width, active_bits)` holds immutable, sorted, unique
active indices. Encodings are supplied by the caller. A bank rejects a different
encoding version or width, and requires equal input/output widths. It does not
migrate or discover encodings automatically.

`ContextKey(source_context, target_context, action)` identifies one independently
trained `CombinatorialMemory`. The same input under different actions or views
cannot silently train the same operator. Context and action labels are supplied,
not inferred.

Only `learn(ObservedTransition(...))` adds evidence. `Forecast` is a separate type
and is rejected by that method. Constructing an `ObservedTransition` is the
caller's assertion that an outcome was actually observed: the API cannot verify
physical truth or stop a caller deliberately relabelling a prediction as reality.

An identical retained event ID is a no-op. A conflicting duplicate is rejected.
New event IDs must increase across the whole bank. If an old event was evicted
from the bounded archive, reusing its ID is rejected rather than counted as new
evidence. Out-of-order delivery is unsupported. Separate actual interactions may
have identical codes, but require separate event IDs.

`replay_consolidation()` makes one bounded pass through existing memories. It
uses stored activation history and creates no clusters, observations, memory
steps or independent confirmations. It does not call `learn()` on archived
episodes. Replay may still prune existing clusters.

## Forward predictions, inverse lookup and actions

`forecast(key, source)` reads the learned supervised transformation. Unseen keys
return an empty prediction and zero training events without allocating a memory.
Empty output is visible to the caller. Votes and quality are not calibrated
probabilities.

`predecessors(key, outcome)` returns all distinct retained sources of that exact
observed outcome, together with their event IDs. Several predecessors may be
valid. This is context-filtered episodic retrieval, not a learned inverse for
unseen outcomes, and it does not reconstruct a source from a compact hash.

`choose_action(source, candidates, utility, exploration_probability=...)` scores
a bounded supplied set of actions using learned forecasts. The caller supplies
the goal utility. Candidates must share source/target contexts. Ties have a
deterministic lexical order, and optional epsilon exploration uses the recorded
policy seed. Selection does not train the factor memories. The external
environment must subsequently provide an actual outcome before learning occurs.

Capacity bounds cover operation count, retained episodes, candidate actions and
clusters per point. New operators are refused at capacity. Evicting an episode
does not delete its learned clusters. NumPy payload statistics include receptor
arrays, output maps, cluster indices/hits and activation histories; they exclude
Python/container overhead, archive objects and temporary allocations, so they
are a lower bound rather than total RAM or process RSS. The bank is
single-threaded and whole-bank serialization is not implemented.

## Executable probe

After installing the package, run:

```bash
python -c 'import json; from text_factors.experience import run_experience_demo; print(json.dumps(run_experience_demo(seed=42), indent=2))'
```

The probe uses a 32-bit structured environment with two mutually exclusive factor
classes and a changing distractor. View A uses factor coordinates `(0, 1)` and
`(2, 3)`; view B uses different coordinates `(16, 17)` and `(18, 19)`. A learned
paired-view operator and one action operator both target common outcome codes.
The second action has reversed outcomes. All three operators use the factor
memory with bounded coactivation history.

Training provides 36 actual paired observations: six distractors, two factors,
three operators. Twelve combinations with two unseen distractors are scored
without learning. Evaluator-only factor IDs and actual outcomes appear in the
report after predictions are made; they are not an input to the learner. A third
unseen distractor is used for one separate action, actual outcome and learning
step after evaluation.

The revised probe checks that views have different input codes, that the same
factor maps to a common nonempty output across views, and that the two factors
remain distinct. An all-zero predictor has 0% exact match; a constant output has
50% exact match and cannot distinguish the factors. These controls prevent code
collapse from passing the agreement check.

Development runs with seeds 7, 17 and 42 each produced 12/12 exact predictions,
bit precision/recall 1.0, nonempty agreement across different views, and distinct
factor outputs. Historical inverse lookup retained six different predecessors
for one outcome. Each run selected the action whose learned output satisfied the
supplied goal. After the separate interaction the bank had 37 observations;
replay added zero observations and zero steps. The final probe uses class 1,
whose useful action is `right` (last in lexical order), so always choosing
the first action cannot pass this check.

On the development runtime, fitting took approximately 0.046–0.058 seconds,
scoring all twelve examples 0.00037–0.00058 seconds, and replay 0.008–0.016 seconds.
Final NumPy payload lower bounds were 213,479–213,531 bytes across the three
seeds. These single-run timings are diagnostic, not a hardware-normalized speed
benchmark. Each report records its measured timings; repeatability comparisons
exclude those timing fields.

This is an easy, supervised integration gate with structured encodings and
explicit operator identities. It establishes that the new observation boundary
can use factor memories for transfer, agreement and action feedback in this toy
setting. It does not establish autonomous concept discovery, advantage over
frequency consolidation, recovery from noisy real-world views, planning ability,
or superiority to an LLM. Stronger factor and transformation benchmarks remain
separate from this probe.
