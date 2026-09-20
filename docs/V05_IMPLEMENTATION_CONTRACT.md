# v0.5 learned dialogue implementation contract

All six goals are implemented as a new experimental track, not a claim of free
conversation or an implementation proven by Redozubov. v0.4 stays available.
The implementation is NumPy-only and uses no pretrained model, downloads,
external API, or implicit training from user messages.

## Interfaces and ownership

Shared types live in `text_factors.learning.schema`: `Entity`, `Event` (nested
promise/report/conditional scopes), `Query`, `Meaning`, `Interpretation`, and
`DialogueContext`. Meaning-tree shape is a disclosed ontology. Surface-language
mapping must be fitted from annotated complete phrases, not runtime grammar
rules for each phrasal pattern. Lexical entity aliases may be explicit annotated
data; this does not imply unseen entity morphology is solved.

- `understanding.py`, `language_data.py`: `LearnedUnderstanding.fit(...)`,
  `interpret(text, context=DialogueContext()) -> Interpretation`, numeric JSON
  `to_dict/from_dict`. A bounded `default_understanding(seed=42)` helper may fit
  training data for demos; production loads trained weights.
- `dynamics.py`, `transition_data.py`: train on before/event/after episodes,
  predict role-bound deltas; compare competing interpretations using a shared
  memory and explicit context transforms. Use the real existing factor core
  where feasible; if a different learned algorithm is used, disclose it.
- `dialogue_learning.py`, `dialogue_data.py`: learned dialogue policy trained on
  sequences, conditioned token generator trained from delexicalized responses;
  bounded decoding and independently checked evidence copying.
- `world.py`: root implementation, event-sourced factual projection with nested
  scopes, evidence sources, corrections, queries, abstention. Writes are
  transactional. Predicted deltas are not automatically accepted as facts.
- `model.py`, `session.py`, `commands.py`, `worker.py`: root integration, model
  checkpoint, explicit training CLI and supervised conversation.
- `evaluation.py`, independent test data: evaluation owner. Test phrasal families
  and entity combinations are separated from training and development. Frozen
  holdout is evaluated only after source freeze; no hidden-test result tuning.

All training APIs take a bounded time budget; all persisted numbers must be
finite and dimensions/capacities validated before allocation. Train into a new
model, commit only on success. CLI training and inference have hard subprocess
deadlines using the tested v0.4 supervisor. Checkpoints contain actual learned
parameters, version/schema, training data fingerprint, and configuration.

## Research boundary

Redozubov, Part 2 (2021), proposes contextual description transformation and
comparison of interpretations with common experience memory. Forward transition
prediction is one learning task, not an equivalent definition of interpretation.
Part 3 leaves substantial semantic-processing and reinforcement-learning work
open. This experiment fixes a bounded context/role basis rather than claiming
autonomous discovery of arbitrary concepts.

Sources:
- https://www.ontology-of-designing.ru/article/2021_3%2841%29/Ontology_Of_Designing_3_2021_4_A.D.Redozubov_309-319.pdf
- https://www.researchgate.net/publication/357581656_Formalization_of_the_meaning_Part_3_Formation_of_contexts
- https://arxiv.org/abs/2203.02155 (supervised demonstrations; not a claim of RLHF here)
- https://arxiv.org/abs/2410.10813 (memory evaluation dimensions, not benchmark reuse)

## Acceptance and reporting

Report separately: semantic exact match/roles/scope; transition exact match and
counterexamples; dialogue-policy accuracy; grounded generation and abstention;
multi-turn state corrections/reference resolution; checkpoint and timeout
integrity. Include untrained, shuffled, and memorization/baseline controls where
appropriate. Training examples, development examples, independent phrasal-family
holdout, and adversarial challenges are distinguishable. Failed or incomplete
runs remain failures, not silently dropped from denominators. No universal-chat
claim and no headline percentage without its denominator and scope.
