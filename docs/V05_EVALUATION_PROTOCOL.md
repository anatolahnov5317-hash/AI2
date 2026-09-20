# v0.5 learned dialogue: prospective independent evaluation protocol

## Freeze and separation

This protocol precedes the first held-out prediction. The evaluation owner
authors `learning/evaluation_cases.py` independently. Production modules must
not import or inspect it. Production agents receive only interface contracts
and high-level coverage categories, never individual held-out texts, expected
outputs or error traces. Development examples are explicitly separate.

The language-training inventory supplied before test construction covers
bare locations, subject-first locations, actor/verb/object destinations,
actor/verb/object/dative transfers, possessive clauses, direct where/holder/
possession questions, verification, explicit corrections/retractions and
scoped promise/report/conditional statements. Development follows these
published families. Held-out language tests vary phrasal order and composition,
not merely entity strings within a single sentence template. Lexical aliases
remain declared data; no claim of unseen Russian morphology is made.

Before the first sealed prediction, the user also supplied public received-from
phrases with actor/recipient role reversal. The declared training inventory now
includes that family. Its addition does not come from evaluation feedback; no
sealed phrase or expected output was revised. Record the actual training and
checkpoint fingerprints in the freeze rather than implying the earlier
inventory was the final training data.

Three splits have separate hashes: public development, independent held-out,
and reserved challenge. A canonical source/model freeze manifest records
production-source hashes, actual numeric checkpoint digest, each component's
training fingerprint and corpus hashes. Held-out/challenge evaluation rejects
a missing or stale freeze. It is not sufficient to set a report label to
"held-out". Root explicitly authorizes the first evaluation after source freeze.
Any production change after seeing an evaluated split consumes that split as
development; the unused reserve remains available. A report cannot infer its
own independence from a high score.

The pre-prediction schema was extended with query spatial relation and query
polarity. No test phrases or expected relations were changed. Final corpus
serialization anchors after that ontology amendment:

```text
development  14468d13d23bc86594fed2c742bb9151da38d867335624a0dc033eae01944573
held_out     56d1ff43603f2402abb5bfc7dad2a223a7f07d66c2da71ee5da769ee538f455f
challenge    f1be18b8ab429fb23a24c0e4c0b7a1768dcf9330f58258584dd286c35835fede
```

## Measurements

| Component | Independent input | Measurement |
| --- | --- | --- |
| Understanding | Whole phrases, role swaps, scoped compositions, dialogue referents | Meaning-core exact match; role/path and scope agreement; abstention; all failures retained |
| Dynamics | Before/event episodes with role-bound counterexamples | Effect exact match; supported empty transition versus abstention; interpretation compatibility ordering |
| Policy | Multi-turn feature combinations | Unrestricted action prediction and safe admissibility separately |
| Generator | Action, slots, and independently supplied evidence | Token generation completion, evidence-copy grounding, invented entity mentions, safe refusal of unsupported slots |
| End-to-end session | Independent long dialogue sequences | State and answer assertions after every turn; correction/retraction, references, scope isolation, unsupported claims, grounded output, latency |

The ontology, role basis, factual projection, transactional guards and evidence
copy checks remain disclosed engineering constraints. Learned language heads,
transition memories, sequence policy and token probabilities are measured
separately. A count language model is a learned limited generator, not a
pretrained open-domain language model. Scored word order or exact finite
meaning trees do not establish unrestricted conversation.

Understanding results are stratified into independent phrasal families, scoped
composition, dialogue reference, and abstention/atomicity checks. A pooled score
does not replace those separate axes. Each stratum retains its full planned
denominator if an error or deadline prevents a prediction.

## Controls and denominators

Run trained, untrained, shuffled-training-label and training-only memorization
controls where the component API supports them. A control unavailable for a
component is explicitly marked unavailable; do not silently substitute a
different algorithm or an oracle. Memorization may inspect published training
examples only and must abstain on unmatched normalized complete phrases.
Controls use the identical frozen cases and no test-conditioned thresholds.

Each task reports requested, attempted, completed, correct, failed, skipped and
timed-out counts. Accuracy uses the full requested denominator; a partial
prefix also has a separately named observed-prefix score. Exceptions and safe
abstention on a required answer count as failures on that example. Safe
abstention is correct only when independently expected. Report full numerators
and denominators, never a headline percentage without its domain and sample
size. Incomplete control results cannot be called a pass.

`failed` comprises incorrect returned predictions plus execution errors;
`execution_failed` identifies only the latter. `timed_out` counts attempted
predictions that raise or report a timeout. `skipped` (also named
`not_attempted`) retains every remaining planned example, including unavailable
controls. Thus requested = correct + failed + timed_out + skipped. In contrast,
`completed` counts returned predictions whether correct or incorrect. A
completed experiment is not a semantic pass. Empty interpretation paths earn
no role or scope credit. Declared component timeouts and emergency session
responses cannot earn correct-abstention credit.

Dialogue exact-match scoring includes the independently expected action label.
A safe alternative label can therefore fail the combined turn metric while
state correctness and unsupported assertions remain separately visible. The
case labels and corpus hashes are preserved after such development findings.

## Reliability and audit trail

All work is bounded by a cooperative global deadline and per-call budgets.
The CLI must use a hard-timeout subprocess as well. Progress emits independent
strict-JSON snapshots after returned examples/turns. Completed prefixes survive
supervisor termination; an incomplete run is labelled incomplete explicitly.
Store inputs, expected/actual semantic structures, evidence, control identity,
latencies, errors, source/checkpoint/corpus/training hashes and component
attribution. Numeric checkpoint integrity and restore equivalence are separate
regression tests and not evidence of semantic competence.

No held-out or challenge predictions are run by ordinary unit tests. Unit tests
exercise fixtures, metrics, leakage guards, freeze validation and public
development examples only. Reserved failures must not be shown to production
agents for iterative hidden-test tuning.
