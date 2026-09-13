# Grounded dialogue cases: prospectively frozen corpus v1

The case builder in `evaluation/grounded_dialogue.py` is deterministic and does
not receive a model, predictions or a random seed. All controls use identical
ordered whole dialogues. Seeds 7, 17 and 42 affect the model only.

Three separately named splits were authored before prediction inspection:

| Split | Cases | Purpose |
| --- | ---: | --- |
| development | 3 | Public integration checks, never reported as hold-out evidence |
| held_out | 14 | Prospective whole-dialogue/entity-combination evaluation |
| challenge | 14 | Reserved entity/role permutation, not used in development tests |

Before-prediction SHA-256 anchors (also pinned in tests):

```text
development  9ee2dd82e861ff459d51176cb1ba9c04b6736e5adf269bd1ddd280d041a624be
held_out     d05b5e07f6c0cae7316bd1fd1c112b4def391e2afb2fc5a5ce3fb546e828892f
challenge    298ad1cbec0f7892a6d08fcf9f2bb2e33adba8ca1b2344ada0aa099731ea2b5e
```

The 14 families are movement with object reference, transfer, role reversal,
negative location, negative transfer, negative movement, correction followed by
retraction, ordinary retraction, topic isolation, unknown-object question,
hypothesis, untaught predicate, unsupported clause rollback and ambiguous
pronoun rollback. Each turn specifies the expected current-topic state using
typed semantic tuples. Query assertions and negative verification truth values
are checked independently of response wording and internal event numbering.

Training contains only explicitly taught raw lexical cue/predicate pairs, not
test dialogues, entities, expected states or answers. Ordinary positive cases
reuse taught cues and explicit grammar. Unknown-cue cases should abstain. The
claim is therefore **not** unseen-language generalization, acquired syntax,
discovered state-transition laws or open-domain conversational intelligence.
The oracle maps the same finite teaching cues to their labels, exists only in
the evaluator and does not consult the expected state. The nearest control is
explicit normalized-string lookup. Neither is a factor-inference fallback.

Every report embeds the exact corpus, its SHA-256, the teacher corpus and hash,
source hashes, complete expected/actual turn traces, semantic state/answer
scores, unsupported assertions, abstention, latency, failures and completion
counts. Safety declines are not confused with evaluator failures. Correctness
of free-form wording is not independently measured: semantic answer fields and
state are checked; renderer faithfulness needs its separate unit tests.

A returned prefix with `complete: false` is never a full-run pass. Cooperative
deadlines stop subsequent work. A supervising subprocess supplies hard timeout
protection; the progress callback emits independent full-report checkpoints
after returned turns so a supervisor can retain completed work.

If implementation or corpus is adjusted after inspecting a split's predictions,
that split becomes development data. Preserve the original report and use an
untouched reserve for a new claim. The challenge set varies entities and roles
within the same known task families, not the task definitions themselves.
