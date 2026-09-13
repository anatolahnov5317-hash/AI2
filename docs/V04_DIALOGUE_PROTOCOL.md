# AI2 grounded dialogue protocol (prospective, 2026-09-13)

Base: `5fb1e5bd84fb5331dd2603decf2caea7a1f553ae`, AI2 0.3.0a4.
The unmodified source snapshot passes 280 tests. All 87 downloaded source,
test, configuration and documentation files match their original Git blob SHA.
Historical result archives are retained in the remote base tree, not rewritten.

## Scope and attribution

This increment builds a bounded Russian text dialogue about people, objects,
locations, transfers and corrections. It is not an open-domain language model.
No external LLM is used. The grammar, typed roles, world-state transitions,
correction semantics and output wording are explicit engineering scaffolds.
The AI2 contribution is learned raw lexical cue -> predicate interpretation:
`LearnedSDRTransform` -> common factor memory -> semantic portrait evidence.
No runtime target lookup or oracle fallback may substitute for missing learned
evidence. Questions/interpersonal acts can use explicit grammar scaffolding.

## Planned measurements and controls

Record complete input/expected/actual turn traces. Keep training lexical pairs,
development probes and held-out whole dialogues distinguishable. New entity
combinations and new lexical forms are separate axes; do not call the former
unseen-language generalization. Freeze scenario generation before evaluation.

Compare factor, untrained, shuffled target, nearest example and evaluator-only
oracle variants on the same dialogues. Report semantic answer success, state
updates, unsupported statements, abstention/clarification and elapsed time.
Do not interpret programmed event reasoning as discovered rules.
Use at least three fixed seeds (7, 17, 42). If revisions are made after inspecting
results, mark the data as development and preserve an additional challenge set.

## Reliability contract

Every text, token count, clause count, event history, entity set, teaching set,
candidate set and saved state has an explicit cap. Check cooperative deadlines
before and after model calls. Run evaluation in supervised processes with hard
deadlines, retain completed prefixes and never claim incomplete runs as passed.
Apply multi-clause turns transactionally: ambiguity/failure/timeout must not
leave partial facts. Separate hypothetical/reported statements from assertions.
Persist complete validated snapshots atomically, use no pickle, validate sizes,
types, schemas and integrity before replay. Restore must reproduce predictions
and dialogue state. Do not overwrite unrelated user files.

Pre-evaluation integration clarification: successful questions are retained as
dialogue-focus events, not facts. This fixes stale pronoun references after a
question. Known entity types come from the disclosed finite grammar tables;
unknown types abstain when an older person reference could be unsafe. Cached
receipt claims are checked against historical evidence and renderer output.

## Source boundary

- Redozubov, Formalization of the meaning, Parts 1-3 (2021): contextual
  interpretation, comparison with memory, syntax and initial teaching signals.
- Morzhakov and Redozubov (2017), arXiv:1712.05954: visual recognition, not chat.
- Wen et al. (2017), ACL E17-1042: separable dialogue tracking, policy, generation.
- Weston et al. (2015), arXiv:1502.05698: prerequisite toy reasoning tests,
  not certification of unrestricted conversational intelligence.

Success of a scripted constrained dialogue or of a nearest lookup baseline does
not establish AGI or superiority of AI2. Experimental failures remain reportable.
