# TEST BRANCH — real-data full implementation

> **TEST / EXPERIMENTAL ONLY.**
>
> This branch is intentionally isolated from `main` and from the staged
> implementation branches. It is a research integration branch for the
> real-data architecture described in the September 20, 2026 specifications.
> Nothing in this branch is production-ready by default and nothing should be
> merged to `main` without passing the explicit gates below.

Branch: `test/real-data-full-implementation`

Base: `implementation/open-mention-learning`

## Purpose

This branch integrates the open-observations and mention-learning work with
new domain-independent primitives required for the real-data route:

- explicit resource/training budgets and resumable progress;
- compositional claim encoding that preserves relation and role direction;
- learnable sparse context transforms and local-maxima selection;
- independent evidence roots so repeated derivations do not inflate support;
- scoped uncertainty records instead of one global unknown/stale flag;
- dependency-driven revision in bounded resumable batches;
- pilot-gate metrics for groundedness and useful coverage.

The old v6 and learned-chat paths remain available as regression controls.

## Non-goals

- This branch does **not** claim general intelligence.
- It does **not** claim the new language/coreference models are solved.
- It does **not** silently promote model predictions to observed facts.
- It does **not** remove engineering limits; limits are explicit contracts.

## Merge gates

Before any merge into a supported branch:

1. CI must pass on Python 3.10/3.11/3.12.
2. New tests must cover budget stops, compositional role direction,
   local-maxima suppression, evidence deduplication, scoped uncertainty,
   dependency revision, and checkpoint/resume.
3. Regression suites from the base branch must remain green.
4. The implementation status document must distinguish implemented,
   experimental and still-missing requirements.
