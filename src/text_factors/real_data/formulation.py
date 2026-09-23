"""P21: guarded surface realization of an already verified content plan.

The learned choice here concerns the *surface frame*, not factual paraphrase.
Every factual span is copied verbatim from a pinned, authorized source slice.
This deliberately narrow first step must not be advertised as free-language
generation or as proof of semantic faithfulness on unseen Russian text.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from .contracts import AnswerReceipt, SourceSlice
from .engine import RealDataEngine


@dataclass(frozen=True, slots=True)
class PlanQuote:
    claim_id: str
    text: str
    source: SourceSlice
    evidence_roots: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.claim_id or not self.text or not self.evidence_roots:
            raise ValueError("content plan requires a sourced nonempty quote")


@dataclass(frozen=True, slots=True)
class VerifiedContentPlan:
    receipt: AnswerReceipt
    quotes: tuple[PlanQuote, ...]

    def __post_init__(self) -> None:
        if not self.receipt.complete or not self.receipt.grounded:
            raise ValueError("content plan requires a complete grounded receipt")
        if (
            not self.quotes
            or tuple(q.claim_id for q in self.quotes) != self.receipt.claim_ids
        ):
            raise ValueError("content plan must cover every receipt claim exactly once")
        if len(set(self.receipt.claim_ids)) != len(self.quotes):
            raise ValueError("duplicate claim in content plan")
        if not set(root for q in self.quotes for root in q.evidence_roots) <= set(
            self.receipt.evidence_roots
        ):
            raise ValueError("plan quote evidence differs from receipt")


def plan_from_dialogue(answer: object) -> VerifiedContentPlan:
    """Adapt the P18 response shape without importing or bypassing its gate.

    The declared `answered` status alone does not prove publication. The P21
    draft still needs a *new* live grant read, and it cannot be published until
    the operational store supports a safe non-extractive receipt contract.
    """
    receipt = getattr(answer, "receipt", None)
    raw_quotes = getattr(answer, "quotes", None)
    if (
        getattr(answer, "status", None) != "answered"
        or not isinstance(receipt, AnswerReceipt)
        or getattr(answer, "state_version", None) != receipt.state_version
        or type(raw_quotes) is not tuple
    ):
        raise ValueError("only a published grounded dialogue answer is a plan")
    quotes: list[PlanQuote] = []
    for quote in raw_quotes:
        source = getattr(quote, "source", None)
        roots = getattr(quote, "evidence_roots", None)
        if not isinstance(source, SourceSlice) or type(roots) is not tuple:
            raise ValueError("invalid dialogue quote")
        quotes.append(
            PlanQuote(
                getattr(quote, "claim_id", ""),
                getattr(quote, "text", ""),
                source,
                roots,
            )
        )
    return VerifiedContentPlan(receipt, tuple(quotes))


@dataclass(frozen=True, slots=True)
class SurfaceStyle:
    """Only non-factual connective text can be learned as a surface frame."""

    lead: str = ""
    join: str = "\n"
    end: str = ""

    def __post_init__(self) -> None:
        if self.lead not in {"", "По источнику: ", "В материалах указано: "}:
            raise ValueError("unverified factual text in style lead")
        if self.join not in {"\n", "; ", ". Также указано: "}:
            raise ValueError("unverified factual text in style join")
        if self.end not in {"", "."}:
            raise ValueError("unverified factual text in style end")

    def render(self, quotes: tuple[str, ...]) -> str:
        if not quotes or any(not item for item in quotes):
            raise ValueError("style needs nonempty verified source quotes")
        return self.lead + self.join.join(quotes) + self.end


@dataclass(frozen=True, slots=True)
class StyleExample:
    group_id: str
    style: SurfaceStyle
    faithful: bool | None  # Unknown is not a negative or a positive example.
    quality: float = 0.0

    def __post_init__(self) -> None:
        if not self.group_id or not 0 <= self.quality <= 1:
            raise ValueError("invalid style example")
        if self.faithful is not None and type(self.faithful) is not bool:
            raise ValueError("faithfulness must be a reviewed boolean or unknown")


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    group_id: str
    quotes: tuple[str, ...]
    candidate: str
    quote_unsupported: bool | None
    free_unsupported: bool | None

    def __post_init__(self) -> None:
        if not self.group_id or not self.quotes or any(not q for q in self.quotes):
            raise ValueError("calibration requires a group and source quotes")
        for value in (self.quote_unsupported, self.free_unsupported):
            if value is not None and type(value) is not bool:
                raise ValueError("unknown label must not be treated as false")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    paired_groups: int
    quote_unsupported_groups: int
    free_unsupported_groups: int
    new_name_groups: int
    new_number_groups: int
    free_enabled: bool


_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_NAME = re.compile(r"\b[А-ЯЁA-Z][а-яёa-z]+\b")
_CONNECTIVE_WORDS = {"По", "В", "Также"}


def _new_anchors(quotes: tuple[str, ...], candidate: str) -> tuple[bool, bool]:
    baseline = " ".join(quotes)
    names = set(_NAME.findall(candidate)) - set(_NAME.findall(baseline))
    numbers = set(_NUMBER.findall(candidate)) - set(_NUMBER.findall(baseline))
    return bool(names - _CONNECTIVE_WORDS), bool(numbers)


def compare_to_excerpts(cases: tuple[CalibrationCase, ...]) -> CalibrationResult:
    """Count *independent history groups*; uncertain labels leave denominator.

    Reprints/revisions of a story count as one group. A failure in any revision
    marks the whole known group as a failure, rather than adding confidence.
    """
    grouped: dict[str, list[CalibrationCase]] = defaultdict(list)
    for case in cases:
        grouped[case.group_id].append(case)
    paired = quote_bad = free_bad = name_bad = number_bad = 0
    for group in grouped.values():
        if any(
            row.quote_unsupported is None or row.free_unsupported is None
            for row in group
        ):
            continue
        paired += 1
        quote_bad += int(any(row.quote_unsupported for row in group))
        free_bad += int(any(row.free_unsupported for row in group))
        name_bad += int(
            any(_new_anchors(row.quotes, row.candidate)[0] for row in group)
        )
        number_bad += int(
            any(_new_anchors(row.quotes, row.candidate)[1] for row in group)
        )
    return CalibrationResult(
        paired,
        quote_bad,
        free_bad,
        name_bad,
        number_bad,
        paired >= 2 and free_bad <= quote_bad and not name_bad and not number_bad,
    )


@dataclass(frozen=True, slots=True)
class FormulationDraft:
    text: str
    mode: str  # 'free_frame', 'excerpt', 'blocked'
    receipt: AnswerReceipt | None
    quotes: tuple[PlanQuote, ...]
    reason: str | None = None


class GuardedFormulator:
    """Learn a safe style on open groups; fall back on any known deterioration."""

    def __init__(self) -> None:
        self.style: SurfaceStyle | None = None
        self.training_groups: frozenset[str] = frozenset()
        self.calibration: CalibrationResult | None = None
        self._permanently_disabled = False

    def fit(self, examples: tuple[StyleExample, ...]) -> None:
        grouped: dict[SurfaceStyle, dict[str, list[StyleExample]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for item in examples:
            grouped[item.style][item.group_id].append(item)
        self.training_groups = frozenset(item.group_id for item in examples)
        scored: list[tuple[float, int, SurfaceStyle]] = []
        for style, groups in grouped.items():
            supported = [
                max(row.quality for row in rows)
                for rows in groups.values()
                if rows and all(row.faithful is True for row in rows)
            ]
            # A reviewed failure in even one group makes the style ineligible.
            unsafe = any(
                any(row.faithful is False for row in rows) for rows in groups.values()
            )
            if not unsafe and len(supported) >= 2:
                scored.append((sum(supported) / len(supported), len(supported), style))
        self.style = (
            sorted(scored, key=lambda row: (-row[0], -row[1], repr(row[2])))[0][2]
            if scored
            else None
        )
        self.calibration = None
        # A bad validation cannot be erased by retraining the same gate object.

    def calibrate(self, cases: tuple[CalibrationCase, ...]) -> CalibrationResult:
        if self.training_groups & {row.group_id for row in cases}:
            raise ValueError("calibration histories overlap with training")
        if self.style is None:
            raise ValueError("no trained style to calibrate")
        if any(row.candidate != self.style.render(row.quotes) for row in cases):
            raise ValueError("calibration candidate differs from trained style")
        result = compare_to_excerpts(cases)
        if not result.free_enabled:
            self._permanently_disabled = True
        self.calibration = result
        return result

    def draft(
        self,
        plan: VerifiedContentPlan,
        engine: RealDataEngine,
        *,
        allowed_scopes: tuple[str, ...],
        read_authorized_quote: Callable[[PlanQuote], str],
    ) -> FormulationDraft:
        """Produce a draft; the caller must recheck access at final publication.

        The reader MUST validate the live grant and source hash; a naked archive
        `read_span` is not an authorization check. No reader means no answer.
        """

        def blocked(reason: str) -> FormulationDraft:
            return FormulationDraft("", "blocked", None, (), reason)

        try:
            fresh = engine.receipt(
                question_id=plan.receipt.question_id,
                answer_text="verified draft",
                claim_ids=plan.receipt.claim_ids,
                model_version=plan.receipt.model_version,
                allowed_scopes=allowed_scopes,
            )
            if (
                not fresh.complete
                or fresh.state_version != plan.receipt.state_version
                or fresh.evidence_roots != plan.receipt.evidence_roots
            ):
                return blocked("grounding_changed")
            for quote in plan.quotes:
                claim = engine.claims[quote.claim_id]
                if (
                    claim.source != quote.source
                    or not set(quote.evidence_roots)
                    <= set(engine.evidence.claim_roots(quote.claim_id))
                    or not any(
                        engine.evidence.root(root).source == quote.source
                        for root in quote.evidence_roots
                    )
                ):
                    return blocked("quote_or_access_changed")
                current_text = read_authorized_quote(quote)
                if (
                    current_text != quote.text
                    or hashlib.sha256(current_text.encode("utf-8")).hexdigest()
                    != quote.source.sha256
                ):
                    return blocked("quote_or_access_changed")
            # The checked receipt is recreated with exactly the released body.
            excerpt = "\n".join(f"«{quote.text}»" for quote in plan.quotes)
            use_free = (
                self.style is not None
                and self.calibration is not None
                and self.calibration.free_enabled
                and not self._permanently_disabled
            )
            if use_free:
                style = self.style
                assert style is not None
                # Check even a restored/mutated style before using its literals.
                SurfaceStyle(style.lead, style.join, style.end)
                text = style.render(tuple(quote.text for quote in plan.quotes))
                # The exact textual proof is stronger than a name/number filter:
                # nothing beyond the vetted connective vocabulary can be added.
                if _new_anchors(tuple(q.text for q in plan.quotes), text) != (
                    False,
                    False,
                ):
                    use_free = False
            if not use_free:
                text = excerpt
            receipt = engine.receipt(
                question_id=plan.receipt.question_id,
                answer_text=text,
                claim_ids=plan.receipt.claim_ids,
                model_version=plan.receipt.model_version,
                allowed_scopes=allowed_scopes,
            )
            if not receipt.complete or receipt.state_version != fresh.state_version:
                return blocked("grounding_changed_during_render")
            return FormulationDraft(
                text, "free_frame" if use_free else "excerpt", receipt, plan.quotes
            )
        except (ValueError, KeyError, PermissionError):
            return blocked("source_or_access_unavailable")
