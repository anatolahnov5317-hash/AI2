"""Learned categorical dialogue policy and a small conditional token count LM.

Learning is explicit and supervised. The policy estimates categorical feature
likelihoods from full dialogue sequences, including preceding actions. The text
model estimates conditional unigram/bigram/trigram counts and decodes tokens;
it stores no complete responses and performs no complete-response retrieval.

This is not a general Russian generator. Typed copy slots, the available action
set, and an independent closed-language evidence verifier are HARD engineering
constraints, not learned discoveries. The verifier may reject fluent output.
Unknown entity morphology is not guessed: use canonical spelling, or a surface
form explicitly annotated in training. No implicit online updates occur.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .schema import bounded_text, exact_fields

ACTIONS = (
    "answer",
    "ack",
    "clarify",
    "unknown",
    "explain",
    "greet",
    "thanks",
    "help",
    "retracted",
    "corrected",
    "nonactual",
)
_DEFAULT_FEATURES: dict[str, Any] = {
    "act": "unknown",
    "has_answer": False,
    "ambiguous": False,
    "unsupported": False,
    "nonactual": False,
    "corrected": False,
    "retracted": False,
    "pending": "",
    "task": "",
    "prev_action": "",
    "evidence_count": 0,
    "failed": False,
}
_MAX_TURNS = 2048
_MAX_VOCAB = 256
_MAX_ROWS = 4096
_MAX_COUNT = 1_000_000
_COPY = frozenset({"object", "place", "holder", "prep", "source", "truth"})
_SLOT_KEYS = _COPY | {"reason"}
_BOS, _EOS = "<bos>", "<eos>"
_TOKEN = re.compile(r"<[a-z_]+>|[^\W_]+(?:[-_][^\W_]+)*|[^\s]", re.UNICODE)
_NAME = re.compile(
    r"[^\W_]+(?:[-_][^\W_]+)*(?: [^\W_]+(?:[-_][^\W_]+)*){0,3}\Z", re.UNICODE
)
_FORBIDDEN_NAME_WORDS = frozenset(
    {"не", "нет", "да", "в", "на", "у", "или", "и", "лежит", "передал", "есть"}
)


class LearningTimeout(TimeoutError):
    """A cooperative training/decoding deadline expired before commit."""


class _Deadline:
    def __init__(self, seconds: float) -> None:
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or not 0 < seconds <= 60
        ):
            raise ValueError("seconds must be finite in (0, 60]")
        self.end = perf_counter() + seconds

    def check(self) -> None:
        if perf_counter() >= self.end:
            raise LearningTimeout("learned_dialogue_deadline")


def _seed(value: Any) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("invalid dialogue seed")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.casefold()))


def _surface(tokens: Sequence[str], slots: Mapping[str, str]) -> str:
    pieces = [
        slots[token[1:-1]] if token.startswith("<") and token.endswith(">") else token
        for token in tokens
    ]
    text = " ".join(pieces)
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    return re.sub(
        r"(^|[.!?]\s+)([^\W\d_])",
        lambda match: match.group(1) + match.group(2).upper(),
        text,
    )


def _feature_value(value: Any) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        if not -1_000_000 <= value <= 1_000_000:
            raise ValueError("feature integer exceeds bound")
        return str(value)
    if type(value) is float:
        if not math.isfinite(value) or abs(value) > 1_000_000:
            raise ValueError("invalid feature number")
        return format(value, ".6g")
    return bounded_text(value, "policy feature", cap=128)


def _features(value: Any, *, previous_action: str | None = None) -> dict[str, str]:
    if type(value) is not dict or len(value) > 32:
        raise ValueError("features must be a bounded dictionary")
    for key in value:
        bounded_text(key, "feature name", cap=48, empty=False)
    result = {key: _feature_value(val) for key, val in _DEFAULT_FEATURES.items()}
    result.update({key: _feature_value(val) for key, val in value.items()})
    if previous_action is not None and "prev_action" not in value:
        result["prev_action"] = previous_action
    return result


def _evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or len(value) > 8:
        raise ValueError("evidence must contain at most eight records")
    result = []
    required = {"subject", "relation", "value", "negated", "spatial", "source"}
    for item in value:
        if type(item) is not dict or not required <= set(item) <= required | {
            "event_id"
        }:
            raise ValueError("invalid evidence fields")
        record = dict(item)
        for key in ("subject", "value", "source"):
            bounded_text(record[key], "evidence " + key, empty=False)
        if (
            type(record["relation"]) is not str
            or record["relation"] not in {"location", "holder"}
            or type(record["spatial"]) is not str
            or record["spatial"] not in {"in", "on"}
            or type(record["negated"]) is not bool
        ):
            raise ValueError("invalid evidence relation")
        if "event_id" in record and (
            type(record["event_id"]) is not int or not 1 <= record["event_id"] < 2**53
        ):
            raise ValueError("invalid evidence ID")
        result.append(record)
    return result


def _slots(value: Any) -> dict[str, str]:
    if type(value) is not dict or not set(value) <= _SLOT_KEYS:
        raise ValueError("invalid copy slot keys")
    return {key: bounded_text(val, "copy slot", cap=128) for key, val in value.items()}


def dialogues_from_data(value: Any) -> list[list[dict[str, Any]]]:
    """Strict JSON training loader shared by policy, generator and CLI."""
    if type(value) is not list or not 1 <= len(value) <= 128:
        raise ValueError("training must contain 1..128 dialogue sequences")
    total = 0
    for dialogue in value:
        if type(dialogue) is not list or not 1 <= len(dialogue) <= 64:
            raise ValueError("dialogue must contain 1..64 turns")
        total += len(dialogue)
        if total > _MAX_TURNS:
            raise ValueError("too many training turns")
        for turn in dialogue:
            exact_fields(
                turn,
                {"user", "features", "action", "response", "slots", "evidence"},
                "demonstration turn",
            )
            bounded_text(turn["user"], "training utterance", cap=2048, empty=False)
            bounded_text(turn["response"], "training response", cap=2048, empty=False)
            if type(turn["action"]) is not str or turn["action"] not in ACTIONS:
                raise ValueError("invalid demonstrated action")
            _features(turn["features"])
            _slots(turn["slots"])
            _evidence(turn["evidence"])
            response_tokens = _tokens(turn["response"])
            if not 1 <= len(response_tokens) <= 64 or any(
                token.startswith("<") and token not in {f"<{key}>" for key in _COPY}
                for token in response_tokens
            ):
                raise ValueError("invalid response tokens")
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: str
    scores: dict[str, float]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "scores": dict(self.scores),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GeneratedReply:
    text: str
    tokens: tuple[str, ...]
    grounded: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": list(self.tokens),
            "grounded": self.grounded,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Any) -> GeneratedReply:
        value = exact_fields(
            value, {"text", "tokens", "grounded", "reason"}, "generated reply"
        )
        bounded_text(value["text"], "generated text", cap=4096)
        bounded_text(value["reason"], "generation reason", cap=128)
        if (
            type(value["tokens"]) is not list
            or len(value["tokens"]) > 64
            or type(value["grounded"]) is not bool
        ):
            raise ValueError("invalid generated reply")
        for token in value["tokens"]:
            bounded_text(token, "generated token", cap=128, empty=False)
        return cls(
            value["text"], tuple(value["tokens"]), value["grounded"], value["reason"]
        )


class LearnedDialoguePolicy:
    """Categorical naive-Bayes policy fitted from sequence demonstrations."""

    def __init__(self, seed: int = 42, *, alpha: float = 0.1) -> None:
        self.seed = _seed(seed)
        if (
            type(alpha) not in (int, float)
            or not math.isfinite(alpha)
            or not 0 < alpha <= 1
        ):
            raise ValueError("policy smoothing must be finite in (0, 1]")
        self.alpha = float(alpha)
        self.action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
        self.tables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        self.training_fingerprint = ""
        self.training_turns = 0

    def fit(
        self, dialogues: Any, *, seconds: float = 5.0, shuffle_targets: bool = False
    ) -> LearnedDialoguePolicy:
        deadline = _Deadline(seconds)
        deadline.check()
        data = dialogues_from_data(dialogues)
        if type(shuffle_targets) is not bool:
            raise ValueError("shuffle_targets must be boolean")
        rows: list[tuple[dict[str, str], str]] = []
        for sequence in data:
            previous = ""
            for turn in sequence:
                deadline.check()
                rows.append(
                    (
                        _features(turn["features"], previous_action=previous),
                        turn["action"],
                    )
                )
                previous = turn["action"]
        labels = [label for _, label in rows]
        if shuffle_targets:
            np.random.default_rng(self.seed).shuffle(labels)
        feature_names = sorted({name for features, _ in rows for name in features})
        if len(feature_names) > 32:
            raise ValueError("too many feature names")
        tables = {}
        for name in feature_names:
            deadline.check()
            vocabulary = tuple(sorted({features.get(name, "") for features, _ in rows}))
            if len(vocabulary) > 128:
                raise ValueError("feature vocabulary exceeds limit")
            indices = {word: i for i, word in enumerate(vocabulary)}
            counts = np.zeros((len(ACTIONS), len(vocabulary)), dtype=np.int64)
            for (features, _), action in zip(rows, labels, strict=True):
                counts[ACTIONS.index(action), indices[features.get(name, "")]] += 1
            tables[name] = (vocabulary, counts)
        action_counts = np.asarray(
            [labels.count(action) for action in ACTIONS], dtype=np.int64
        )
        fingerprint = hashlib.sha256(
            _canonical(
                {"data": data, "shuffle_targets": shuffle_targets, "seed": self.seed}
            )
        ).hexdigest()
        deadline.check()
        self.tables, self.action_counts = tables, action_counts
        self.training_fingerprint, self.training_turns = fingerprint, len(rows)
        return self

    def choose(
        self, features: dict[str, Any], allowed_actions: Sequence[str]
    ) -> PolicyDecision:
        if (
            not isinstance(allowed_actions, (list, tuple, set, frozenset))
            or not 1 <= len(allowed_actions) <= len(ACTIONS)
            or any(type(a) is not str or a not in ACTIONS for a in allowed_actions)
        ):
            raise ValueError("invalid eligible action set")
        eligible = tuple(sorted(set(allowed_actions)))
        values = _features(features)
        scores = np.log(
            (self.action_counts + self.alpha)
            / (self.training_turns + self.alpha * len(ACTIONS))
        )
        for name, (vocabulary, counts) in self.tables.items():
            word = values.get(name, "")
            numerator = (
                counts[:, vocabulary.index(word)] + self.alpha
                if word in vocabulary
                else np.full(len(ACTIONS), self.alpha)
            )
            scores += np.log(
                numerator / (self.action_counts + self.alpha * (len(vocabulary) + 1))
            )
        selected_scores = {
            action: float(scores[ACTIONS.index(action)]) for action in eligible
        }
        action = max(eligible, key=lambda item: selected_scores[item])
        return PolicyDecision(
            action,
            selected_scores,
            "learned" if self.training_turns else "untrained_tie_or_prior",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ai2-learned-policy-v1",
            "seed": self.seed,
            "alpha": self.alpha,
            "action_counts": self.action_counts.tolist(),
            "training_turns": self.training_turns,
            "training_fingerprint": self.training_fingerprint,
            "tables": [
                {"name": name, "vocabulary": list(vocab), "counts": counts.tolist()}
                for name, (vocab, counts) in sorted(self.tables.items())
            ],
        }

    @classmethod
    def from_dict(cls, value: Any) -> LearnedDialoguePolicy:
        value = exact_fields(
            value,
            {
                "schema",
                "seed",
                "alpha",
                "action_counts",
                "training_turns",
                "training_fingerprint",
                "tables",
            },
            "policy checkpoint",
        )
        if value["schema"] != "ai2-learned-policy-v1":
            raise ValueError("invalid policy schema")
        model = cls(value["seed"], alpha=value["alpha"])
        total = _training_metadata(value)
        counts = _count_vector(value["action_counts"], len(ACTIONS))
        if (
            sum(counts) != total
            or type(value["tables"]) is not list
            or len(value["tables"]) > 32
        ):
            raise ValueError("inconsistent policy counts")
        tables = {}
        for row in value["tables"]:
            row = exact_fields(row, {"name", "vocabulary", "counts"}, "policy table")
            name = bounded_text(row["name"], "feature name", cap=48, empty=False)
            vocabulary = row["vocabulary"]
            if (
                name in tables
                or type(vocabulary) is not list
                or not 1 <= len(vocabulary) <= 128
            ):
                raise ValueError("invalid feature vocabulary")
            for item in vocabulary:
                bounded_text(item, "feature value")
            if (
                vocabulary != sorted(set(vocabulary))
                or type(row["counts"]) is not list
                or len(row["counts"]) != len(ACTIONS)
            ):
                raise ValueError("invalid policy table shape")
            matrix = [
                _count_vector(values, len(vocabulary)) for values in row["counts"]
            ]
            if any(sum(matrix[i]) != counts[i] for i in range(len(ACTIONS))):
                raise ValueError("policy likelihood totals disagree")
            tables[name] = (tuple(vocabulary), np.asarray(matrix, dtype=np.int64))
        if bool(total) != bool(tables):
            raise ValueError("missing trained policy parameters")
        model.action_counts, model.tables = np.asarray(counts, dtype=np.int64), tables
        model.training_turns, model.training_fingerprint = (
            total,
            value["training_fingerprint"],
        )
        return model


def _count_vector(value: Any, width: int) -> list[int]:
    if (
        type(value) is not list
        or len(value) != width
        or any(type(n) is not int or not 0 <= n <= _MAX_COUNT for n in value)
    ):
        raise ValueError("invalid numeric count vector")
    return list(value)


def _training_metadata(value: dict[str, Any]) -> int:
    total, fingerprint = value["training_turns"], value["training_fingerprint"]
    if (
        type(total) is not int
        or not 0 <= total <= _MAX_TURNS
        or type(fingerprint) is not str
        or (total and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None)
        or (not total and fingerprint)
    ):
        raise ValueError("invalid training metadata")
    return total


def _copy_name(value: str) -> None:
    if not _NAME.fullmatch(value) or any(
        word in _FORBIDDEN_NAME_WORDS for word in value.casefold().split()
    ):
        raise ValueError("unsafe entity copy surface")


class EvidenceVerifier:
    """Independent fixed evidence/role/negation checker, not a learned scorer."""

    def __init__(self, forms: Mapping[tuple[str, str], set[str]] | None = None) -> None:
        self.forms = {key: set(values) for key, values in (forms or {}).items()}

    def bind(
        self, slots: Any, evidence: Any
    ) -> tuple[dict[str, str], list[dict[str, Any]]]:
        provided, facts = _slots(slots), _evidence(evidence)
        if len(facts) > 1:
            raise ValueError("generate one verified fact per segment")
        bound = {}
        if provided.get("truth"):
            if provided["truth"] not in {"yes", "no"}:
                raise ValueError("invalid truth slot")
            bound["truth"] = "да" if provided["truth"] == "yes" else "нет"
        if not facts:
            if provided.get("truth") or any(
                provided.get(key) for key in _COPY - {"truth", "object"}
            ):
                raise ValueError("copy slots need evidence")
            # A query may name an object without supplying a fact about it.
            # Accept that topic, but do not expose it as an evidence-copy slot:
            # the limited nonfactual language can still say it does not know.
            if provided.get("object"):
                _copy_name(provided["object"])
            return bound, facts
        fact = facts[0]
        role = "place" if fact["relation"] == "location" else "holder"
        wrong_role = "holder" if role == "place" else "place"
        if provided.get(wrong_role):
            raise ValueError("copy role disagrees with evidence")
        for key, canonical in (("object", fact["subject"]), (role, fact["value"])):
            chosen = provided.get(key) or canonical
            annotations = self.forms.get((key, canonical.casefold()), set())
            if (
                key != "object"
                and chosen.casefold() == canonical.casefold()
                and annotations
            ):
                chosen = sorted(annotations)[0]
            _copy_name(chosen)
            accepted = {canonical.casefold()} | {
                form.casefold() for form in annotations
            }
            if chosen.casefold() not in accepted:
                raise ValueError("copy name is not supported by evidence")
            bound[key] = chosen
        prep = ("на" if fact["spatial"] == "on" else "в") if role == "place" else "у"
        supplied_prep = {"in": "в", "on": "на"}.get(
            provided.get("prep", ""), provided.get("prep", "")
        )
        if supplied_prep and supplied_prep != prep:
            raise ValueError("copy spatial relation disagrees with evidence")
        bound["prep"] = prep
        source = provided.get("source") or fact["source"]
        accepted_sources = {fact["source"]}
        if fact["source"] == "user" and "event_id" in fact:
            accepted_sources.add(f"сообщение {fact['event_id']}")
        if source not in accepted_sources or not _NAME.fullmatch(source):
            raise ValueError("unsupported evidence source")
        bound["source"] = source
        return bound, facts

    def verify(
        self,
        text: str,
        action: str,
        bound: Mapping[str, str],
        facts: Sequence[dict[str, Any]],
    ) -> tuple[bool, str]:
        try:
            bounded_text(text, "generated reply", cap=4096, empty=False)
        except ValueError:
            return False, "invalid_generated_text"
        tokens = _tokens(text)
        if not tokens or len(tokens) > 128:
            return False, "generation_length"
        factual = action in {"answer", "ack", "corrected", "explain"} and bool(facts)
        if factual:
            fact = facts[0]
            object_tokens = _tokens(bound["object"])
            role = "place" if fact["relation"] == "location" else "holder"
            value_tokens = _tokens(bound[role])
            prefix_words = {
                "ack": {"запомнил", "учёл", "учел", ":"},
                "corrected": {"исправил", "исправление", "учтено", ":"},
                "answer": {"по", "данным", ":", ",", "."},
                "explain": {"по", "данным", ":", ","},
            }[action]
            if "truth" in bound:
                prefix_words = prefix_words | {bound["truth"]}
            for start in range(len(tokens)):
                if tokens[start : start + len(object_tokens)] != object_tokens:
                    continue
                prefix = tokens[:start]
                if any(word not in prefix_words for word in prefix):
                    continue
                if ("truth" in bound) != (bound.get("truth") in prefix):
                    continue
                cursor = start + len(object_tokens)
                negative = cursor < len(tokens) and tokens[cursor] == "не"
                cursor += int(negative)
                if (
                    negative != fact["negated"]
                    or cursor >= len(tokens)
                    or tokens[cursor] != bound["prep"]
                ):
                    continue
                cursor += 1
                if tokens[cursor : cursor + len(value_tokens)] != value_tokens:
                    continue
                suffix = tokens[cursor + len(value_tokens) :]
                if action == "explain":
                    source = _tokens(bound["source"])
                    valid_suffixes = (
                        (".", "источник", ":", *source, "."),
                        (".", "основание", ":", *source, "."),
                    )
                    if suffix not in valid_suffixes:
                        continue
                elif suffix not in {(".",), ("!",)}:
                    continue
                return True, "verified_evidence"
            return False, "unsupported_fact_or_negation_or_role"
        if action in {"answer", "explain"}:
            return False, "missing_factual_evidence"
        words = set(tokens) - {".", ",", "!", "?", ":"}
        safe_words = {
            "unknown": {"этого", "я", "пока", "не", "знаю", "это", "мне", "известно"},
            "clarify": {
                "уточните",
                "пожалуйста",
                "какой",
                "предмет",
                "вы",
                "имеете",
                "в",
                "виду",
                "запрос",
                "поддерживаемом",
                "формате",
                "кого",
                "речь",
                "о",
                "ком",
            },
            "greet": {"привет", "здравствуйте", "добрый", "день"},
            "thanks": {"пожалуйста", "рад", "помочь"},
            "help": {
                "сообщите",
                "где",
                "предмет",
                "или",
                "задайте",
                "вопрос",
                "о",
                "нём",
                "нем",
                "помощь",
            },
            "retracted": {"отмена", "учтена", "утверждение", "отменено"},
            "nonactual": {
                "не",
                "отмечаю",
                "это",
                "действие",
                "как",
                "выполненное",
                "сохраняю",
            },
            "ack": {"сообщение", "учтено"},
            "corrected": {"исправление", "учтено"},
        }.get(action, set())
        if not words or not words <= safe_words:
            return False, "outside_nonfactual_safety_language"
        required = {
            "unknown": {"не"},
            "clarify": {"уточните"},
            "nonactual": {"не", "выполненное"},
            "retracted": set(),
            "ack": {"учтено"},
            "corrected": {"учтено"},
        }.get(action, set())
        if not required <= words or (
            action == "nonactual" and not words & {"отмечаю", "сохраняю"}
        ):
            return False, "missing_safety_qualifier"
        return True, "verified_nonfactual_language"


class LearnedTokenGenerator:
    """Count-trained conditional token LM, decoded with a bounded beam."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = _seed(seed)
        self.rows: dict[tuple[str, str, str], dict[str, int]] = {}
        self.vocabulary: tuple[str, ...] = ()
        self.verifier = EvidenceVerifier()
        self.training_fingerprint = ""
        self.training_turns = 0

    @staticmethod
    def _condition(
        action: str, facts: Sequence[dict[str, Any]], slots: Mapping[str, str]
    ) -> str:
        return "|".join(
            (
                action,
                facts[0]["relation"] if facts else "none",
                str(int(facts[0]["negated"])) if facts else "0",
                slots.get("truth") or "-",
            )
        )

    def fit(
        self, dialogues: Any, *, seconds: float = 5.0, shuffle_targets: bool = False
    ) -> LearnedTokenGenerator:
        deadline = _Deadline(seconds)
        deadline.check()
        data = dialogues_from_data(dialogues)
        if type(shuffle_targets) is not bool:
            raise ValueError("shuffle_targets must be boolean")
        turns = [turn for sequence in data for turn in sequence]
        forms: dict[tuple[str, str], set[str]] = defaultdict(set)
        for turn in turns:
            deadline.check()
            for fact in turn["evidence"]:
                role = "place" if fact["relation"] == "location" else "holder"
                for key, canonical in (
                    ("object", fact["subject"]),
                    (role, fact["value"]),
                ):
                    surface = turn["slots"].get(key) or canonical
                    _copy_name(surface)
                    forms[key, canonical.casefold()].add(surface)
        if len(forms) > 512 or any(len(variants) > 16 for variants in forms.values()):
            raise ValueError("surface annotation capacity")
        verifier = EvidenceVerifier(forms)
        targets = [turn["response"] for turn in turns]
        if shuffle_targets:
            np.random.default_rng(self.seed).shuffle(targets)
        rows: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
        vocabulary = {_EOS}
        for turn, response in zip(turns, targets, strict=True):
            deadline.check()
            bound, facts = verifier.bind(turn["slots"], turn["evidence"])
            tokens = _tokens(response)
            if not shuffle_targets:
                try:
                    rendered = _surface(tokens, bound)
                except KeyError as exc:
                    raise ValueError(
                        "training response uses unsupported copy slot"
                    ) from exc
                if not verifier.verify(rendered, turn["action"], bound, facts)[0]:
                    raise ValueError("training response is not evidence-grounded")
            condition = self._condition(turn["action"], facts, turn["slots"])
            previous2 = previous1 = _BOS
            for token in (*tokens, _EOS):
                deadline.check()
                vocabulary.add(token)
                for context in (condition, turn["action"]):
                    rows[context, previous2, previous1][token] += 1
                    rows[context, "", previous1][token] += 1
                    rows[context, "", ""][token] += 1
                previous2, previous1 = previous1, token
            if len(rows) > _MAX_ROWS or len(vocabulary) > _MAX_VOCAB:
                raise ValueError("token model capacity exceeded")
        fingerprint = hashlib.sha256(
            _canonical(
                {"data": data, "shuffle_targets": shuffle_targets, "seed": self.seed}
            )
        ).hexdigest()
        deadline.check()
        self.rows = {key: dict(counts) for key, counts in rows.items()}
        self.vocabulary, self.verifier = tuple(sorted(vocabulary)), verifier
        self.training_turns, self.training_fingerprint = len(turns), fingerprint
        return self

    def _distribution(
        self,
        action: str,
        condition: str,
        prefix: tuple[str, ...],
        allowed_slots: set[str],
    ) -> list[tuple[str, float]]:
        previous2, previous1 = ((_BOS, _BOS) + prefix)[-2:]
        probabilities: dict[str, float] = defaultdict(float)
        for specificity, context in ((0.9, condition), (0.1, action)):
            for weight, key in (
                (0.75, (context, previous2, previous1)),
                (0.2, (context, "", previous1)),
                (0.05, (context, "", "")),
            ):
                row = self.rows.get(key, {})
                total = sum(row.values())
                if not total:
                    continue
                for token, count in row.items():
                    if (
                        token.startswith("<")
                        and token != _EOS
                        and token[1:-1] not in allowed_slots
                    ):
                        continue
                    probabilities[token] += specificity * weight * count / total
        return sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:16]

    def generate(
        self,
        action: str,
        slots: dict[str, str],
        evidence: Sequence[dict[str, Any]],
        *,
        max_tokens: int = 48,
        seconds: float = 0.5,
        beam_width: int = 4,
    ) -> GeneratedReply:
        if type(action) is not str or action not in ACTIONS:
            raise ValueError("invalid generation action")
        if (
            type(max_tokens) is not int
            or not 1 <= max_tokens <= 64
            or type(beam_width) is not int
            or not 1 <= beam_width <= 8
        ):
            raise ValueError("invalid decoding capacity")
        deadline = _Deadline(seconds)
        if not self.training_turns:
            return GeneratedReply("", (), False, "untrained_generator")
        try:
            bound, facts = self.verifier.bind(slots, evidence)
        except ValueError as exc:
            return GeneratedReply(
                "", (), False, "invalid_evidence_copy: " + str(exc)[:90]
            )
        condition = self._condition(action, facts, slots)
        beams: list[tuple[tuple[str, ...], float]] = [((), 0.0)]
        completed: list[tuple[tuple[str, ...], float]] = []
        try:
            for _ in range(max_tokens + 1):
                deadline.check()
                following = []
                for prefix, score in beams:
                    for token, probability in self._distribution(
                        action, condition, prefix, set(bound)
                    ):
                        if probability <= 0:
                            continue
                        candidate_score = score + math.log(probability)
                        if token == _EOS:
                            if prefix:
                                completed.append((prefix, candidate_score))
                        elif len(prefix) < max_tokens:
                            following.append(((*prefix, token), candidate_score))
                following.sort(
                    key=lambda item: (-item[1] / max(1, len(item[0])) ** 0.7, item[0])
                )
                beams = following[:beam_width]
                completed.sort(
                    key=lambda item: (-item[1] / max(1, len(item[0])) ** 0.7, item[0])
                )
                completed = completed[:32]
                if not beams:
                    break
            for tokens, _ in completed:
                deadline.check()
                text = _surface(tokens, bound)
                valid, reason = self.verifier.verify(text, action, bound, facts)
                if valid:
                    return GeneratedReply(text, tokens, True, reason)
        except LearningTimeout:
            return GeneratedReply("", (), False, "generation_deadline")
        return GeneratedReply("", (), False, "no_grounded_token_sequence")

    def verify_generated(
        self,
        action: str,
        slots: dict[str, str],
        evidence: Sequence[dict[str, Any]],
        reply: GeneratedReply | str,
    ) -> tuple[bool, str]:
        if action not in ACTIONS:
            return False, "invalid_action"
        try:
            bound, facts = self.verifier.bind(slots, evidence)
            if isinstance(reply, GeneratedReply):
                if len(reply.tokens) > 64 or any(
                    token not in self.vocabulary or token == _EOS
                    for token in reply.tokens
                ):
                    return False, "invalid_decoder_tokens"
                if _surface(reply.tokens, bound) != reply.text:
                    return False, "surface_token_mismatch"
                text = reply.text
            else:
                text = reply
            return self.verifier.verify(text, action, bound, facts)
        except (ValueError, TypeError, KeyError):
            return False, "invalid_evidence_or_reply"

    def verify(
        self,
        action: str,
        slots: dict[str, str],
        evidence: Sequence[dict[str, Any]],
        reply: GeneratedReply | str,
    ) -> bool:
        return self.verify_generated(action, slots, evidence, reply)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ai2-token-count-lm-v1",
            "seed": self.seed,
            "vocabulary": list(self.vocabulary),
            "training_turns": self.training_turns,
            "training_fingerprint": self.training_fingerprint,
            "rows": [
                {"context": list(key), "counts": dict(sorted(values.items()))}
                for key, values in sorted(self.rows.items())
            ],
            "surface_forms": [
                {"role": role, "canonical": canonical, "forms": sorted(forms)}
                for (role, canonical), forms in sorted(self.verifier.forms.items())
            ],
        }

    @classmethod
    def from_dict(cls, value: Any) -> LearnedTokenGenerator:
        value = exact_fields(
            value,
            {
                "schema",
                "seed",
                "vocabulary",
                "training_turns",
                "training_fingerprint",
                "rows",
                "surface_forms",
            },
            "generator checkpoint",
        )
        if value["schema"] != "ai2-token-count-lm-v1":
            raise ValueError("invalid generator schema")
        model = cls(value["seed"])
        total = _training_metadata(value)
        vocabulary = value["vocabulary"]
        if type(vocabulary) is not list or len(vocabulary) > _MAX_VOCAB:
            raise ValueError("invalid token vocabulary")
        for token in vocabulary:
            bounded_text(token, "LM token", cap=128, empty=False)
            if len(_tokens(token)) != 1 or (
                token.startswith("<")
                and token not in {_EOS} | {f"<{key}>" for key in _COPY}
            ):
                raise ValueError("invalid model token")
        if (
            vocabulary != sorted(set(vocabulary))
            or (total and _EOS not in vocabulary)
            or type(value["rows"]) is not list
            or len(value["rows"]) > _MAX_ROWS
        ):
            raise ValueError("invalid LM vocabulary or rows")
        rows = {}
        for row in value["rows"]:
            row = exact_fields(row, {"context", "counts"}, "LM count row")
            key = row["context"]
            if type(key) is not list or len(key) != 3:
                raise ValueError("invalid LM context")
            for item in key:
                bounded_text(item, "LM context token", cap=128)
            context = tuple(key)
            parts = key[0].split("|")
            if not (
                (len(parts) == 1 and parts[0] in ACTIONS)
                or (
                    len(parts) == 4
                    and parts[0] in ACTIONS
                    and parts[1] in {"location", "holder", "none"}
                    and parts[2] in {"0", "1"}
                    and parts[3] in {"yes", "no", "-"}
                )
            ):
                raise ValueError("invalid semantic LM condition")
            if (
                context in rows
                or any(token not in {*vocabulary, "", _BOS} for token in key[1:])
                or type(row["counts"]) is not dict
                or not 1 <= len(row["counts"]) <= _MAX_VOCAB
            ):
                raise ValueError("invalid LM count row")
            counts = row["counts"]
            if any(
                type(token) is not str
                or token not in vocabulary
                or type(count) is not int
                or not 1 <= count <= _MAX_COUNT
                for token, count in counts.items()
            ):
                raise ValueError("invalid learned token counts")
            rows[context] = dict(counts)
        forms = {}
        if (
            type(value["surface_forms"]) is not list
            or len(value["surface_forms"]) > 512
        ):
            raise ValueError("invalid surface forms")
        for row in value["surface_forms"]:
            row = exact_fields(
                row, {"role", "canonical", "forms"}, "surface annotation"
            )
            if type(row["role"]) is not str or row["role"] not in {
                "object",
                "place",
                "holder",
            }:
                raise ValueError("invalid annotation role")
            canonical = bounded_text(
                row["canonical"], "canonical copy name", empty=False
            )
            if (
                canonical != canonical.casefold()
                or type(row["forms"]) is not list
                or not 1 <= len(row["forms"]) <= 16
            ):
                raise ValueError("invalid annotated surface forms")
            for form in row["forms"]:
                bounded_text(form, "surface form", empty=False)
                _copy_name(form)
            key = (row["role"], canonical)
            if key in forms or row["forms"] != sorted(set(row["forms"])):
                raise ValueError("duplicate surface forms")
            forms[key] = set(row["forms"])
        if bool(total) != bool(rows) or (not total and (vocabulary or forms)):
            raise ValueError("inconsistent trained LM metadata")
        model.rows, model.vocabulary, model.verifier = (
            rows,
            tuple(vocabulary),
            EvidenceVerifier(forms),
        )
        model.training_turns, model.training_fingerprint = (
            total,
            value["training_fingerprint"],
        )
        return model


def default_policy(seed: int = 42, *, seconds: float = 5.0) -> LearnedDialoguePolicy:
    from .dialogue_data import training_dialogues

    return LearnedDialoguePolicy(seed).fit(
        training_dialogues(seed=seed), seconds=seconds
    )


def default_generator(seed: int = 42, *, seconds: float = 5.0) -> LearnedTokenGenerator:
    from .dialogue_data import training_dialogues

    return LearnedTokenGenerator(seed).fit(
        training_dialogues(seed=seed), seconds=seconds
    )
