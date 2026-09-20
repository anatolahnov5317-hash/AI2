"""Domain-independent records; offsets are Unicode code points and UTF-8 bytes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


def text_field(value: Any, name: str, *, cap: int = 4096, empty: bool = False) -> str:
    if type(value) is not str or len(value) > cap or (not value and not empty):
        raise ValueError(f"invalid {name}")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError(f"invalid Unicode in {name}") from exc
    return value


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("invalid JSON value") from exc


def fields(value: Any, required: set[str], optional: set[str] | None = None) -> dict:
    if (
        type(value) is not dict
        or not required <= value.keys()
        or value.keys() - required - (optional or set())
    ):
        raise ValueError("invalid observation record fields")
    return value


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    # Resource budgets, not language vocabulary or ontology restrictions.
    chunk_chars: int = 2048
    max_source_bytes: int = 64 * 1024 * 1024
    max_span_chars: int = 8192
    max_annotations: int = 10_000

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            integer(value, key, minimum=1)
        if self.chunk_chars > 65_536:
            raise ValueError("chunk_chars exceeds bounded working-buffer size")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    version: int
    namespace: str
    external_key: str
    group_id: str
    sha256: str
    byte_count: int
    char_count: int
    observation_count: int
    received_at: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    source_id: str
    source_version: int
    ordinal: int
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    raw: bytes
    sha256: str

    @property
    def text(self) -> str:
        return self.raw.decode("utf-8", errors="strict")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        del value["raw"]
        value["text"] = self.text
        value["semantic_status"] = "uninterpreted"
        return value


@dataclass(frozen=True, slots=True)
class Instance:
    instance_id: str
    namespace: str
    external_key: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Mention:
    mention_id: str
    source_id: str
    source_version: int
    char_start: int
    char_end: int
    surface: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Binding:
    mention_id: str
    version: int
    candidate_ids: tuple[str, ...]
    selected_id: str | None
    annotator: str
    evidence: str
    created_at: str

    @property
    def status(self) -> str:
        if self.selected_id is not None:
            return "annotated"
        return "unresolved" if not self.candidate_ids else "ambiguous"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status, "origin": "external_annotation"}
