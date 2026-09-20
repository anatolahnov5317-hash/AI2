"""Bridge from the append-only observation archive into real-data contracts."""

from __future__ import annotations

import hashlib

from ..observations import ObservationArchive
from .contracts import SourceSlice
from .evidence import EvidenceRoot


def source_slice(
    archive: ObservationArchive,
    source_id: str,
    version: int,
    start: int,
    end: int,
) -> SourceSlice:
    record = archive.get_source(source_id, version)
    text = archive.read_span(source_id, version, start, end)
    scope = record.metadata.get("access_scope", "default")
    if type(scope) is not str or not scope:
        raise ValueError("source access_scope metadata must be a nonempty string")
    return SourceSlice(
        source_id=source_id,
        source_version=version,
        start=start,
        end=end,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        access_scope=scope,
    )


def evidence_root(
    archive: ObservationArchive,
    source: SourceSlice,
    *,
    root_id: str,
) -> EvidenceRoot:
    record = archive.get_source(source.source_id, source.source_version)
    return EvidenceRoot(
        root_id=root_id,
        group_id=record.group_id,
        source_id=record.source_id,
        source_version=record.version,
    )
