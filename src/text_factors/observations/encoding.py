"""Reversible open-alphabet input and a mapped, optional case-fold search view."""

from __future__ import annotations

import codecs
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


def encode_bytes(text: str) -> tuple[int, ...]:
    """Byte IDs are an input adapter, not semantic codes or a word vocabulary."""
    if type(text) is not str:
        raise ValueError("text must be a string")
    return tuple(text.encode("utf-8", errors="strict"))


def decode_bytes(tokens: Iterable[int]) -> str:
    values = []
    for token in tokens:
        if type(token) is not int or not 0 <= token <= 255:
            raise ValueError("invalid byte token")
        values.append(token)
    return bytes(values).decode("utf-8", errors="strict")


def utf8_chunks(
    blocks: Iterable[bytes], *, chunk_chars: int, max_bytes: int
) -> Iterator[str]:
    """Keep BOM, CRLF, controls and combining marks; reject invalid UTF-8.

    Byte blocks may split a UTF-8 sequence. Output boundaries are code-point
    boundaries, not sentence or semantic boundaries.
    """
    if type(chunk_chars) is not int or chunk_chars < 1:
        raise ValueError("invalid chunk size")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("invalid byte budget")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    pending = ""
    consumed = 0
    for block in blocks:
        if type(block) is not bytes:
            raise ValueError("input blocks must be bytes")
        consumed += len(block)
        if consumed > max_bytes:
            raise ValueError("source exceeds configured byte budget")
        for offset in range(0, len(block), 65_536):
            pending += decoder.decode(block[offset : offset + 65_536], final=False)
            while len(pending) >= chunk_chars:
                yield pending[:chunk_chars]
                pending = pending[chunk_chars:]
    pending += decoder.decode(b"", final=True)
    if pending:
        yield pending


@dataclass(frozen=True, slots=True)
class SearchView:
    text: str
    origins: tuple[int, ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(self.origins)
        ):
            raise ValueError("invalid search span")
        return self.origins[start], self.origins[end - 1] + 1


def casefold_view(text: str) -> SearchView:
    """One-to-many folds retain provenance; original observations never change."""
    encode_bytes(text)
    output: list[str] = []
    origins: list[int] = []
    for index, char in enumerate(text):
        folded = char.casefold()
        output.append(folded)
        origins.extend([index] * len(folded))
    return SearchView("".join(output), tuple(origins))
