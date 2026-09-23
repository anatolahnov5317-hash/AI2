"""Bounded lexical antecedent retrieval for train/validation ablations.

This index has no access to entity IDs, annotations or the archive. Its output
is a list of *candidates*, never a decision to merge two instances. The normal
production scorer continues to use a recency window unless the caller passes
this selector explicitly to the private assessment helper.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


class BoundedSurfaceRetrieval:
    """Reserve at most eight slots for older surface-related antecedents.

    Exact casefold matches support repeated object descriptions and names;
    optional common-prefix matches support inflected single-word surfaces such
    as «Иван», «Ивана» and «Ивану». Prefix matches remain proposals only.
    The index holds references to *mention positions*, never copied text/labels.
    """

    def __init__(
        self,
        text: str,
        spans: Sequence[Mapping[str, Any]],
        *,
        prefix_forms: bool = False,
    ) -> None:
        self._surfaces: list[str] = []
        self._exact: dict[str, list[int]] = defaultdict(list)
        self._prefix: dict[str, list[int]] = defaultdict(list)
        self._prefix_forms = prefix_forms
        for index, span in enumerate(spans):
            folded = text[span["start"] : span["end"]].casefold()
            self._surfaces.append(folded)
            if len(folded) >= 4:
                self._exact[folded].append(index)
                if prefix_forms and folded.isalpha():
                    self._prefix[folded[:4]].append(index)

    def __call__(self, current: int, budget: int) -> tuple[int, ...]:
        if type(current) is not int or not 0 <= current < len(self._surfaces):
            raise ValueError("invalid current mention index")
        if type(budget) is not int or budget < 1:
            raise ValueError("antecedent budget must be positive")
        start = max(0, current - budget)
        if start == 0 or budget == 1:
            return tuple(range(start, current))
        surface = self._surfaces[current]
        reserve = min(8, max(1, budget // 4))
        matches = self._exact.get(surface, ())
        cut = bisect_left(matches, start)
        old = list(matches[max(0, cut - reserve) : cut])
        if self._prefix_forms and len(surface) >= 4 and surface.isalpha():
            alternatives = self._prefix.get(surface[:4], ())
            prefix_cut = bisect_left(alternatives, start)
            # Bound prefix probes even when a common four-character prefix
            # appears thousands of times. Do not copy the history slice.
            lower_bound = max(-1, prefix_cut - 1 - reserve * 16)
            for offset in range(prefix_cut - 1, lower_bound, -1):
                index = alternatives[offset]
                if len(old) >= reserve:
                    break
                earlier = self._surfaces[index]
                if abs(len(earlier) - len(surface)) <= 3 and index not in old:
                    old.append(index)
        if not old:
            return tuple(range(start, current))
        old = old[:reserve]
        local = range(current - (budget - len(old)), current)
        return tuple(sorted({*local, *old}))
