"""Compositional sparse encoding for open claims.

The exact hash is used only for integrity/exact identity. Similarity is carried by
role-aware sparse atoms so swapping arguments changes the lexical code while new
identifiers can still share structural code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .contracts import Claim


@dataclass(frozen=True, slots=True)
class SparseCode:
    width: int
    bits: tuple[int, ...]
    atoms: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("width must be positive")
        if tuple(sorted(set(self.bits))) != self.bits:
            raise ValueError("bits must be sorted and unique")
        if any(type(bit) is not int or not 0 <= bit < self.width for bit in self.bits):
            raise ValueError("bit outside sparse-code width")

    def jaccard(self, other: SparseCode) -> float:
        if self.width != other.width:
            raise ValueError("cannot compare codes with different widths")
        left, right = set(self.bits), set(other.bits)
        union = left | right
        return 1.0 if not union else len(left & right) / len(union)


@dataclass(frozen=True, slots=True)
class CompositionalRepresentation:
    exact_hash: str
    lexical: SparseCode
    structural: SparseCode

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_hash": self.exact_hash,
            "lexical": {
                "width": self.lexical.width,
                "bits": list(self.lexical.bits),
                "atoms": list(self.lexical.atoms),
            },
            "structural": {
                "width": self.structural.width,
                "bits": list(self.structural.bits),
                "atoms": list(self.structural.atoms),
            },
        }


class CompositionalEncoder:
    def __init__(
        self, *, width: int = 4096, active_bits_per_atom: int = 8, seed: int = 17
    ) -> None:
        if type(width) is not int or width < 128:
            raise ValueError("width must be at least 128")
        if (
            type(active_bits_per_atom) is not int
            or active_bits_per_atom <= 0
            or active_bits_per_atom >= width
        ):
            raise ValueError("invalid active_bits_per_atom")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.width = width
        self.active_bits_per_atom = active_bits_per_atom
        self.seed = seed

    def _atom_bits(self, atom: str) -> tuple[int, ...]:
        bits: set[int] = set()
        counter = 0
        prefix = f"{self.seed}:{atom}:".encode()
        while len(bits) < self.active_bits_per_atom:
            digest = hashlib.blake2b(
                prefix + str(counter).encode("ascii"), digest_size=32
            ).digest()
            for offset in range(0, len(digest), 4):
                value = int.from_bytes(
                    digest[offset : offset + 4], "little"
                )
                bits.add(value % self.width)
                if len(bits) >= self.active_bits_per_atom:
                    break
            counter += 1
        return tuple(sorted(bits))

    def _encode_atoms(self, atoms: list[str]) -> SparseCode:
        unique_atoms = tuple(sorted(set(atoms)))
        bits: set[int] = set()
        for atom in unique_atoms:
            bits.update(self._atom_bits(atom))
        return SparseCode(self.width, tuple(sorted(bits)), unique_atoms)

    @staticmethod
    def exact_hash(claim: Claim) -> str:
        payload = json.dumps(
            claim.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def encode_claim(self, claim: Claim) -> CompositionalRepresentation:
        lexical = [
            f"relation:{claim.relation_id}",
            f"status:{claim.status.value}",
        ]
        structural = [
            f"relation:{claim.relation_id}",
            f"status:{claim.status.value}",
        ]
        if claim.valid_from is not None:
            lexical.append(f"valid-from:{claim.valid_from}")
            structural.append("has-valid-from")
        if claim.valid_to is not None:
            lexical.append(f"valid-to:{claim.valid_to}")
            structural.append("has-valid-to")
        if claim.speaker_id is not None:
            lexical.append(f"speaker:{claim.speaker_id}")
            structural.append("has-speaker")
        for argument in claim.arguments:
            lexical.extend(
                (
                    f"role:{argument.role}",
                    f"value:{argument.value_id}",
                    f"type:{argument.value_type}",
                    f"arg:{argument.role}={argument.value_id}",
                    f"arg-type:{argument.role}={argument.value_type}",
                )
            )
            structural.extend(
                (
                    f"role:{argument.role}",
                    f"type:{argument.value_type}",
                    f"arg-type:{argument.role}={argument.value_type}",
                )
            )
        return CompositionalRepresentation(
            exact_hash=self.exact_hash(claim),
            lexical=self._encode_atoms(lexical),
            structural=self._encode_atoms(structural),
        )
