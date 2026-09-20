"""Experimental domain-independent real-data primitives.

This package is intentionally isolated on the test branch. It composes with the
existing observations and mention-learning layers without replacing them.
"""

from .budget import BudgetExceeded, BudgetSnapshot, BudgetTracker, ResourceBudget
from .composition import (
    CompositionalEncoder,
    CompositionalRepresentation,
    SparseCode,
)
from .contexts import ContextRegistry, ContextResult, SparseTransform, jaccard
from .contracts import (
    AnswerReceipt,
    Claim,
    ClaimStatus,
    Interpretation,
    LearningEpisode,
    RoleValue,
    SourceSlice,
    UncertaintyScope,
)
from .dependencies import DependencyGraph, RevisionCheckpoint, RevisionResult
from .evidence import EvidenceLedger, EvidenceRoot
from .uncertainty import UncertaintyIndex

__all__ = [
    "AnswerReceipt",
    "BudgetExceeded",
    "BudgetSnapshot",
    "BudgetTracker",
    "Claim",
    "ClaimStatus",
    "CompositionalEncoder",
    "CompositionalRepresentation",
    "ContextRegistry",
    "ContextResult",
    "DependencyGraph",
    "EvidenceLedger",
    "EvidenceRoot",
    "Interpretation",
    "LearningEpisode",
    "ResourceBudget",
    "RevisionCheckpoint",
    "RevisionResult",
    "RoleValue",
    "SourceSlice",
    "SparseCode",
    "SparseTransform",
    "UncertaintyIndex",
    "UncertaintyScope",
    "jaccard",
]
