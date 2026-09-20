"""Experimental domain-independent real-data primitives.

This package is intentionally isolated on the test branch. It composes with the
existing observations and mention-learning layers without replacing them.
"""

from .archive_bridge import evidence_root, source_slice
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
from .engine import RealDataEngine
from .evidence import EvidenceLedger, EvidenceRoot
from .persistence import load_state, save_state, state_payload
from .pilot import PilotGateConfig, PilotSample, evaluate_pilot
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
    "PilotGateConfig",
    "PilotSample",
    "RealDataEngine",
    "ResourceBudget",
    "RevisionCheckpoint",
    "RevisionResult",
    "RoleValue",
    "SourceSlice",
    "SparseCode",
    "SparseTransform",
    "UncertaintyIndex",
    "UncertaintyScope",
    "evaluate_pilot",
    "evidence_root",
    "jaccard",
    "load_state",
    "save_state",
    "source_slice",
    "state_payload",
]
