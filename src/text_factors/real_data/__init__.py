"""Experimental domain-independent real-data primitives.

This package is intentionally isolated on the test branch. It composes with the
existing observations and mention-learning layers without replacing them.
"""

from .archive_bridge import evidence_root, source_slice
from .attention import AttentionCandidate, AttentionExample, HashedAttentionRanker
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
from .recognition import (
    ObservedReadout,
    PrequentialStep,
    ProspectiveReadout,
    ProspectiveResponse,
    evaluate_frozen_transfer,
    prequential_step,
    prospective_read,
    recognize_observation,
)
from .uncertainty import UncertaintyIndex

__all__ = [
    "AnswerReceipt",
    "AttentionCandidate",
    "AttentionExample",
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
    "HashedAttentionRanker",
    "Interpretation",
    "LearningEpisode",
    "ObservedReadout",
    "PrequentialStep",
    "PilotGateConfig",
    "PilotSample",
    "ProspectiveReadout",
    "ProspectiveResponse",
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
    "evaluate_frozen_transfer",
    "evidence_root",
    "jaccard",
    "load_state",
    "prequential_step",
    "prospective_read",
    "recognize_observation",
    "save_state",
    "source_slice",
    "state_payload",
]
