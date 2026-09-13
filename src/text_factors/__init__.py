"""Text Factors: sparse, explainable associative memory for text."""

from .config import ModelConfig
from .encoder import DEFAULT_ALPHABET, LEGACY_ALPHABET, SparseSymbolEncoder
from .memory import ClusterStatus, FactorSummary, MemoryReadout
from .metrics import bit_precision_recall, jaccard_similarity
from .model import ConceptEvidence, TextFactorModel, TransformResult
from .recognition import (
    CandidateRelation,
    ContextView,
    InterpretationClaim,
    RecognitionCandidate,
    RecognitionLimits,
    RecognitionResult,
    recognize_views,
)

__all__ = [
    "ClusterStatus",
    "CandidateRelation",
    "ConceptEvidence",
    "ContextView",
    "DEFAULT_ALPHABET",
    "LEGACY_ALPHABET",
    "InterpretationClaim",
    "FactorSummary",
    "MemoryReadout",
    "ModelConfig",
    "RecognitionCandidate",
    "RecognitionLimits",
    "RecognitionResult",
    "SparseSymbolEncoder",
    "TextFactorModel",
    "TransformResult",
    "bit_precision_recall",
    "jaccard_similarity",
    "recognize_views",
]

__version__ = "0.3.0a2"
