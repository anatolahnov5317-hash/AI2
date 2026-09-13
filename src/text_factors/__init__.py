"""Text Factors: sparse, explainable associative memory for text."""

from .config import ModelConfig
from .context_affinity import ContextAffinity, ContextAffinityConfig
from .context_pipeline import LearnedContextPipeline, PipelineResult, TransformTrace
from .dialogue import GroundedDialogue, GroundingPolicy, WordResolution
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
    "ContextAffinity",
    "ContextAffinityConfig",
    "ContextView",
    "DEFAULT_ALPHABET",
    "LEGACY_ALPHABET",
    "LearnedContextPipeline",
    "InterpretationClaim",
    "FactorSummary",
    "GroundedDialogue",
    "GroundingPolicy",
    "MemoryReadout",
    "ModelConfig",
    "PipelineResult",
    "RecognitionCandidate",
    "RecognitionLimits",
    "RecognitionResult",
    "SparseSymbolEncoder",
    "TextFactorModel",
    "TransformResult",
    "TransformTrace",
    "WordResolution",
    "bit_precision_recall",
    "jaccard_similarity",
    "recognize_views",
]

__version__ = "0.3.0a3"
