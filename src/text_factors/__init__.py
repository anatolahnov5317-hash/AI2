"""Text Factors: sparse, explainable associative memory for text."""

from .config import ModelConfig
from .encoder import DEFAULT_ALPHABET, LEGACY_ALPHABET, SparseSymbolEncoder
from .memory import ClusterStatus, FactorSummary, MemoryReadout
from .metrics import bit_precision_recall, jaccard_similarity
from .model import ConceptEvidence, TextFactorModel, TransformResult

__all__ = [
    "ClusterStatus",
    "ConceptEvidence",
    "DEFAULT_ALPHABET",
    "LEGACY_ALPHABET",
    "FactorSummary",
    "MemoryReadout",
    "ModelConfig",
    "SparseSymbolEncoder",
    "TextFactorModel",
    "TransformResult",
    "bit_precision_recall",
    "jaccard_similarity",
]

__version__ = "0.3.0a1"
