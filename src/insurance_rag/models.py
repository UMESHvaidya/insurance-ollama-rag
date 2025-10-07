"""Data models for the RAG system"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class CoverageStatus(str, Enum):
    """Enumeration of possible coverage statuses"""
    COVERED = "✅ Covered"
    NOT_COVERED = "❌ Not Covered"
    AMBIGUOUS = "⚠️ Ambiguous"

class ConfidenceLevel(str, Enum):
    """Confidence level of the analysis"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

@dataclass
class CoverageResponse:
    """Structured response for insurance coverage queries"""
    status: CoverageStatus
    explanation: str
    reference: str
    confidence: ConfidenceLevel
    query: str

    def __str__(self) -> str:
        """Format response for display"""
        return f"""
{self.status.value}
{self.explanation}
Reference: {self.reference}
Confidence: {self.confidence.value}
""".strip()

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "status": self.status.value,
            "explanation": self.explanation,
            "reference": self.reference,
            "confidence": self.confidence.value,
        }
