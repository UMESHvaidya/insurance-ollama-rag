"""Insurance Policy RAG System with Local Ollama"""

from .rag_pipeline import InsurancePolicyRAG
from .models import CoverageResponse, CoverageStatus

__version__ = "0.1.0"
__all__ = ["InsurancePolicyRAG", "CoverageResponse", "CoverageStatus"]
