"""Tests for the RAG system"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from insurance_rag.models import CoverageStatus, CoverageResponse, ConfidenceLevel
from insurance_rag.document_loader import PolicyDocumentLoader
from insurance_rag.llm_analyzer import CoverageAnalyzer


class TestPolicyDocumentLoader:
    """Tests for document loader"""

    def test_init(self):
        loader = PolicyDocumentLoader(chunk_size=500, chunk_overlap=100)
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 100

    def test_load_nonexistent_pdf(self):
        loader = PolicyDocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_pdf("nonexistent.pdf")

    def test_load_non_pdf_file(self):
        loader = PolicyDocumentLoader()
        with pytest.raises(ValueError):
            loader.load_pdf("test.txt")


class TestCoverageAnalyzer:
    """Tests for coverage analyzer"""

    @patch("insurance_rag.llm_analyzer.ollama")
    def test_parse_covered_response(self, mock_ollama):
        # Mock ollama.list() for connection check
        mock_ollama.list.return_value = []

        analyzer = CoverageAnalyzer(model_name="gemma2:2b")

        llm_output = """

STATUS: Covered
EXPLANATION: The procedure is explicitly listed in the policy.
REFERENCE: Page 5, Section 2.1
CONFIDENCE: High
"""

        result = analyzer._parse_response(llm_output, "Test query")

        assert result.status == CoverageStatus.COVERED
        assert result.confidence == ConfidenceLevel.HIGH
        assert "explicitly listed" in result.explanation

    @patch("insurance_rag.llm_analyzer.ollama")
    def test_parse_not_covered_response(self, mock_ollama):
        mock_ollama.list.return_value = []

        analyzer = CoverageAnalyzer(model_name="gemma2:2b")

        llm_output = """

STATUS: Not Covered
EXPLANATION: This is explicitly excluded from coverage.
REFERENCE: Page 10, Exclusions section
CONFIDENCE: High
"""

        result = analyzer._parse_response(llm_output, "Test query")

        assert result.status == CoverageStatus.NOT_COVERED
        assert result.confidence == ConfidenceLevel.HIGH

    @patch("insurance_rag.llm_analyzer.ollama")
    def test_connection_error(self, mock_ollama):
        # Simulate Ollama not running
        mock_ollama.list.side_effect = Exception("Connection refused")

        with pytest.raises(ConnectionError):
            CoverageAnalyzer(model_name="gemma2:2b")


class TestCoverageResponse:
    """Tests for coverage response model"""

    def test_to_dict(self):
        response = CoverageResponse(
            status=CoverageStatus.COVERED,
            explanation="Test explanation",
            reference="Page 1",
            confidence=ConfidenceLevel.HIGH,
            query="Test query",
        )

        result = response.to_dict()

        assert result["status"] == "✅ Covered"
        assert result["query"] == "Test query"
        assert result["confidence"] == "High"

    def test_str_format(self):
        response = CoverageResponse(
            status=CoverageStatus.AMBIGUOUS,
            explanation="Unclear from policy",
            reference="Multiple sections",
            confidence=ConfidenceLevel.MEDIUM,
            query="Test query",
        )

        output = str(response)

        assert "⚠️ Ambiguous" in output
        assert "Unclear from policy" in output
        assert "Multiple sections" in output


class TestConfig:
    """Test configuration"""

    def test_default_settings(self):
        from insurance_rag.config import Settings

        settings = Settings(OLLAMA_BASE_URL="http://localhost:11434")

        assert settings.ollama_model == "gemma2:2b"
        assert settings.chunk_size == 800
        assert settings.retrieval_k == 4
