"""LLM-based coverage analysis using local Ollama"""

import re
from typing import Optional
import ollama

from .models import CoverageResponse, CoverageStatus, ConfidenceLevel


class CoverageAnalyzer:
    """Analyzes insurance coverage using local Ollama LLM"""

    PROMPT_TEMPLATE = """You are an insurance policy analyst. Analyze the policy excerpts and determine coverage.

POLICY EXCERPTS:
{context}

CHAT HISTORY:
{history}

USER QUERY: {question}

TASK: Determine if the item is Covered, Not Covered, or Ambiguous.

Respond in this EXACT format:
STATUS: [Covered/Not Covered/Ambiguous]
EXPLANATION: [2-3 sentences explaining your determination]
REFERENCE: [Cite page numbers and sections if found]
CONFIDENCE: [High/Medium/Low]

Be precise and cite specific policy language when available."""

    def __init__(
        self,
        model_name: str = "gemma2:2b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ):
        """
        Initialize coverage analyzer with Ollama

        Args:
            model_name: Ollama model name (e.g., gemma2:2b, llama2, mistral)
            base_url: Ollama API base URL
            temperature: Temperature for generation
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

        # Verify Ollama connection
        try:
            ollama.list()
            print(f"✓ Connected to Ollama at {base_url}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                f"Make sure Ollama is running: 'ollama serve'. Error: {e}"
            )

    def analyze(self, query: str, context: str, history: list[str] = None) -> CoverageResponse:
        """
        Analyze coverage for a query using Ollama

        Args:
            query: User query
            context: Retrieved policy context
            history: Optional list of previous Q&A strings

        Returns:
            CoverageResponse object
        """
        # Format history
        history_text = "\n".join(history) if history else "No previous history."

        # Build prompt
        prompt = self.PROMPT_TEMPLATE.format(
            context=context,
            history=history_text,
            question=query
        )

        try:
            # Call Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": 300,  # Limit response length
                },
            )

            result_text = response["response"]

        except Exception as e:
            # Fallback response if Ollama fails
            print(f"⚠️ Ollama error: {e}")
            result_text = """STATUS: Ambiguous

EXPLANATION: Unable to analyze due to system error. Please try again.
REFERENCE: N/A
CONFIDENCE: Low"""

        # Parse response
        coverage = self._parse_response(result_text, query)

        return coverage

    def _parse_response(self, llm_output: str, query: str) -> CoverageResponse:
        """
        Parse LLM output into structured CoverageResponse

        Args:
            llm_output: Raw LLM response text
            query: Original query

        Returns:
            CoverageResponse object
        """
        # Extract fields using regex
        status_match = re.search(r"STATUS:\s*(.*?)(?:\n|$)", llm_output, re.IGNORECASE)
        explanation_match = re.search(
            r"EXPLANATION:\s*(.*?)(?=REFERENCE:|CONFIDENCE:|$)",
            llm_output,
            re.IGNORECASE | re.DOTALL,
        )
        reference_match = re.search(
            r"REFERENCE:\s*(.*?)(?=CONFIDENCE:|$)", llm_output, re.IGNORECASE | re.DOTALL
        )
        confidence_match = re.search(r"CONFIDENCE:\s*(.*?)(?:\n|$)", llm_output, re.IGNORECASE)

        # Extract and clean values
        status_str = status_match.group(1).strip() if status_match else "Ambiguous"
        explanation = (
            explanation_match.group(1).strip()
            if explanation_match
            else "Unable to determine from policy."
        )
        reference = (
            reference_match.group(1).strip() if reference_match else "No specific reference found."
        )
        confidence_str = confidence_match.group(1).strip() if confidence_match else "Medium"

        # Normalize status
        status_lower = status_str.lower()
        if "covered" in status_lower and "not" not in status_lower:
            status = CoverageStatus.COVERED
        elif "not covered" in status_lower or "not" in status_lower:
            status = CoverageStatus.NOT_COVERED
        else:
            status = CoverageStatus.AMBIGUOUS

        # Normalize confidence
        conf_lower = confidence_str.lower()
        if "high" in conf_lower:
            confidence = ConfidenceLevel.HIGH
        elif "low" in conf_lower:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.MEDIUM

        return CoverageResponse(
            status=status,
            explanation=explanation.strip(),
            reference=reference.strip(),
            confidence=confidence,
            query=query,
        )
