from unittest.mock import MagicMock, patch
import pytest
from insurance_rag.llm_analyzer import CoverageAnalyzer

class TestHistory:
    @patch("insurance_rag.llm_analyzer.ollama")
    def test_analyzer_uses_history(self, mock_ollama):
        """Test that history is included in the prompt"""
        # Setup
        mock_ollama.generate.return_value = {
            "response": "STATUS: Covered\nEXPLANATION: Test.\nREFERENCE: Ref.\nCONFIDENCE: High"
        }
        
        analyzer = CoverageAnalyzer()
        
        # specific history to check for
        history = ["User: Q1", "Assistant: A1"]
        query = "Follow up Q"
        context = "Some context"
        
        # Action
        analyzer.analyze(query, context, history=history)
        
        # Verify
        call_args = mock_ollama.generate.call_args
        prompt_used = call_args[1]['prompt']
        
        assert "CHAT HISTORY:" in prompt_used
        assert "User: Q1" in prompt_used
        assert "Assistant: A1" in prompt_used
        assert "USER QUERY: Follow up Q" in prompt_used

    @patch("insurance_rag.llm_analyzer.ollama")
    def test_analyzer_no_history(self, mock_ollama):
        """Test prompt format when no history is provided"""
        mock_ollama.generate.return_value = {
            "response": "STATUS: Covered\nEXPLANATION: Test.\nREFERENCE: Ref.\nCONFIDENCE: High"
        }
        
        analyzer = CoverageAnalyzer()
        analyzer.analyze("Q", "Ctx")
        
        call_args = mock_ollama.generate.call_args
        prompt_used = call_args[1]['prompt']
        
        assert "CHAT HISTORY:\nNo previous history." in prompt_used
