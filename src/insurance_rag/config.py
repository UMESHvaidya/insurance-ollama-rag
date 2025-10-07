"""Configuration management for the RAG system"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="gemma2:2b", alias="OLLAMA_MODEL")

    # Embedding Model (using sentence-transformers for local embeddings)
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL"
    )

    # RAG Parameters
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")  # Smaller for gemma2:2b
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    retrieval_k: int = Field(default=4, alias="RETRIEVAL_K")  # Fewer chunks for smaller model

    # Temperature for LLM (0 = deterministic)
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    # Context window for Ollama model
    context_length: int = Field(default=2048, alias="CONTEXT_LENGTH")

    # Directories
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance

_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get or create settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
