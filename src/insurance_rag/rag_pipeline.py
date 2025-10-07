"""Main RAG pipeline orchestration for local Ollama"""

from pathlib import Path
from typing import Optional

from .config import get_settings
from .models import CoverageResponse
from .document_loader import PolicyDocumentLoader
from .vectorstore import VectorStoreManager
from .retriever import DocumentRetriever
from .llm_analyzer import CoverageAnalyzer

class InsurancePolicyRAG:
    """Main RAG system for insurance policy analysis with local Ollama"""

    def __init__(self):
        """Initialize RAG pipeline with local models"""
        # Load settings
        self.settings = get_settings()

        print("🚀 Initializing Insurance Policy RAG with Local Ollama...")
        print(f"   Model: {self.settings.ollama_model}")
        print(f"   Ollama URL: {self.settings.ollama_base_url}")

        # Initialize components
        self.document_loader = PolicyDocumentLoader(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )

        self.vectorstore_manager = VectorStoreManager(
            embedding_model=self.settings.embedding_model
        )

        self.analyzer = CoverageAnalyzer(
            model_name=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.llm_temperature
        )

        self.retriever: Optional[DocumentRetriever] = None
        self.document_loaded = False

    def load_policy(self, pdf_path: str | Path) -> None:
        """
        Load and index insurance policy document

        Args:
            pdf_path: Path to policy PDF
        """
        print(f"\n📄 Loading policy document: {pdf_path}")

        # Load and split document
        pages, chunks = self.document_loader.load_and_split(pdf_path)
        print(f"✓ Loaded {len(pages)} pages")
        print(f"✓ Created {len(chunks)} text chunks")

        # Create vector store
        print("🔨 Building vector index (this may take a minute on first run)...")
        self.vectorstore_manager.create_vectorstore(chunks)

        # Initialize retriever
        vectorstore = self.vectorstore_manager.get_vectorstore()
        self.retriever = DocumentRetriever(
            vectorstore=vectorstore,
            retrieval_k=self.settings.retrieval_k
        )

        self.document_loaded = True
        print("✅ Policy document indexed successfully!\n")

    def query(self, query: str) -> CoverageResponse:
        """
        Query the insurance policy

        Args:
            query: Natural language question

        Returns:
            CoverageResponse object
        """
        if not self.document_loaded:
            raise ValueError("No policy loaded. Call load_policy() first.")

        print(f"🔍 Analyzing query: '{query}'")

        # Retrieve relevant documents
        documents = self.retriever.retrieve(query)

        # Format context with length limit for smaller models
        context = self.retriever.format_context(
            documents,
            max_length=self.settings.context_length
        )

        # Analyze coverage using Ollama
        print("🤖 Querying local Ollama model...")
        response = self.analyzer.analyze(query, context)

        return response
