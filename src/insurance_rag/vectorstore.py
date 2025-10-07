"""Vector store management for document embeddings using local models"""

from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreManager:
    """Manages vector database for document embeddings using local embeddings"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store manager with local embeddings

        Args:
            embedding_model: Name of HuggingFace embedding model to use
        """
        print(f"📦 Loading local embedding model: {embedding_model}")

        # Use local HuggingFace embeddings (free and offline)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},  # Use 'cuda' if GPU available
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore: Optional[Chroma] = None
        print("✓ Embedding model loaded")

    def create_vectorstore(
        self, chunks: List[Document], collection_name: str = "insurance_policy"
    ) -> None:
        """
        Create vector store from document chunks

        Args:
            chunks: List of document chunks
            collection_name: Name for the collection
        """
        self.vectorstore = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings, collection_name=collection_name
        )

    def get_vectorstore(self) -> Chroma:
        """Get the vector store instance"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore() first.")
        return self.vectorstore

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        return self.vectorstore.similarity_search(query, k=k)
