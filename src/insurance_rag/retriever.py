"""Document retrieval logic"""

from typing import List
from langchain.schema import Document


class DocumentRetriever:
    """Retrieves relevant documents for queries"""

    def __init__(self, vectorstore, retrieval_k: int = 4):
        """
        Initialize retriever

        Args:
            vectorstore: Vector store instance
            retrieval_k: Number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.retrieval_k = retrieval_k

        # Create base retriever
        self.base_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": retrieval_k}
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query

        Returns:
            List of relevant documents
        """
        documents = self.base_retriever.get_relevant_documents(query)
        return documents

    def format_context(self, documents: List[Document], max_length: int = 2000) -> str:
        """
        Format retrieved documents into context string with length limit

        Args:
            documents: List of documents
            max_length: Maximum context length for smaller models

        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0

        for doc in documents:
            page_num = doc.metadata.get("page", "unknown")
            content = doc.page_content

            # Build context part
            part = f"[Page {page_num}]\n{content}"
            part_length = len(part)

            # Check if adding this would exceed max length
            if total_length + part_length > max_length:
                # Add truncated version
                remaining = max_length - total_length
                if remaining > 100:  # Only add if reasonable amount left
                    context_parts.append(part[:remaining] + "...")
                break

            context_parts.append(part)
            total_length += part_length

        return "\n\n".join(context_parts)
