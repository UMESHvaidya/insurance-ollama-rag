"""Document loading and processing utilities"""

from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

class PolicyDocumentLoader:
    """Loads and processes insurance policy PDF documents"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize document loader

        Args:
            chunk_size: Size of text chunks (optimized for smaller models)
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_pdf(self, pdf_path: str | Path) -> List[Document]:
        """
        Load PDF document

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of document pages
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File must be a PDF: {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def load_and_split(self, pdf_path: str | Path) -> tuple[List[Document], List[Document]]:
        """
        Load PDF and split into chunks

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (full pages, chunks)
        """
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)

        return documents, chunks
