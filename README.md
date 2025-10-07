# Insurance Policy RAG System - Poetry + Ollama (Local & Free)

# Complete project structure optimized for local Ollama with gemma2:2b

"""
PROJECT STRUCTURE:
==================

insurance-policy-rag/
├── pyproject.toml # Poetry configuration
├── poetry.lock # Locked dependencies
├── README.md # Project documentation
├── .env.example # Environment variables template
├── .gitignore # Git ignore file
├── data/ # Policy documents directory
│ └── .gitkeep
├── src/
│ └── insurance_rag/
│ ├── **init**.py # Package initialization
│ ├── config.py # Configuration management
│ ├── models.py # Data models
│ ├── document_loader.py # PDF loading and processing
│ ├── vectorstore.py # Vector database management
│ ├── retriever.py # Document retrieval logic
│ ├── llm_analyzer.py # LLM-based analysis
│ └── rag_pipeline.py # Main RAG orchestration
├── tests/ # Test directory
│ ├── **init**.py
│ └── test_rag.py
├── scripts/ # Utility scripts
│ └── interactive.py # Interactive CLI
└── main.py # Entry point

"""

# ==============================================================================

# FILE: pyproject.toml

# ==============================================================================

PYPROJECT_TOML = """
[tool.poetry]
name = "insurance-policy-rag"
version = "0.1.0"
description = "RAG system for insurance policy document analysis using local Ollama"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "insurance_rag", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10"
langchain = "^0.1.0"
langchain-community = "^0.0.20"
chromadb = "^0.4.22"
pypdf = "^4.0.0"
sentence-transformers = "^2.3.0"
python-dotenv = "^1.0.0"
rich = "^13.7.0"
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
ollama = "^0.1.6"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.12.0"
ruff = "^0.1.9"
mypy = "^1.8.0"
ipython = "^8.12.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
"""

# ==============================================================================

# FILE: src/insurance_rag/**init**.py

# ==============================================================================

INIT_PY = """
\"\"\"Insurance Policy RAG System with Local Ollama\"\"\"

from .rag_pipeline import InsurancePolicyRAG
from .models import CoverageResponse, CoverageStatus

**version** = "0.1.0"
**all** = ["InsurancePolicyRAG", "CoverageResponse", "CoverageStatus"]
"""

# ==============================================================================

# FILE: src/insurance_rag/config.py

# ==============================================================================

CONFIG_PY = """
\"\"\"Configuration management for the RAG system\"\"\"

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
\"\"\"Application settings loaded from environment variables\"\"\"

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

\_settings: Optional[Settings] = None

def get_settings() -> Settings:
\"\"\"Get or create settings instance\"\"\"
global \_settings
if \_settings is None:
\_settings = Settings()
return \_settings
"""

# ==============================================================================

# FILE: src/insurance_rag/models.py

# ==============================================================================

MODELS_PY = """
\"\"\"Data models for the RAG system\"\"\"

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class CoverageStatus(str, Enum):
\"\"\"Enumeration of possible coverage statuses\"\"\"
COVERED = "✅ Covered"
NOT_COVERED = "❌ Not Covered"
AMBIGUOUS = "⚠️ Ambiguous"

class ConfidenceLevel(str, Enum):
\"\"\"Confidence level of the analysis\"\"\"
HIGH = "High"
MEDIUM = "Medium"
LOW = "Low"

@dataclass
class CoverageResponse:
\"\"\"Structured response for insurance coverage queries\"\"\"
status: CoverageStatus
explanation: str
reference: str
confidence: ConfidenceLevel
query: str

    def __str__(self) -> str:
        \"\"\"Format response for display\"\"\"
        return f\"\"\"

{self.status.value}
{self.explanation}
Reference: {self.reference}
Confidence: {self.confidence.value}
\"\"\".strip()

    def to_dict(self) -> dict:
        \"\"\"Convert to dictionary\"\"\"
        return {
            "query": self.query,
            "status": self.status.value,
            "explanation": self.explanation,
            "reference": self.reference,
            "confidence": self.confidence.value,
        }

"""

# ==============================================================================

# FILE: src/insurance_rag/document_loader.py

# ==============================================================================

DOCUMENT_LOADER_PY = """
\"\"\"Document loading and processing utilities\"\"\"

from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

class PolicyDocumentLoader:
\"\"\"Loads and processes insurance policy PDF documents\"\"\"

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        \"\"\"
        Initialize document loader

        Args:
            chunk_size: Size of text chunks (optimized for smaller models)
            chunk_overlap: Overlap between chunks for context preservation
        \"\"\"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )

    def load_pdf(self, pdf_path: str | Path) -> List[Document]:
        \"\"\"
        Load PDF document

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of document pages
        \"\"\"
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File must be a PDF: {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        \"\"\"
        Split documents into chunks

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        \"\"\"
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def load_and_split(self, pdf_path: str | Path) -> tuple[List[Document], List[Document]]:
        \"\"\"
        Load PDF and split into chunks

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (full pages, chunks)
        \"\"\"
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)

        return documents, chunks

"""

# ==============================================================================

# FILE: src/insurance_rag/vectorstore.py

# ==============================================================================

VECTORSTORE_PY = """
\"\"\"Vector store management for document embeddings using local models\"\"\"

from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
\"\"\"Manages vector database for document embeddings using local embeddings\"\"\"

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        \"\"\"
        Initialize vector store manager with local embeddings

        Args:
            embedding_model: Name of HuggingFace embedding model to use
        \"\"\"
        print(f"📦 Loading local embedding model: {embedding_model}")

        # Use local HuggingFace embeddings (free and offline)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vectorstore: Optional[Chroma] = None
        print("✓ Embedding model loaded")

    def create_vectorstore(self, chunks: List[Document], collection_name: str = "insurance_policy") -> None:
        \"\"\"
        Create vector store from document chunks

        Args:
            chunks: List of document chunks
            collection_name: Name for the collection
        \"\"\"
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name
        )

    def get_vectorstore(self) -> Chroma:
        \"\"\"Get the vector store instance\"\"\"
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore() first.")
        return self.vectorstore

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        \"\"\"
        Search for similar documents

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        \"\"\"
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        return self.vectorstore.similarity_search(query, k=k)

"""

# ==============================================================================

# FILE: src/insurance_rag/retriever.py

# ==============================================================================

RETRIEVER_PY = """
\"\"\"Document retrieval logic\"\"\"

from typing import List
from langchain.schema import Document

class DocumentRetriever:
\"\"\"Retrieves relevant documents for queries\"\"\"

    def __init__(self, vectorstore, retrieval_k: int = 4):
        \"\"\"
        Initialize retriever

        Args:
            vectorstore: Vector store instance
            retrieval_k: Number of documents to retrieve
        \"\"\"
        self.vectorstore = vectorstore
        self.retrieval_k = retrieval_k

        # Create base retriever
        self.base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k}
        )

    def retrieve(self, query: str) -> List[Document]:
        \"\"\"
        Retrieve relevant documents for a query

        Args:
            query: Search query

        Returns:
            List of relevant documents
        \"\"\"
        documents = self.base_retriever.get_relevant_documents(query)
        return documents

    def format_context(self, documents: List[Document], max_length: int = 2000) -> str:
        \"\"\"
        Format retrieved documents into context string with length limit

        Args:
            documents: List of documents
            max_length: Maximum context length for smaller models

        Returns:
            Formatted context string
        \"\"\"
        context_parts = []
        total_length = 0

        for doc in documents:
            page_num = doc.metadata.get('page', 'unknown')
            content = doc.page_content

            # Build context part
            part = f"[Page {page_num}]\\n{content}"
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

        return "\\n\\n".join(context_parts)

"""

# ==============================================================================

# FILE: src/insurance_rag/llm_analyzer.py

# ==============================================================================

LLM_ANALYZER_PY = """
\"\"\"LLM-based coverage analysis using local Ollama\"\"\"

import re
from typing import Optional
import ollama

from .models import CoverageResponse, CoverageStatus, ConfidenceLevel

class CoverageAnalyzer:
\"\"\"Analyzes insurance coverage using local Ollama LLM\"\"\"

    PROMPT_TEMPLATE = \"\"\"You are an insurance policy analyst. Analyze the policy excerpts and determine coverage.

POLICY EXCERPTS:
{context}

USER QUERY: {question}

TASK: Determine if the item is Covered, Not Covered, or Ambiguous.

Respond in this EXACT format:
STATUS: [Covered/Not Covered/Ambiguous]
EXPLANATION: [2-3 sentences explaining your determination]
REFERENCE: [Cite page numbers and sections if found]
CONFIDENCE: [High/Medium/Low]

Be precise and cite specific policy language when available.\"\"\"

    def __init__(self, model_name: str = "gemma2:2b", base_url: str = "http://localhost:11434",
                 temperature: float = 0.1):
        \"\"\"
        Initialize coverage analyzer with Ollama

        Args:
            model_name: Ollama model name (e.g., gemma2:2b, llama2, mistral)
            base_url: Ollama API base URL
            temperature: Temperature for generation
        \"\"\"
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

        # Verify Ollama connection
        try:
            ollama.list()
            print(f"✓ Connected to Ollama at {base_url}")
        except KeyboardInterrupt:
            console.print("\\n\\n[cyan]👋 Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]\\n")

    return 0

if **name** == "**main**":
sys.exit(main())
"""

# ==============================================================================

# FILE: README.md

# ==============================================================================

README_MD = """

# 🏥 Insurance Policy RAG System

A **100% Free & Local** Retrieval-Augmented Generation (RAG) system for analyzing insurance policy documents using **Ollama** and local embeddings.

## ✨ Features

- 🆓 **Completely Free**: No API keys or cloud services required
- 🔒 **Privacy First**: All processing happens locally on your machine
- 📄 **PDF Processing**: Intelligent document loading with context preservation
- 🔍 **Smart Retrieval**: Vector-based similarity search with ChromaDB
- 🤖 **Local LLM**: Powered by Ollama (gemma2:2b, llama2, mistral, etc.)
- 🎨 **Beautiful CLI**: Rich terminal interface with colors and spinners
- ⚙️ **Configurable**: Easy configuration via .env file

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Poetry** - Python dependency manager
3. **Ollama** - Local LLM runtime

### Step 1: Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download

# Start Ollama service
ollama serve

# Pull the model (in a new terminal)
ollama pull gemma2:2b  # Lightweight and fast (2GB)
# or
ollama pull llama2     # More capable (4GB)
# or
ollama pull mistral    # Good balance (4GB)
```

### Step 2: Install Poetry

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Verify installation
poetry --version
```

### Step 3: Setup Project

```bash
# Clone/create project directory
mkdir insurance-policy-rag
cd insurance-policy-rag

# Copy all the code files (see structure below)

# Install dependencies
poetry install

# Create .env file
cp .env.example .env

# Edit .env if needed (optional, defaults work fine)
nano .env
```

### Step 4: Add Your Policy Document

```bash
# Create data directory
mkdir -p data

# Copy your insurance policy PDF
cp /path/to/your/policy.pdf data/insurance_policy.pdf
```

### Step 5: Run!

```bash
# Run example queries
poetry run python main.py

# Or use interactive mode
poetry run python scripts/interactive.py
```

## 📁 Project Structure

```
insurance-policy-rag/
├── pyproject.toml              # Poetry configuration
├── poetry.lock                 # Locked dependencies
├── README.md                   # This file
├── .env.example               # Environment template
├── .gitignore                 # Git ignore
├── data/                      # Policy documents
│   └── insurance_policy.pdf
├── src/insurance_rag/         # Main package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Settings (Ollama config)
│   ├── models.py             # Data models
│   ├── document_loader.py    # PDF processing
│   ├── vectorstore.py        # Vector DB (local embeddings)
│   ├── retriever.py          # Document retrieval
│   ├── llm_analyzer.py       # Ollama integration
│   └── rag_pipeline.py       # Main orchestrator
├── scripts/
│   └── interactive.py        # Interactive CLI
├── tests/
│   └── test_rag.py          # Tests
└── main.py                   # Entry point
```

## ⚙️ Configuration

Configure via `.env` file:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b

# Embedding Model (HuggingFace, runs locally)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG Parameters (optimized for smaller models)
CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVAL_K=4
LLM_TEMPERATURE=0.1
CONTEXT_LENGTH=2048

# Directories
DATA_DIR=data
```

## 🎯 Model Recommendations

| Model         | Size | RAM Needed | Speed  | Quality    | Best For                     |
| ------------- | ---- | ---------- | ------ | ---------- | ---------------------------- |
| **gemma2:2b** | 2GB  | 4GB        | ⚡⚡⚡ | ⭐⭐⭐     | Fast queries, basic analysis |
| **llama2**    | 4GB  | 8GB        | ⚡⚡   | ⭐⭐⭐⭐   | Better understanding         |
| **mistral**   | 4GB  | 8GB        | ⚡⚡   | ⭐⭐⭐⭐   | Balanced performance         |
| **llama3:8b** | 8GB  | 16GB       | ⚡     | ⭐⭐⭐⭐⭐ | Best quality                 |

Switch models anytime:

```bash
# Pull new model
ollama pull llama2

# Update .env
OLLAMA_MODEL=llama2

# Restart the app
poetry run python main.py
```

## 📊 Example Output

```
❓ QUERY: Is cataract surgery covered?
──────────────────────────────────────────────────────────────────────
✅ Covered
Cataract surgery is explicitly listed as a covered procedure under
the Day Care Procedures section of the policy document.
Reference: Page 6, Section 3.2 - Day Care Procedures
Confidence: High
```

## 🎯 Coverage Status Types

- **✅ Covered**: Explicitly mentioned as covered
- **❌ Not Covered**: Explicitly excluded or not mentioned
- **⚠️ Ambiguous**: Unclear, conditional, or needs clarification

## 💻 Usage Examples

### Programmatic Usage

```python
from insurance_rag import InsurancePolicyRAG

# Initialize
rag = InsurancePolicyRAG()

# Load policy
rag.load_policy("data/insurance_policy.pdf")

# Query
response = rag.query("Is dental treatment covered?")
print(response)

# Access response fields
print(f"Status: {response.status}")
print(f"Explanation: {response.explanation}")
print(f"Reference: {response.reference}")
print(f"Confidence: {response.confidence}")

# Convert to dict
data = response.to_dict()
```

### Interactive Mode Features

- 🎨 Colored output based on coverage status
- ⚡ Loading spinners for better UX
- 💬 Natural conversation flow
- 🔄 Keep asking multiple questions
- 📋 Clear, formatted responses

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=insurance_rag

# Verbose output
poetry run pytest -v
```

## 🛠️ Development

```bash
# Format code
poetry run black src/ scripts/ tests/

# Lint
poetry run ruff check src/ scripts/ tests/

# Type checking
poetry run mypy src/

# Add new dependency
poetry add <package-name>

# Add dev dependency
poetry add --group dev <package-name>
```

## 🔧 Troubleshooting

### Issue: "Cannot connect to Ollama"

```bash
# Make sure Ollama is running
ollama serve

# Check if it's running
curl http://localhost:11434/api/tags
```

### Issue: "Model not found"

```bash
# Pull the model
ollama pull gemma2:2b

# List available models
ollama list
```

### Issue: "Out of memory"

```bash
# Use a smaller model
ollama pull gemma2:2b  # Only 2GB

# Or reduce context length in .env
CONTEXT_LENGTH=1024
CHUNK_SIZE=500
```

### Issue: "Slow performance"

```bash
# Use smaller chunks
CHUNK_SIZE=600
RETRIEVAL_K=3

# Or use a smaller/faster model
OLLAMA_MODEL=gemma2:2b
```

### Issue: "Poetry command not found"

```bash
# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or use full path
~/.local/bin/poetry install
```

## 🚀 Performance Tips

1. **GPU Acceleration**: If you have NVIDIA GPU:

   ```python
   # In vectorstore.py, change:
   model_kwargs={'device': 'cuda'}  # Instead of 'cpu'
   ```

2. **Faster Embeddings**: Use smaller embedding model:

   ```bash
   EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast, 80MB
   ```

3. **Persistent Vector Store**: Cache embeddings:
   ```python
   # Modify vectorstore.py to persist to disk
   persist_directory="./chroma_db"
   ```

## 📝 Customization Examples

### Use Different PDF Location

```python
rag = InsurancePolicyRAG()
rag.load_policy("/custom/path/policy.pdf")
```

### Analyze Multiple Policies

```python
rag = InsurancePolicyRAG()

# Policy 1
rag.load_policy("data/health_policy.pdf")
response1 = rag.query("Is surgery covered?")

# Policy 2
rag.load_policy("data/life_policy.pdf")
response2 = rag.query("What is the coverage amount?")
```

### Custom Prompts

Edit `src/insurance_rag/llm_analyzer.py` to modify the prompt template.

## 🌟 Why Local LLMs?

### Advantages

- ✅ **No API Costs**: Completely free forever
- ✅ **Privacy**: Your data never leaves your machine
- ✅ **No Rate Limits**: Query as much as you want
- ✅ **Offline**: Works without internet (after initial model download)
- ✅ **Customizable**: Full control over models and prompts

### Considerations

- ⚠️ **Setup Required**: Need to install and run Ollama
- ⚠️ **Hardware**: Needs decent RAM (4-16GB depending on model)
- ⚠️ **Speed**: Slower than cloud APIs (but still fast enough)
- ⚠️ **Quality**: Smaller models less capable than GPT-4

## 🔄 Migration from OpenAI

If you have an existing OpenAI-based RAG system:

```python
# Old (OpenAI)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(api_key="sk-...", model="gpt-4")
embeddings = OpenAIEmbeddings(api_key="sk-...")

# New (Local)
import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

# Use ollama.generate() for LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

## 📚 Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Available Models](https://ollama.com/library)
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Guide](https://docs.trychroma.com/)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 💬 Support

- **Issues**: Open a GitHub issue
- **Questions**: Discussions tab
- **Updates**: Watch the repository

## 🎉 What's Next?

Possible enhancements:

- [ ] Web UI with Streamlit/Gradio
- [ ] Multi-document comparison
- [ ] Export reports to PDF
- [ ] Document question history
- [ ] Fine-tune models on insurance documents
- [ ] Add voice input/output
- [ ] Mobile app

---

Made with ❤️ using local AI. No clouds were harmed in the making of this project! ☁️🚫
"""

# ==============================================================================

# FILE: .env.example

# ==============================================================================

ENV_EXAMPLE = """

# Ollama Configuration

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b

# Embedding Model (HuggingFace - runs locally, no API key needed)

# Options: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2

EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG Parameters (optimized for smaller models)

CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVAL_K=4
LLM_TEMPERATURE=0.1
CONTEXT_LENGTH=2048

# Directories

DATA_DIR=data
"""

# ==============================================================================

# FILE: .gitignore

# ==============================================================================

GITIGNORE = """

# Python

**pycache**/
_.py[cod]
_$py.class
_.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
_.egg-info/
.installed.cfg
\*.egg

# Virtual Environment

venv/
env/
ENV/
.venv

# Poetry

poetry.lock

# IDEs

.vscode/
.idea/
_.swp
_.swo
\*~

# Environment variables

.env
.env.local

# Data files

data/_.pdf
data/_.txt
!data/.gitkeep

# Chroma DB

chroma*db/
*.db
\_.sqlite3

# HuggingFace cache

.cache/
transformers_cache/

# Testing

.pytest_cache/
.coverage
htmlcov/

# OS

.DS_Store
Thumbs.db
"""

# ==============================================================================

# FILE: tests/test_rag.py

# ==============================================================================

TEST_RAG_PY = """
\"\"\"Tests for the RAG system\"\"\"

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from insurance_rag.models import CoverageStatus, CoverageResponse, ConfidenceLevel
from insurance_rag.document_loader import PolicyDocumentLoader
from insurance_rag.llm_analyzer import CoverageAnalyzer

class TestPolicyDocumentLoader:
\"\"\"Tests for document loader\"\"\"

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
\"\"\"Tests for coverage analyzer\"\"\"

    @patch('insurance_rag.llm_analyzer.ollama')
    def test_parse_covered_response(self, mock_ollama):
        # Mock ollama.list() for connection check
        mock_ollama.list.return_value = []

        analyzer = CoverageAnalyzer(model_name="gemma2:2b")

        llm_output = \"\"\"

STATUS: Covered
EXPLANATION: The procedure is explicitly listed in the policy.
REFERENCE: Page 5, Section 2.1
CONFIDENCE: High
\"\"\"

        result = analyzer._parse_response(llm_output, "Test query")

        assert result.status == CoverageStatus.COVERED
        assert result.confidence == ConfidenceLevel.HIGH
        assert "explicitly listed" in result.explanation

    @patch('insurance_rag.llm_analyzer.ollama')
    def test_parse_not_covered_response(self, mock_ollama):
        mock_ollama.list.return_value = []

        analyzer = CoverageAnalyzer(model_name="gemma2:2b")

        llm_output = \"\"\"

STATUS: Not Covered
EXPLANATION: This is explicitly excluded from coverage.
REFERENCE: Page 10, Exclusions section
CONFIDENCE: High
\"\"\"

        result = analyzer._parse_response(llm_output, "Test query")

        assert result.status == CoverageStatus.NOT_COVERED
        assert result.confidence == ConfidenceLevel.HIGH

    @patch('insurance_rag.llm_analyzer.ollama')
    def test_connection_error(self, mock_ollama):
        # Simulate Ollama not running
        mock_ollama.list.side_effect = Exception("Connection refused")

        with pytest.raises(ConnectionError):
            CoverageAnalyzer(model_name="gemma2:2b")

class TestCoverageResponse:
\"\"\"Tests for coverage response model\"\"\"

    def test_to_dict(self):
        response = CoverageResponse(
            status=CoverageStatus.COVERED,
            explanation="Test explanation",
            reference="Page 1",
            confidence=ConfidenceLevel.HIGH,
            query="Test query"
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
            query="Test query"
        )

        output = str(response)

        assert "⚠️ Ambiguous" in output
        assert "Unclear from policy" in output
        assert "Multiple sections" in output

class TestConfig:
\"\"\"Test configuration\"\"\"

    def test_default_settings(self):
        from insurance_rag.config import Settings

        settings = Settings(OLLAMA_BASE_URL="http://localhost:11434")

        assert settings.ollama_model == "gemma2:2b"
        assert settings.chunk_size == 800
        assert settings.retrieval_k == 4

"""

# ==============================================================================

# COMPLETE SETUP SCRIPT (setup.sh)

# ==============================================================================

SETUP_SCRIPT = """#!/bin/bash

# Complete setup script for Insurance Policy RAG with Ollama

set -e

echo "🚀 Setting up Insurance Policy RAG System"
echo "=========================================="
echo ""

# Check if Ollama is installed

if ! command -v ollama &> /dev/null; then
echo "⚠️ Ollama not found. Please install it first:"
echo " Visit: https://ollama.com/download"
exit 1
fi

# Check if Poetry is installed

if ! command -v poetry &> /dev/null; then
echo "📦 Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
fi

echo "✓ Poetry installed"

# Install dependencies

echo ""
echo "📦 Installing Python dependencies..."
poetry install

echo "✓ Dependencies installed"

# Create data directory

echo ""
echo "📁 Creating data directory..."
mkdir -p data
touch data/.gitkeep

echo "✓ Data directory created"

# Create .env file

if [ ! -f .env ]; then
echo ""
echo "⚙️ Creating .env file..."
cp .env.example .env
echo "✓ .env file created"
else
echo ""
echo "✓ .env file already exists"
fi

# Check if Ollama is running

echo ""
echo "🔍 Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
echo "✓ Ollama is running"
else
echo "⚠️ Ollama is not running. Starting it..."
echo " Run in another terminal: ollama serve"
echo ""
echo " Waiting for Ollama to start..."
sleep 5
fi

# Pull model

echo ""
echo "🤖 Pulling Ollama model (gemma2:2b)..."
echo " This may take a few minutes..."
ollama pull gemma2:2b

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo " 1. Place your insurance policy PDF in: data/insurance_policy.pdf"
echo " 2. Run: poetry run python main.py"
echo " 3. Or interactive mode: poetry run python scripts/interactive.py"
echo ""
echo "🎉 Happy querying!"
"""

# print("""

# ✅ COMPLETE PROJECT CODE GENERATED - POETRY + OLLAMA VERSION

All files have been updated for local Ollama with gemma2:2b!

🎯 KEY CHANGES MADE:
✓ Switched from OpenAI to Ollama (100% free & local)
✓ Changed from UV to Poetry package manager
✓ Using HuggingFace embeddings (local, no API needed)
✓ Optimized chunk sizes for smaller models
✓ Added Ollama connection checks
✓ Updated all configuration for local setup

📦 WHAT YOU GET:
• Complete Poetry project structure
• Local Ollama integration (gemma2:2b)
• Free HuggingFace embeddings
• Interactive CLI with Rich UI
• Full test suite
• Comprehensive documentation

🚀 QUICK START:
""")

print(SETUP_SCRIPT)

print("""

Save this as 'setup.sh', make it executable, and run:
chmod +x setup.sh
./setup.sh

Or follow manual setup:

1. poetry install
2. cp .env.example .env
3. ollama serve # In another terminal
4. ollama pull gemma2:2b
5. cp your_policy.pdf data/insurance_policy.pdf
6. poetry run python main.py

================================================================================
""") Exception as e:
raise ConnectionError(
f"Cannot connect to Ollama at {base_url}. "
f"Make sure Ollama is running: 'ollama serve'. Error: {e}"
)

    def analyze(self, query: str, context: str) -> CoverageResponse:
        \"\"\"
        Analyze coverage for a query using Ollama

        Args:
            query: User query
            context: Retrieved policy context

        Returns:
            CoverageResponse object
        \"\"\"
        # Build prompt
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=query)

        try:
            # Call Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": 300,  # Limit response length
                }
            )

            result_text = response['response']

        except Exception as e:
            # Fallback response if Ollama fails
            print(f"⚠️ Ollama error: {e}")
            result_text = \"\"\"STATUS: Ambiguous

EXPLANATION: Unable to analyze due to system error. Please try again.
REFERENCE: N/A
CONFIDENCE: Low\"\"\"

        # Parse response
        coverage = self._parse_response(result_text, query)

        return coverage

    def _parse_response(self, llm_output: str, query: str) -> CoverageResponse:
        \"\"\"
        Parse LLM output into structured CoverageResponse

        Args:
            llm_output: Raw LLM response text
            query: Original query

        Returns:
            CoverageResponse object
        \"\"\"
        # Extract fields using regex
        status_match = re.search(r'STATUS:\\s*(.*?)(?:\\n|$)', llm_output, re.IGNORECASE)
        explanation_match = re.search(
            r'EXPLANATION:\\s*(.*?)(?=REFERENCE:|CONFIDENCE:|$)',
            llm_output,
            re.IGNORECASE | re.DOTALL
        )
        reference_match = re.search(
            r'REFERENCE:\\s*(.*?)(?=CONFIDENCE:|$)',
            llm_output,
            re.IGNORECASE | re.DOTALL
        )
        confidence_match = re.search(r'CONFIDENCE:\\s*(.*?)(?:\\n|$)', llm_output, re.IGNORECASE)

        # Extract and clean values
        status_str = status_match.group(1).strip() if status_match else "Ambiguous"
        explanation = explanation_match.group(1).strip() if explanation_match else "Unable to determine from policy."
        reference = reference_match.group(1).strip() if reference_match else "No specific reference found."
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
            query=query
        )

"""

# ==============================================================================

# FILE: src/insurance_rag/rag_pipeline.py

# ==============================================================================

RAG_PIPELINE_PY = """
\"\"\"Main RAG pipeline orchestration for local Ollama\"\"\"

from pathlib import Path
from typing import Optional

from .config import get_settings
from .models import CoverageResponse
from .document_loader import PolicyDocumentLoader
from .vectorstore import VectorStoreManager
from .retriever import DocumentRetriever
from .llm_analyzer import CoverageAnalyzer

class InsurancePolicyRAG:
\"\"\"Main RAG system for insurance policy analysis with local Ollama\"\"\"

    def __init__(self):
        \"\"\"Initialize RAG pipeline with local models\"\"\"
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
        \"\"\"
        Load and index insurance policy document

        Args:
            pdf_path: Path to policy PDF
        \"\"\"
        print(f"\\n📄 Loading policy document: {pdf_path}")

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
        print("✅ Policy document indexed successfully!\\n")

    def query(self, query: str) -> CoverageResponse:
        \"\"\"
        Query the insurance policy

        Args:
            query: Natural language question

        Returns:
            CoverageResponse object
        \"\"\"
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

"""

# ==============================================================================

# FILE: main.py

# ==============================================================================

MAIN_PY = """
\"\"\"Main entry point for the Insurance Policy RAG system with Ollama\"\"\"

import sys
from pathlib import Path
from dotenv import load_dotenv

from insurance_rag import InsurancePolicyRAG
from insurance_rag.config import get_settings

def check_ollama():
\"\"\"Check if Ollama is running\"\"\"
import ollama
try:
ollama.list()
return True
except:
return False

def main():
\"\"\"Run example queries on a policy document\"\"\" # Load environment variables
load_dotenv()

    print("=" * 70)
    print("🏥 INSURANCE POLICY RAG SYSTEM (Local Ollama + Free)")
    print("=" * 70)
    print()

    # Check Ollama
    if not check_ollama():
        print("❌ Ollama is not running!")
        print()
        print("Please start Ollama:")
        print("  1. Install: https://ollama.com/download")
        print("  2. Start: ollama serve")
        print("  3. Pull model: ollama pull gemma2:2b")
        print()
        return 1

    # Get settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1

    # Initialize RAG system
    try:
        rag = InsurancePolicyRAG()
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print()
        print("Make sure Ollama model is available:")
        print(f"  ollama pull {settings.ollama_model}")
        return 1

    # Check for policy document
    pdf_path = settings.data_dir / "insurance_policy.pdf"

    if not pdf_path.exists():
        print(f"❌ Policy document not found: {pdf_path}")
        print(f"   Please place your insurance policy PDF in: {settings.data_dir}/")
        return 1

    # Load policy
    try:
        rag.load_policy(pdf_path)
    except Exception as e:
        print(f"❌ Error loading policy: {e}")
        return 1

    # Example queries
    queries = [
        "Is cataract surgery covered?",
        "What about dental treatment?",
        "Are pre-existing conditions covered?",
        "Is ambulance service included?",
    ]

    print("📋 Running example queries...")
    print("=" * 70)

    for query in queries:
        print(f"\\n❓ QUERY: {query}")
        print("-" * 70)

        try:
            response = rag.query(query)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")

        print()

    print("=" * 70)
    print("✅ Analysis complete!")
    print("   For interactive mode, run: poetry run python scripts/interactive.py")

    return 0

if **name** == "**main**":
sys.exit(main())
"""

# ==============================================================================

# FILE: scripts/interactive.py

# ==============================================================================

INTERACTIVE_PY = """
\"\"\"Interactive CLI for querying insurance policies with Ollama\"\"\"

import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
import ollama

from insurance_rag import InsurancePolicyRAG
from insurance_rag.config import get_settings

def check_ollama():
\"\"\"Check if Ollama is running\"\"\"
try:
ollama.list()
return True
except:
return False

def main():
\"\"\"Run interactive CLI\"\"\"
load_dotenv()
console = Console()

    console.print(Panel.fit(
        "🏥 [bold cyan]Insurance Policy RAG System[/bold cyan]\\n"
        "🤖 Powered by Local Ollama (Free & Private)\\n"
        "Interactive Query Interface",
        border_style="cyan"
    ))

    # Check Ollama
    if not check_ollama():
        console.print("\\n[red]❌ Ollama is not running![/red]")
        console.print("\\nPlease start Ollama:")
        console.print("  1. Install: https://ollama.com/download")
        console.print("  2. Start: [cyan]ollama serve[/cyan]")
        console.print("  3. Pull model: [cyan]ollama pull gemma2:2b[/cyan]")
        return 1

    # Initialize
    try:
        settings = get_settings()
        console.print(f"\\n🔧 Using model: [cyan]{settings.ollama_model}[/cyan]")
        rag = InsurancePolicyRAG()
    except Exception as e:
        console.print(f"[red]❌ Initialization error: {e}[/red]")
        console.print("\\nMake sure the model is available:")
        console.print(f"  [cyan]ollama pull {settings.ollama_model}[/cyan]")
        return 1

    # Load policy
    pdf_path = settings.data_dir / "insurance_policy.pdf"

    if not pdf_path.exists():
        console.print(f"\\n[red]❌ Policy not found: {pdf_path}[/red]")
        return 1

    try:
        rag.load_policy(pdf_path)
    except Exception as e:
        console.print(f"[red]❌ Error loading policy: {e}[/red]")
        return 1

    # Interactive loop
    console.print("\\n[bold green]Ready![/bold green] Ask questions about your policy")
    console.print("[dim](type 'quit', 'exit', or 'q' to exit)[/dim]\\n")

    while True:
        try:
            query = Prompt.ask("[bold yellow]Your question[/bold yellow]").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                console.print("\\n[cyan]👋 Goodbye![/cyan]")
                break

            if not query:
                continue

            # Query with spinner
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                progress.add_task(description="Analyzing with Ollama...", total=None)
                response = rag.query(query)

            # Color-code status
            status_color = {
                "✅ Covered": "green",
                "❌ Not Covered": "red",
                "⚠️ Ambiguous": "yellow"
            }.get(response.status.value, "white")

            console.print(f"\\n[bold {status_color}]{response.status.value}[/bold {status_color}]")
            console.print(response.explanation)
            console.print(f"[dim]Reference: {response.reference}[/dim]")
            console.print(f"[dim]Confidence: {response.confidence.value}[/dim]\\n")

        except
