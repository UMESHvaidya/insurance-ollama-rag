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