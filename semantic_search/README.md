# MerkleDB Semantic Search + RAG

A production-ready semantic search and RAG (Retrieval Augmented Generation) system built on MerkleDB, demonstrating real-world vector database capabilities.

## Architecture

```
Documents (Markdown) → Chunking → Embeddings → MerkleDB
                                                    ↓
User Query → Embedding Model → KNN Search → Top-K Results
                                                    ↓
                                         [Optional] LLM → Answer
```

**Components:**
- **ingest.py**: Document ingestion pipeline - indexes markdown files into MerkleDB
- **search.py**: Semantic search CLI - find relevant documents using natural language
- **rag.py**: RAG system - answer questions using retrieved context + LLM

## Features

- **Fast Semantic Search**: All-MiniLM-L6-v2 model (384D embeddings)
- **Smart Chunking**: 500-char chunks with 50-char overlap, sentence-boundary aware
- **Batch Processing**: Efficient batch insertion (10 chunks at a time)
- **Interactive Modes**: CLI interfaces for search and Q&A
- **Multiple LLM Backends**: Ollama (local) or OpenAI
- **Production Ready**: Handles real documentation corpus (~2,900 lines)

## Installation

### Prerequisites

1. **MerkleDB running**:
```bash
cd C:\Users\baian\merkle_db
mix run --no-halt
```

2. **Python 3.8+** with pip

### Install Dependencies

```bash
cd semantic_search
pip install -r requirements.txt
```

This installs:
- `sentence-transformers` - Embedding model
- `requests` - HTTP client for MerkleDB API
- `numpy` - Numerical operations

### Optional: LLM Backend for RAG

**Option 1: Ollama (Recommended for local)**
```bash
# Install Ollama from https://ollama.ai
# Pull a model
ollama pull llama2
```

**Option 2: OpenAI**
```bash
pip install openai
export LLM_API_KEY="sk-..."
```

## Usage

### 1. Document Ingestion

Index your documentation into MerkleDB:

```bash
# Index all markdown files in merkle_db repository
python ingest.py

# Index a specific directory
python ingest.py C:\path\to\docs
```

**What it does:**
- Finds all `.md` and `.markdown` files
- Chunks documents (500 chars, 50 overlap)
- Generates embeddings using all-MiniLM-L6-v2
- Batch-inserts into MerkleDB collection `docs`
- Creates collection automatically if needed

**Example Output:**
```
🚀 Initializing Document Ingester
   MerkleDB: http://localhost:4000
   Collection: docs
📦 Loading embedding model: all-MiniLM-L6-v2
   Embedding dimension: 384
✅ Created collection 'docs' (dimension: 384)

📚 Finding documents in: C:\Users\baian\merkle_db
   Found 12 documents

🔄 Processing documents...
   📄 README.md
      → 8 chunks
      ✅ Inserted batch (10 chunks)
   📄 GETTING_STARTED.md
      → 12 chunks
...
✨ Ingestion complete!
   Total chunks indexed: 156
   Documents processed: 12
```

### 2. Semantic Search

Search indexed documents using natural language queries:

```bash
# Interactive mode
python search.py -i

# Single query
python search.py "how to insert vectors into MerkleDB"

# Top-K results (default: 5)
python search.py -k 10 "performance optimization"
```

**Interactive Mode:**
```
🔍 Interactive Semantic Search
   Type your query and press Enter
   Type 'quit' or 'exit' to stop

🔍 Search> how does replication work?
🔍 Query: "how does replication work?"
   Found 5 results

================================================================================

[1] Score: 0.8234
    Document: REPLICATION
    Chunk: #42
    ID: REPLICATION_chunk_42

[2] Score: 0.7891
    Document: ARCHITECTURE
    Chunk: #15
    ID: ARCHITECTURE_chunk_15
...
```

**Advanced Search Parameters:**
```bash
# In interactive mode, set k parameter
🔍 Search> k=10 vector similarity search
```

### 3. RAG (Question Answering)

Ask questions and get answers based on your documentation:

```bash
# Interactive mode with Ollama
python rag.py -i

# Single question
python rag.py "What are the key features of MerkleDB?"

# Specify LLM provider and model
python rag.py --provider ollama --model llama2 "How do I configure replication?"

# Use OpenAI
export LLM_API_KEY="sk-..."
python rag.py --provider openai --model gpt-4 "Explain the architecture"

# Retrieve more context documents
python rag.py -k 5 "performance benchmarks"
```

**Interactive Mode:**
```
🤖 RAG System
   LLM Provider: ollama
   LLM Model: llama2

🤖 Interactive RAG Q&A System
   Ask questions about MerkleDB documentation
   Type 'quit' or 'exit' to stop

❓ Question> What is MerkleDB?

🔍 Retrieving relevant documents...
   Found 3 relevant documents

📚 Retrieved Context:
--------------------------------------------------------------------------------
  [1] README_chunk_0 (score: 0.8945)
  [2] ARCHITECTURE_chunk_3 (score: 0.8234)
  [3] GETTING_STARTED_chunk_1 (score: 0.7891)
--------------------------------------------------------------------------------

🤖 Generating answer...

💡 Answer:
================================================================================
MerkleDB is a high-performance vector database built in Elixir, designed
for semantic search and similarity matching. It features SIMD-accelerated
operations using AVX2/AVX-512, distributed replication with Raft consensus,
and a REST API for easy integration...
================================================================================
```

## Configuration

Environment variables:

```bash
# MerkleDB Connection
export MERKLEDB_URL="http://localhost:4000"
export MERKLEDB_COLLECTION="docs"

# LLM Configuration (for RAG)
export LLM_PROVIDER="ollama"           # ollama, openai
export LLM_MODEL="llama2"              # Model name
export LLM_API_KEY="sk-..."            # For OpenAI
```

## Performance Expectations

**Ingestion:**
- ~2,900 lines of documentation
- ~150-200 chunks (500 chars each)
- Time: ~30-60 seconds (includes model loading)
- Memory: ~500MB (sentence-transformer model)

**Search:**
- First query: ~2-3 seconds (model loading)
- Subsequent queries: ~100-300ms
- MerkleDB KNN search: <50ms
- Embedding generation: 50-250ms

**RAG:**
- Retrieval: Same as search (~100-300ms)
- LLM generation:
  - Ollama (local): 2-10 seconds
  - OpenAI API: 1-3 seconds

## Architecture Details

### Embedding Model

**all-MiniLM-L6-v2**
- Dimensions: 384
- Max sequence length: 256 tokens
- Performance: Fast (CPU-friendly)
- Quality: Good for most use cases
- Size: ~80MB

### Chunking Strategy

- **Chunk size**: 500 characters
- **Overlap**: 50 characters
- **Boundary aware**: Splits on sentence boundaries when possible
- **Minimum chunk**: 20 characters (filters noise)

**Why this works:**
- 500 chars ≈ 1-2 paragraphs (good semantic unit)
- 50-char overlap prevents information loss at boundaries
- Sentence-aware splitting maintains coherence

### MerkleDB Integration

**Collection Schema:**
```json
{
  "dimension": 384,
  "precision": "f32"
}
```

**Vector Format:**
```json
{
  "id": "README_chunk_42",
  "vector": [0.123, -0.456, ...],  // 384 floats
  "metadata": {
    "source": "C:\\Users\\baian\\merkle_db\\README.md",
    "filename": "README.md",
    "title": "MerkleDB",
    "type": "documentation",
    "chunk_index": 42,
    "chunk_text": "First 200 chars of chunk..."
  }
}
```

**Search Request:**
```json
{
  "vector": [0.123, -0.456, ...],
  "k": 5
}
```

**Search Response:**
```json
{
  "results": [
    {
      "id": "README_chunk_42",
      "score": 0.8945,
      "metadata": {...}
    },
    ...
  ]
}
```

## Troubleshooting

### MerkleDB not responding
```bash
# Check if MerkleDB is running
curl http://localhost:4000/health

# Start MerkleDB
cd C:\Users\baian\merkle_db
mix run --no-halt
```

### Model download fails
```bash
# Sentence transformers downloads models on first use
# Ensure internet connection and sufficient disk space (~100MB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Ollama connection error
```bash
# Ensure Ollama is running
ollama list

# Pull model if missing
ollama pull llama2
```

### Search returns no results
```bash
# Verify collection has data
curl http://localhost:4000/v1/collections

# Re-run ingestion
python ingest.py
```

## Examples

### Example 1: Search Performance Benchmarking

```bash
# Index documentation
python ingest.py

# Search with timing
time python search.py "vector database performance"
```

### Example 2: Custom Document Collection

```bash
# Set custom collection name
export MERKLEDB_COLLECTION="technical_docs"

# Index specific directory
python ingest.py C:\path\to\technical\docs

# Search the custom collection
python search.py -i
```

### Example 3: RAG with GPT-4

```bash
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-4"
export LLM_API_KEY="sk-..."

python rag.py "Explain MerkleDB's replication architecture in detail"
```

## Extending the System

### Add New Document Types

Edit `ingest.py`:

```python
def find_documents(self, root_dir: Path) -> List[Path]:
    patterns = ["*.md", "*.markdown", "*.txt", "*.rst"]  # Add patterns
    ...
```

### Customize Chunking

Adjust parameters in `ingest.py`:

```python
CHUNK_SIZE = 1000  # Larger chunks for more context
CHUNK_OVERLAP = 100  # More overlap for better coverage
```

### Use Different Embedding Model

Change model in `ingest.py` and `search.py`:

```python
MODEL_NAME = "paraphrase-MiniLM-L6-v2"  # Alternative model
# Or larger model:
MODEL_NAME = "all-mpnet-base-v2"  # 768 dimensions, higher quality
```

**Note**: If changing dimensions, recreate MerkleDB collection.

## Production Considerations

1. **Scaling**: For large corpora (>100K documents), consider:
   - Batch size tuning
   - Parallel embedding generation
   - MerkleDB cluster deployment

2. **Model Selection**:
   - **all-MiniLM-L6-v2**: Fast, good for <100K docs
   - **all-mpnet-base-v2**: Higher quality, 768D
   - **text-embedding-ada-002**: OpenAI, highest quality, API-based

3. **Error Handling**: All scripts include basic error handling, but add:
   - Retry logic for MerkleDB API calls
   - Input validation for untrusted queries
   - Rate limiting for public deployments

4. **Monitoring**:
   - Track search latency percentiles
   - Monitor MerkleDB memory usage
   - Log slow queries for optimization

## License

Same as MerkleDB - see repository root.

## Credits

Built on:
- MerkleDB - High-performance vector database
- Sentence Transformers - State-of-the-art embeddings
- Ollama / OpenAI - LLM backends for RAG
