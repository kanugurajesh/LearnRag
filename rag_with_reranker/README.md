# RAG with Reranker

Enhanced Retrieval-Augmented Generation implementation using cross-encoder reranking for improved document relevance and answer quality.

## Overview

This implementation extends basic RAG by adding a reranking step that uses a cross-encoder model to better assess query-document relevance, leading to more accurate context selection and better answers.

## Architecture

```
Query → Vector Search (top 20) → Cross-Encoder Reranking (top 5) → LLM Generation
```

## Key Features

- **Two-stage retrieval**: Initial vector search + semantic reranking
- **Cross-encoder reranking**: Uses `ms-marco-MiniLM-L-6-v2` for precise relevance scoring
- **Configurable limits**: Adjustable initial retrieval and final reranking counts
- **Detailed logging**: Progress tracking for both retrieval stages

## Prerequisites

```bash
pip install ollama qdrant-client sentence-transformers unstructured pathlib
```

**Ollama models**:
```bash
ollama pull llama3.2:latest
ollama pull mxbai-embed-large:latest
```

**Qdrant database** (via Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### 1. Process Documents
Place your documents in `../public/` directory and run:
```bash
python app.py
```

This will:
- Extract text from all supported file formats
- Generate embeddings using `mxbai-embed-large`
- Store vectors in Qdrant collection `documents_reranked`
- Save processed chunks to `rag_chunks_reranked.json`

### 2. Query with Reranking
```bash
python app.py query "your question here"
```

## How Reranking Works

1. **Initial Retrieval**: Vector similarity search returns top 20 documents
2. **Reranking**: Cross-encoder model scores query-document pairs
3. **Selection**: Top 5 highest-scoring documents selected for context
4. **Generation**: LLM generates answer using reranked context

## Configuration

### Models
- **Embedding**: `mxbai-embed-large:latest` (1024d)
- **Chat**: `llama3.2:latest`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Parameters
```python
# Adjustable in rag_query_with_reranker()
initial_limit = 20    # Initial vector search results
final_limit = 5       # Final reranked documents
```

### Collection
- **Qdrant Collection**: `documents_reranked`
- **Vector Size**: 1024 dimensions
- **Distance Metric**: Cosine similarity

## Supported File Types

PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HTML, HTM, XML, TXT, MD, RTF, ODT, EPUB, MSG, EML, CSV, TSV, JSON

## Output Files

- `extracted_data_reranked.json`: Raw document extraction results
- `rag_chunks_reranked.json`: Processed text chunks ready for RAG

## Example Output

```
=== RAG Query Mode with Reranking ===
Question: What is machine learning?
==================================================
Searching for relevant documents for: What is machine learning?
Retrieved 20 documents from vector search
Reranked to top 5 documents

Answer:
Machine learning is a subset of artificial intelligence that enables 
computers to learn and make decisions from data without being explicitly 
programmed...
```

## Performance Benefits

Compared to vanilla RAG:
- **Better relevance**: Cross-encoder provides more accurate document scoring
- **Improved answers**: Higher quality context leads to better responses
- **Reduced noise**: Filters out less relevant documents from initial retrieval

## Code Structure

```python
get_embedding()              # Generate document embeddings
query_qdrant()              # Vector similarity search
rerank_documents()          # Cross-encoder reranking
rag_query_with_reranker()   # Complete RAG pipeline
```

## Troubleshooting

- **Model download issues**: Ensure internet connection for first-time model download
- **Memory errors**: Reduce `initial_limit` for large document collections
- **Qdrant connection**: Verify Qdrant is running on localhost:6333