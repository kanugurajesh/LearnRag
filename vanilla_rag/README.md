# Vanilla RAG (Retrieval-Augmented Generation)

A simple and straightforward implementation of a Retrieval-Augmented Generation (RAG) system using Ollama, Qdrant, and Unstructured for document processing.

## Features

- **Document Processing**: Extracts text from various file formats (PDF, DOCX, HTML, TXT, MD, etc.) using Unstructured
- **Vector Storage**: Stores document embeddings in Qdrant vector database
- **Semantic Search**: Retrieves relevant documents using cosine similarity
- **Response Generation**: Uses Llama 3.2 model via Ollama for generating contextual responses
- **Flexible Querying**: Command-line interface for both document processing and querying

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Qdrant vector database running on localhost:6333
- Required Python packages (see installation section)

## Installation

1. Install required Python packages:
```bash
pip install ollama qdrant-client unstructured
```

2. Pull required Ollama models:
```bash
ollama pull mxbai-embed-large:latest
ollama pull llama3.2:latest
```

3. Start Qdrant (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### Document Processing Mode

Process and index documents from a directory:

```bash
python app.py
```

This will:
- Extract text from all supported files in the `../public` directory
- Generate embeddings using the `mxbai-embed-large` model
- Store embeddings in Qdrant vector database
- Save extracted data to `extracted_data.json` and `rag_chunks.json`

### Query Mode

Query the indexed documents:

```bash
python app.py query "your question here"
```

Example:
```bash
python app.py query "What is the main topic of the documents?"
```

## Supported File Formats

- PDF (.pdf)
- Microsoft Word (.docx, .doc)
- PowerPoint (.pptx, .ppt)
- Excel (.xlsx, .xls)
- HTML (.html, .htm)
- Text files (.txt, .md)
- RTF (.rtf)
- OpenDocument (.odt)
- Email (.msg, .eml)
- CSV/TSV (.csv, .tsv)
- JSON (.json)
- XML (.xml)
- EPUB (.epub)

## Configuration

### Models
- **Embedding Model**: `mxbai-embed-large:latest` (1024 dimensions)
- **Chat Model**: `llama3.2:latest`

### Vector Database
- **Collection Name**: "documents"
- **Vector Size**: 1024
- **Distance Metric**: Cosine similarity
- **Default Query Limit**: 5 documents

### Text Processing
- **Minimum Chunk Length**: 50 characters
- **Batch Processing**: 10 chunks at a time for progress tracking

## Project Structure

```
vanilla_rag/
├── app.py              # Main application file
├── README.md           # This file
├── extracted_data.json # Raw extracted document data
└── rag_chunks.json     # Processed text chunks for RAG
```

## How It Works

1. **Document Extraction**: Uses Unstructured library to parse various document formats and extract structured text elements
2. **Text Chunking**: Filters and processes extracted text into meaningful chunks (minimum 50 characters)
3. **Embedding Generation**: Creates vector embeddings for each text chunk using Ollama's mxbai-embed-large model
4. **Vector Storage**: Stores embeddings with metadata in Qdrant vector database
5. **Query Processing**: For queries, generates embedding and performs similarity search to retrieve relevant documents
6. **Response Generation**: Uses retrieved context to generate informed responses via Llama 3.2

## Error Handling

The application includes comprehensive error handling for:
- File processing failures
- Embedding generation errors
- Vector database connection issues
- Model availability problems

Failed operations are logged with detailed error messages for debugging.

## Customization

You can modify the following parameters in `app.py`:
- `COLLECTION_NAME`: Change the Qdrant collection name
- `limit` parameter in `rag_query()`: Adjust number of retrieved documents
- `min_length` in `get_all_text_chunks()`: Change minimum chunk size
- Directory path in `extract_data_from_directory()`: Change source directory

## Troubleshooting

1. **Ollama models not found**: Ensure models are pulled using `ollama pull`
2. **Qdrant connection error**: Verify Qdrant is running on localhost:6333
3. **File processing failures**: Check file permissions and formats
4. **Empty results**: Ensure documents are processed and stored in Qdrant before querying