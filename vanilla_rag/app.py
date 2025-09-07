from ollama import embed, chat
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

def get_embedding(text):
    response = embed(model="mxbai-embed-large:latest", input=text)
    return response['embeddings'][0]

def get_chat_response(messages):
    response = chat(model="llama3.2:latest", messages=messages)
    return response.message.content

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

COLLECTION_NAME = "documents"

def create_collection():
    """Create a collection in Qdrant for storing document embeddings"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            # Create collection with vector size 1024 (mxbai-embed-large dimension)
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            print(f"Created collection: {COLLECTION_NAME}")
        else:
            print(f"Collection {COLLECTION_NAME} already exists")
    except Exception as e:
        print(f"Error creating collection: {e}")

def store_embeddings_in_qdrant(chunks):
    """Store text chunks with their embeddings in Qdrant"""
    print(f"Storing {len(chunks)} chunks in Qdrant...")
    
    points = []
    for i, chunk in enumerate(chunks):
        try:
            # Get embedding for the text
            embedding = get_embedding(chunk["text"])
            
            # Create a point for Qdrant
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "element_type": chunk["element_type"],
                    "metadata": chunk["metadata"]
                }
            )
            points.append(point)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks...")
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
    
    # Upload points to Qdrant
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Successfully stored {len(points)} embeddings in Qdrant")
    except Exception as e:
        print(f"Error storing embeddings: {e}")

def query_qdrant(query_text, limit=5):
    """Query Qdrant for similar documents"""
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query_text)
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit
        )
        
        return search_results
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return []

def rag_query(question, limit=5):
    """Perform RAG query: retrieve relevant docs and generate response"""
    print(f"Searching for relevant documents for: {question}")
    
    # Get relevant documents
    search_results = query_qdrant(question, limit)
    
    if not search_results:
        return "No relevant documents found."
    
    # Prepare context from retrieved documents
    context_parts = []
    for result in search_results:
        context_parts.append(f"Source: {result.payload['source_file']}\n{result.payload['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Create prompt for LLM
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's question based on the provided context. If the context doesn't contain enough information to answer the question, say so."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    
    # Get response from Llama
    try:
        response = get_chat_response(messages)
        return response
    except Exception as e:
        print(f"Error getting chat response: {e}")
        return "Error generating response."

import os
from unstructured.partition.auto import partition
import json
from pathlib import Path

def extract_data_from_directory(directory_path="./public"):
    """
    Extract structured data from all files in the specified directory
    """
    # Supported file extensions by Unstructured
    supported_extensions = {
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".xlsx",
        ".xls",
        ".html",
        ".htm",
        ".xml",
        ".txt",
        ".md",
        ".rtf",
        ".odt",
        ".epub",
        ".msg",
        ".eml",
        ".csv",
        ".tsv",
        ".json",
    }

    extracted_data = []
    failed_files = []

    # Get all files in directory
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return [], []

    # Process all files
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                print(f"Processing: {file_path}")

                # Use Unstructured to partition the document
                elements = partition(str(file_path))

                # Extract text and metadata from each element
                file_data = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                    "elements": [],
                }

                for element in elements:
                    element_data = {
                        "text": str(element),
                        "type": element.category,
                        "metadata": (
                            element.metadata.to_dict()
                            if hasattr(element, "metadata") and element.metadata
                            else {}
                        ),
                    }
                    file_data["elements"].append(element_data)

                extracted_data.append(file_data)
                print(f"✓ Successfully processed {file_path.name}")

            except Exception as e:
                print(f"✗ Failed to process {file_path}: {str(e)}")
                failed_files.append({"file_path": str(file_path), "error": str(e)})

    return extracted_data, failed_files


def save_extracted_data(extracted_data, output_file="extracted_data.json"):
    """
    Save extracted data to JSON file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {output_file}")


def get_all_text_chunks(extracted_data, min_length=50):
    """
    Get all text chunks for RAG pipeline
    """
    chunks = []

    for file_data in extracted_data:
        for element in file_data["elements"]:
            text = element["text"].strip()
            if len(text) >= min_length:
                chunk = {
                    "text": text,
                    "source_file": file_data["file_name"],
                    "element_type": element["type"],
                    "metadata": element["metadata"],
                }
                chunks.append(chunk)

    return chunks


# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        # Query mode
        if len(sys.argv) < 3:
            print("Usage: python app.py query 'your question here'")
            sys.exit(1)
        
        question = " ".join(sys.argv[2:])
        print(f"\n=== RAG Query Mode ===")
        print(f"Question: {question}")
        print("=" * 50)
        
        answer = rag_query(question)
        print(f"\nAnswer:\n{answer}")
        
    else:
        # Document processing mode
        print("Starting extraction from ../public directory...")
        extracted_data, failed_files = extract_data_from_directory("../public")

        # Print summary
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Successfully processed: {len(extracted_data)} files")
        print(f"Failed files: {len(failed_files)}")

        if failed_files:
            print("\nFailed files:")
            for fail in failed_files:
                print(f"  - {fail['file_path']}: {fail['error']}")

        # Save raw extracted data
        if extracted_data:
            save_extracted_data(extracted_data)

            # Get text chunks for RAG
            chunks = get_all_text_chunks(extracted_data)
            print(f"\nTotal text chunks extracted: {len(chunks)}")

            # Save chunks separately for RAG pipeline
            with open("rag_chunks.json", "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            print("RAG-ready chunks saved to rag_chunks.json")

            # Print sample of extracted content
            print(f"\n=== SAMPLE EXTRACTED CONTENT ===")
            for i, file_data in enumerate(extracted_data[:2]):  # Show first 2 files
                print(f"\nFile: {file_data['file_name']}")
                print(f"Elements found: {len(file_data['elements'])}")

                # Show first few elements
                for j, element in enumerate(file_data["elements"][:3]):
                    print(
                        f"  Element {j+1} ({element['type']}): {element['text'][:100]}..."
                    )
            
            # Create Qdrant collection and store embeddings
            print(f"\n=== STORING IN QDRANT ===")
            create_collection()
            store_embeddings_in_qdrant(chunks)
            
            print(f"\n=== RAG SETUP COMPLETE ===")
            print("You can now query the documents using:")
            print("python app.py query 'your question here'")
            
        else:
            print("No files were successfully processed.")
