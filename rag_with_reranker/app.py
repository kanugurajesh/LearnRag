from ollama import embed, chat
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import CrossEncoder
import uuid
import os
from unstructured.partition.auto import partition
import json
from pathlib import Path

def get_embedding(text):
    response = embed(model="mxbai-embed-large:latest", input=text)
    return response["embeddings"][0]

def get_chat_response(messages):
    response = chat(model="llama3.2:latest", messages=messages)
    return response.message.content

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

# Initialize reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

COLLECTION_NAME = "documents_reranked"

def create_collection():
    """Create a collection in Qdrant for storing document embeddings"""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if COLLECTION_NAME not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
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
            embedding = get_embedding(chunk["text"])

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "element_type": chunk["element_type"],
                    "metadata": chunk["metadata"],
                },
            )
            points.append(point)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks...")
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    try:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Successfully stored {len(points)} embeddings in Qdrant")
    except Exception as e:
        print(f"Error storing embeddings: {e}")


def query_qdrant(query_text, limit=20):
    """Query Qdrant for similar documents (retrieve more for reranking)"""
    try:
        query_embedding = get_embedding(query_text)

        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=limit
        )

        return search_results
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return []


def rerank_documents(query, documents, top_k=5):
    """Rerank documents using cross-encoder model"""
    if not documents:
        return []
    
    # Prepare query-document pairs for reranking
    pairs = []
    for doc in documents:
        pairs.append([query, doc.payload["text"]])
    
    # Get reranking scores
    scores = reranker.predict(pairs)
    
    # Combine documents with their reranking scores
    scored_docs = list(zip(documents, scores))
    
    # Sort by reranking score (descending)
    ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    # Return top_k documents
    return [doc for doc, score in ranked_docs[:top_k]]


def rag_query_with_reranker(question, initial_limit=20, final_limit=5):
    """Perform RAG query with reranking: retrieve docs, rerank, then generate response"""
    print(f"Searching for relevant documents for: {question}")

    # Step 1: Retrieve initial set of documents
    search_results = query_qdrant(question, initial_limit)

    if not search_results:
        return "No relevant documents found."

    print(f"Retrieved {len(search_results)} documents from vector search")

    # Step 2: Rerank documents
    reranked_docs = rerank_documents(question, search_results, final_limit)
    print(f"Reranked to top {len(reranked_docs)} documents")

    # Step 3: Prepare context from reranked documents
    context_parts = []
    for i, doc in enumerate(reranked_docs):
        context_parts.append(
            f"Source {i+1}: {doc.payload['source_file']}\n{doc.payload['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate response
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's question based on the provided context. The context has been ranked by relevance. If the context doesn't contain enough information to answer the question, say so.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    try:
        response = get_chat_response(messages)
        return response
    except Exception as e:
        print(f"Error getting chat response: {e}")
        return "Error generating response."


def extract_data_from_directory(directory_path="./public"):
    """Extract structured data from all files in the specified directory"""
    supported_extensions = {
        ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
        ".html", ".htm", ".xml", ".txt", ".md", ".rtf", ".odt",
        ".epub", ".msg", ".eml", ".csv", ".tsv", ".json",
    }

    extracted_data = []
    failed_files = []

    directory = Path(directory_path)

    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return [], []

    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                print(f"Processing: {file_path}")
                elements = partition(str(file_path))

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


def save_extracted_data(extracted_data, output_file="extracted_data_reranked.json"):
    """Save extracted data to JSON file"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {output_file}")


def get_all_text_chunks(extracted_data, min_length=50):
    """Get all text chunks for RAG pipeline"""
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
        # Query mode with reranking
        if len(sys.argv) < 3:
            print("Usage: python app.py query 'your question here'")
            sys.exit(1)

        question = " ".join(sys.argv[2:])
        print(f"\n=== RAG Query Mode with Reranking ===")
        print(f"Question: {question}")
        print("=" * 50)

        answer = rag_query_with_reranker(question)
        print(f"\nAnswer:\n{answer}")

    else:
        # Document processing mode
        print("Starting extraction from ../public directory...")
        extracted_data, failed_files = extract_data_from_directory("../public")

        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Successfully processed: {len(extracted_data)} files")
        print(f"Failed files: {len(failed_files)}")

        if failed_files:
            print("\nFailed files:")
            for fail in failed_files:
                print(f"  - {fail['file_path']}: {fail['error']}")

        if extracted_data:
            save_extracted_data(extracted_data)

            chunks = get_all_text_chunks(extracted_data)
            print(f"\nTotal text chunks extracted: {len(chunks)}")

            with open("rag_chunks_reranked.json", "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            print("RAG-ready chunks saved to rag_chunks_reranked.json")

            print(f"\n=== SAMPLE EXTRACTED CONTENT ===")
            for i, file_data in enumerate(extracted_data[:2]):
                print(f"\nFile: {file_data['file_name']}")
                print(f"Elements found: {len(file_data['elements'])}")

                for j, element in enumerate(file_data["elements"][:3]):
                    print(
                        f"  Element {j+1} ({element['type']}): {element['text'][:100]}..."
                    )

            print(f"\n=== STORING IN QDRANT ===")
            create_collection()
            store_embeddings_in_qdrant(chunks)

            print(f"\n=== RAG WITH RERANKER SETUP COMPLETE ===")
            print("You can now query the documents using:")
            print("python app.py query 'your question here'")
            print("\nReranking process:")
            print("1. Retrieves top 20 documents using vector similarity")
            print("2. Reranks using cross-encoder model")
            print("3. Uses top 5 reranked documents for answer generation")

        else:
            print("No files were successfully processed.")