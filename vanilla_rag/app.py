# from ollama import embed

# def get_embedding(text):
#     response = embed(model="mxbai-embed-large:latest", input=text)
#     return response['embeddings']

# if __name__ == "__main__":
#     sample_text = "Hello, this is a sample text for embedding."
#     embedding = get_embedding(sample_text)
#     print(embedding[0])

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
    # Extract data from public directory in parent folder
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
    else:
        print("No files were successfully processed.")
