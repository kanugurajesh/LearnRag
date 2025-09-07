from ollama import embed
import numpy as np

def get_embedding(text):
    """
    Get embedding for a given text using Ollama's mxbai-embed-large model.

    Args:
        text (str): Input text to embed

    Returns:
        list: Embedding vector
    """
    try:
        response = embed(model="mxbai-embed-large:latest", input=text)

        # The response['embeddings'] is typically a list of lists
        # Each inner list represents an embedding for each input text
        embeddings = response["embeddings"]

        print(f"Type of embeddings: {type(embeddings)}")
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")

        # Return the first (and likely only) embedding
        return embeddings[0] if embeddings else []

    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def get_embeddings_batch(texts):
    """
    Get embeddings for multiple texts at once.

    Args:
        texts (list): List of input texts to embed

    Returns:
        list: List of embedding vectors
    """
    try:
        response = embed(model="mxbai-embed-large:latest", input=texts)
        return response["embeddings"]
    except Exception as e:
        print(f"Error getting batch embeddings: {e}")
        return None


def save_embedding_to_file(embedding, filename="embedding.txt"):
    """
    Save embedding to a text file.

    Args:
        embedding (list): Embedding vector
        filename (str): Output filename
    """
    try:
        with open(filename, "w") as f:
            # Save as comma-separated values
            f.write(",".join(map(str, embedding)))
        print(f"Embedding saved to {filename}")
    except Exception as e:
        print(f"Error saving embedding: {e}")


def load_embedding_from_file(filename="embedding.txt"):
    """
    Load embedding from a text file.

    Args:
        filename (str): Input filename

    Returns:
        list: Embedding vector
    """
    try:
        with open(filename, "r") as f:
            content = f.read().strip()
            return [float(x) for x in content.split(",")]
    except Exception as e:
        print(f"Error loading embedding: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1, vec2 (list): Two embedding vectors

    Returns:
        float: Cosine similarity score
    """
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        return dot_product / (norm1 * norm2)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return None


if __name__ == "__main__":
    # Single text embedding
    sample_text = "Hello, this is a sample text for embedding."
    embedding = get_embedding(sample_text)

    if embedding:
        print(f"Embedding for '{sample_text}':")
        print(f"First 10 values: {embedding[:10]}")
        print(f"Embedding length: {len(embedding)}")

        # Save to file
        save_embedding_to_file(embedding, "sample_embedding.txt")

    # Batch embedding example
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is a completely different topic about cats.",
    ]

    print("\n" + "=" * 50)
    print("Batch embedding example:")

    batch_embeddings = get_embeddings_batch(texts)
    if batch_embeddings:
        for i, (text, emb) in enumerate(zip(texts, batch_embeddings)):
            print(f"Text {i+1}: '{text}'")
            print(f"Embedding length: {len(emb)}")
            print(f"First 5 values: {emb[:5]}")
            print()

    # Similarity comparison example
    if batch_embeddings and len(batch_embeddings) >= 3:
        print("Similarity comparison:")
        sim_1_2 = cosine_similarity(batch_embeddings[0], batch_embeddings[1])
        sim_1_3 = cosine_similarity(batch_embeddings[0], batch_embeddings[2])

        print(f"Similarity between text 1 and 2: {sim_1_2:.4f}")
        print(f"Similarity between text 1 and 3: {sim_1_3:.4f}")

        if sim_1_2 > sim_1_3:
            print("Text 1 and 2 are more similar (as expected)")
        else:
            print("Text 1 and 3 are more similar (unexpected)")
