from ollama import embed

def get_embedding(text):
    response = embed(model="mxbai-embed-large:latest", input=text)
    return response['embeddings']

if __name__ == "__main__":
    sample_text = "Hello, this is a sample text for embedding."
    embedding = get_embedding(sample_text)
    print(embedding[0])