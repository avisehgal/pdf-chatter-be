import os
import requests
import json  # Import the built-in JSON module
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone with API key directly from the environment variables
pinecone = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")  # Use the environment variable directly
)

# Index name and environment are also retrieved from environment variables
index_name = os.environ.get("INDEX_NAME")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")

# Check if the index exists, if not, create it
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=4096,  # Replace with the appropriate dimension for your embeddings
        metric='cosine',  # Or use the metric that fits your use case
        spec=ServerlessSpec(
            cloud='aws',  # Replace with the appropriate cloud provider
            region=pinecone_environment
        )
    )

index = pinecone.Index(index_name)

def vector_search(query):
    # Generate embeddings using the local Llama 2 model
    response = requests.post(
        "http://localhost:11434/api/embeddings",  # Ensure this is the correct endpoint
        json={"model": "llama2", "prompt": query},
        stream=True  # This enables streaming of the response
    )

    response.raise_for_status()

    # Initialize an empty list to accumulate the embedding vectors
    accumulated_response = []

    # Process each chunk in the streaming response
    for chunk in response.iter_lines():
        if chunk:
            json_chunk = json.loads(chunk.decode("utf-8"))
            print("Received Chunk: ", json_chunk)
            embedding = json_chunk.get("embedding", [])
            accumulated_response.extend(embedding)

    # Ensure the accumulated_response is a list of floats before passing to Pinecone
    if not accumulated_response or not all(isinstance(x, (float, int)) for x in accumulated_response):
        raise ValueError("The response did not contain a valid list of floats")

    # Perform vector search in Pinecone using the embeddings
    
    res = index.query(vector=accumulated_response, top_k=2, include_metadata=True)
    print("Pinecone Search Results: ", res['matches'])
    
    Rag_data = ""
    for match in res['matches']:
        if match['score'] < 0.80:
            continue
        # Check if metadata exists before accessing it
        if 'metadata' in match:
            Rag_data += match['metadata'].get('text', '')

    return Rag_data
