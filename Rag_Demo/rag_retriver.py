import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load credentials from the .env file
load_dotenv("credentials.env")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("openai_api"))

# Initialize Pinecone
pinecone = Pinecone(
    api_key=os.environ.get("Pinecone_api_key")
)

# Check if the index exists, if not, create it
index_name = os.environ.get("index_name")

if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # Replace with the appropriate dimension for your embeddings
        metric='cosine',  # Or use the metric that fits your use case
        spec=ServerlessSpec(
            cloud='gcp',  # Replace with the appropriate cloud provider
            region=os.environ.get("Pinecone_environment")
        )
    )

index = pinecone.Index(index_name)

def vector_search(query):
    # Generate embeddings and perform vector search here
    Rag_data = ""
    xq = client.embeddings.create(input=query, model="text-embedding-ada-002")
    res = index.query([xq.data[0].embedding], top_k=2, include_metadata=True)
    for match in res['matches']:
        if match['score'] < 0.80:
            continue
        Rag_data += match['metadata']['text']
    return Rag_data

def gpt_completion_with_vector_search(prompt, rag):
    DEFAULT_SYSTEM_PROMPT = "Your system prompt here"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": rag + ", Prompt: " + prompt},
        ]
    )
    return response.choices[0].message.content
