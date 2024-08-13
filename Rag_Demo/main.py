import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone, ServerlessSpec
from Rag_Demo.rag_retriver import vector_search
import requests
import fitz
import shutil
from fastapi.responses import StreamingResponse
import json

# Create a Pinecone instance at the global level
pinecone = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Define the index name and environment
index_name = os.environ.get("INDEX_NAME")

# Check if the index exists, if not, create it
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=4096,  # Updated to 4096 as per your embeddings
        metric='cosine',  # Or use the metric that fits your use case
        spec=ServerlessSpec(
            cloud='aws',  # Replace with the appropriate cloud provider
            region=os.environ.get("PINECONE_ENVIRONMENT")
        )
    )

# Create the index object to be reused across functions
index = pinecone.Index(index_name)

# FastAPI app initialization
app = FastAPI()

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to the specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    upload_dir = "C:\\temp"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(file_path)
    print("Extracted Text: ", extracted_text)


    # Generate embeddings using Llama 2 with the correct API endpoint
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "llama2", "prompt": extracted_text},
        stream=True  # Enable streaming of the response
    )
    response.raise_for_status()

    # Process the streamed response to ensure it becomes a list of floats
    embeddings = []
    for chunk in response.iter_lines():
        if chunk:
            json_chunk = json.loads(chunk.decode("utf-8"))
            # Ensure the embeddings are floats
            if "embedding" in json_chunk:
                embeddings = json_chunk["embedding"]  # Assuming it's a list of floats
                break

    # Validate that embeddings is indeed a list of floats
    if not isinstance(embeddings, list) or not all(isinstance(i, float) for i in embeddings):
        raise ValueError("Embeddings are not in the correct format.")
    print("Generated Embeddings: ", embeddings)

    # Upsert the embeddings into Pinecone
    upsert_data = [{"id": file.filename, "values": embeddings, "metadata": {"text": extracted_text}}]
    index.upsert(vectors=upsert_data)

    return {"message": "File uploaded and data stored in Pinecone", "filename": file.filename}


@app.post("/chat")
async def chat(query: str = Form(...)):
    # Perform vector search in Pinecone
    search_results = vector_search(query)

    # Generate a response with Llama 2 based on the search results
    context = " ".join([result['text'] for result in search_results])
    prompt = f"Based on the following context: {context}, answer the following query: {query}"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama2", "prompt": prompt},
        stream=True  # Enable streaming of the response
    )

    def event_stream():
        for chunk in response.iter_lines():
            if chunk:
                json_chunk = json.loads(chunk.decode("utf-8"))
                yield f"data: {json_chunk.get('response', '')}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Example test route
@app.get("/")
async def read_root():
    return {"message": "API is running"}

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
