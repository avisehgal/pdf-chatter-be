from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_retriver import vector_search, gpt_completion_with_vector_search
import os
import shutil
import requests
import pinecone
import fitz

# Initialize Pinecone
pinecone.init(api_key="afd697ea-500f-45a3-85e7-c82ae0d0829b", environment="us-east-1-aws")
index_name = "avi"
index = pinecone.Index(index_name)

app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to the specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    upload_dir = "C:\\temp"  # Use a consistent directory for file storage
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from the uploaded PDF
        extracted_text = extract_text_from_pdf(file_path)

        # Generate embeddings using Llama 2 (assuming you have a local endpoint for it)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": extracted_text}
        )
        response.raise_for_status()
        embeddings = response.json()['embeddings']

        # Upsert the embeddings into Pinecone
        upsert_data = [{"id": file.filename, "values": embeddings}]
        index.upsert(vectors=upsert_data)

        return {"message": "File uploaded and data stored in Pinecone", "filename": file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(query: str = Form(...)):
    try:
        # Perform vector search in Pinecone
        search_results = vector_search(query, index=index)

        # Generate a response with Llama 2 based on the search results
        context = " ".join([result['text'] for result in search_results])
        prompt = f"Based on the following context: {context}, answer the following query: {query}"

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt}
        )
        response.raise_for_status()
        generated_text = response.json()['text']
        return {"response": generated_text}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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