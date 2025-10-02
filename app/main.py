# main.py
# Start command is: uv run uvicorn app.main:app --reload
# http://127.0.0.1:8000/docs for documentation

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy

app = FastAPI()

# Bigram Model Setup
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
    It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """
    Generate text based on bigram probabilities.
    """
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


# spaCy Embedding Setup
nlp = spacy.load("en_core_web_md")

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

@app.post("/embed", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest):
    """
    Generate a vector embedding for input text using spaCy.
    """
    doc = nlp(request.text)
    embedding = doc.vector.tolist() 
    return {"embedding": embedding}


# Root Endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}
