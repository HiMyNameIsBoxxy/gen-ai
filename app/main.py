from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy
import torch

# Import helper_lib modules
from helper_lib.data_loader import get_data_loaders
from helper_lib.model import EnhancedCNN
from helper_lib.utils import get_device, set_seed, preprocess_image, CLASSES

app = FastAPI()

# ------------------------------
# Bigram Model Setup
# ------------------------------
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
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


# ------------------------------
# spaCy Embedding Setup
# ------------------------------
nlp = spacy.load("en_core_web_md")

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

@app.post("/embed", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest):
    doc = nlp(request.text)
    embedding = doc.vector.tolist() 
    return {"embedding": embedding}


# ------------------------------
# CNN Model Setup
# ------------------------------
set_seed(42)
device = get_device()

cnn_model = EnhancedCNN()
# Load pretrained weights (make sure cnn_weights.pth is in project root or adjust path)
cnn_model.load_state_dict(torch.load("cnn_weights.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

@app.post("/predict_cnn")
async def predict_cnn(file: UploadFile = File(...)):
    """
    Predict the CIFAR-10 class of an uploaded image.
    """
    image_bytes = await file.read()
    tensor = preprocess_image(image_bytes).to(device)

    with torch.no_grad():
        outputs = cnn_model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return {"class": CLASSES[pred_idx], "confidence": confidence}


# ------------------------------
# Root Endpoint
# ------------------------------
@app.get("/")
def read_root():
    return {"Hello": "World"}
