from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy
import torch
from io import BytesIO
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from fastapi.responses import Response

# Import helper_lib modules
from helper_lib.data_loader import get_data_loaders
from helper_lib.model import EnhancedCNN
from helper_lib.utils import get_device, set_seed, preprocess_image, CLASSES
from helper_lib.model import get_model

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
# GAN Model Setup
# ------------------------------
gan_model = get_model("GAN").to(device)

@app.on_event("startup")
def load_gan():
    global gan_model
    checkpoint = torch.load("gan_weights.pth", map_location=device)
    gan_model.generator.load_state_dict(checkpoint["generator_state_dict"])
    gan_model.generator.to(device)
    gan_model.generator.eval()
    print("GAN model loaded.")

@app.get("/generate_gan", response_class=Response)
# Generate handwritten digits using the trained GAN model.
def generate_gan(num: int = 16):
    with torch.no_grad():
        z = torch.randn(num, 100, device=device)
        fake_imgs = gan_model.generator(z).cpu()

    grid = make_grid(fake_imgs, nrow=int(num**0.5), normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


# ------------------------------
# Root Endpoint
# ------------------------------
@app.get("/")
def read_root():
    return {"Hello": "World"}
