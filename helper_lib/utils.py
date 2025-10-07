import io
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Sets seed so that results are consistent across runs
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

# Picks the best available hardware to run model
def get_device():
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

# CIFAR-10 class names for mapping model output to human-readable labels
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Preprocess uploaded image for inference
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # Rezize iamge to CIFAR-10 dimensions
        transforms.ToTensor(), # Convert image to PyTorch tensor
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension 
    # PyTorch models expect input as [batch, channels, height, width]
