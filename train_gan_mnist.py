import torch
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.mnist_loader import get_mnist_loader

if __name__ == "__main__":
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load MNIST data
    train_loader = get_mnist_loader(batch_size=128)
    print("MNIST dataset ready.")

    # Initialize GAN
    model = get_model("GAN")
    print("GAN model created.")

    # Train GAN
    trained_gan = train_gan(model, train_loader, device=device, epochs=10)
    print("Training complete. Check saved weights in project folder.")
