import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from helper_lib.model import get_model
from helper_lib.trainer import train_ebm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device.upper()}")

    # ----------------------------------------------------
    # Data Preparation (CIFAR-10)
    # ----------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=1),   # EBM expects 1-channel images
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # ----------------------------------------------------
    # Build EBM Model
    # ----------------------------------------------------
    model = get_model("EBM")     # returns EBM wrapper + energy network
    model.to(device)

    # Optimizer — usually Adam works best for EBMs
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------------------------------------
    # Train Model
    # ----------------------------------------------------
    train_ebm(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=10,           # you can lower to 3–5 for a demo
        sample_interval=2    # plot generated samples every 2 epochs
    )

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
