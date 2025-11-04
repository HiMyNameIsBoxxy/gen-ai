import torch
import torch.nn as nn
from helper_lib.data_loader import get_data_loaders
from helper_lib.model import get_model 
from helper_lib.trainer import train_diffusion

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_data_loaders(batch_size=64)

    # build the diffusion model
    model = get_model("Diffusion").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss()

    train_diffusion(model, train_loader, test_loader, loss_fn, optimizer, epochs=10, device=device)

if __name__ == "__main__":
    main()
