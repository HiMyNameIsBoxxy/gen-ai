import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def generate_samples(model, device, num_samples=16):
    model.generator.to(device)
    model.generator.eval()

    # Generate random latent noise
    z = torch.randn(num_samples, 100, device=device)

    # Generate fake images
    with torch.no_grad():
        fake_imgs = model.generator(z).detach().cpu()

    # Make a grid of images
    grid = make_grid(
        fake_imgs,
        nrow=int(num_samples ** 0.5),   # square-ish grid
        normalize=True,                 # auto rescale from [-1,1] → [0,1]
        value_range=(-1, 1),            # Tanh output range
        pad_value=1                     # white space between images
    )

    # Display
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Generated Samples", fontsize=14)
    plt.tight_layout()
    plt.show()
