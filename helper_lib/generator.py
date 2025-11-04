import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Gan sample generation
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

# Diffusion model sample generation
def generate_diffusion_samples(model, device, num_samples=10, diffusion_steps=100):
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Generate starting noise (standard normal)
        x = torch.randn(num_samples, 3, 32, 32, device=device)

        # Iteratively perform reverse diffusion
        for step in reversed(range(diffusion_steps)):
            # diffusion time normalized between 0 and 1
            t = torch.full((num_samples, 1, 1, 1), step / diffusion_steps, device=device)
            noise_rates, signal_rates = model.schedule_fn(t)

            # Predict the noise and denoised image
            pred_noises, pred_images = model.denoise(x, noise_rates, signal_rates, training=False)

            # Update x for the next step
            x = pred_images

        # Bring values to [0, 1] for visualization
        samples = model.denormalize(x).detach().cpu()

        # Plot grid of generated samples
        grid = make_grid(samples, nrow=int(num_samples ** 0.5), normalize=True, pad_value=1)
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title("Diffusion Model Generated Samples", fontsize=14)
        plt.tight_layout()
        plt.show()