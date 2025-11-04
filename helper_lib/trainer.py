import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# CNN Training Function
# Training loop for model broken into 10 epochs
def train_model(model, train_loader, device, epochs=10, lr=0.0005):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    # Loop over dataset multiple times (epochs)
    for epoch in range(epochs):
        running_loss, running_correct, running_total = 0, 0, 0
        loop = tqdm(train_loader, ncols=100, desc=f"Epoch {epoch+1}/{epochs}")

        # Batch loop
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # Loss calculation
            loss = criterion(outputs, labels)
            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accuracy tracking
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            loop.set_postfix(loss=running_loss/(running_total//labels.size(0)),
                             acc=running_correct/running_total)
    
    # Save model
    torch.save(model.state_dict(), "cnn_weights.pth")
    print("Model saved as cnn_weights.pth")




# GAN Training Function (Generator and Discriminator)
def train_gan(model, data_loader, criterion=None, optimizer=None, device='cuda', epochs=10):
    # Use Binary Cross Entropy Loss
    if criterion is None:
        criterion = nn.BCELoss()

    # Separate optimizers for generator and discriminator
    optimizer_G = optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Move model to device
    model.to(device)
    model.train()

    # Fixed noise for monitoring progress
    fixed_noise = torch.randn(16, 100, device=device)

    for epoch in range(epochs):
        loop = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{epochs}]", ncols=100)
        for real_imgs, _ in loop:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real images loss
            outputs_real = model.discriminator(real_imgs)
            loss_real = criterion(outputs_real, real_labels)

            # Fake images loss
            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = model.generator(z)
            outputs_fake = model.discriminator(fake_imgs.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

            # Total discriminator loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = model.generator(z)
            outputs = model.discriminator(fake_imgs)

            # Generator wants discriminator to predict "real" (label = 1)
            loss_G = criterion(outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()

            # Logging
            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        torch.save({
            'generator_state_dict': model.generator.state_dict(),
            'discriminator_state_dict': model.discriminator.state_dict(),
        }, "gan_weights.pth")


    print("GAN training complete and model saved.")
    return model



# Diffusion Model Training Function (on CIFAR-10-style images)
def train_diffusion(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=120):
            images = images.to(device)
            loss = model.train_step(images, optimizer, criterion)
            train_losses.append(loss)

        avg_train_loss = sum(train_losses) / len(train_losses)

        # ---- Validation Loop ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", ncols=120):
                images = images.to(device)
                loss = model.test_step(images, criterion)
                val_losses.append(loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save the full Diffusion model (wrapper + EMA)
    torch.save(model.state_dict(), "diffusion_weights.pth")
    print("Training complete. Full Diffusion model saved as diffusion_weights.pth")

    return model


# -------------------------------------------------------
# Energy-Based Model (EBM) Trainer
# -------------------------------------------------------
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

@torch.no_grad()
def clip_img(x):
    return torch.clamp((x + 1) / 2, 0, 1)  # scale from [-1,1] → [0,1]

def plot_samples(samples, n=8):
    samples = clip_img(samples)
    samples = samples.cpu()
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        img = samples[i].permute(1, 2, 0).squeeze()
        axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i].axis("off")
    plt.show()

def train_ebm(model, train_loader, optimizer, device='cuda', epochs=10, sample_interval=2):
    """
    Trains an Energy-Based Model (EBM).
    Args:
        model: instance of EBM wrapper class.
        train_loader: DataLoader for training images.
        optimizer: torch optimizer (usually Adam).
        device: 'cuda' or 'cpu'.
        epochs: number of epochs.
        sample_interval: how often (in epochs) to generate and show samples.
    """
    model.to(device)

    print("Starting Energy-Based Model training...")
    for epoch in range(epochs):
        model.train()
        model.reset_metrics()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=120)
        for real_imgs, _ in loop:
            real_imgs = real_imgs.to(device)

            # --- One training step ---
            metrics = model.train_step(real_imgs, optimizer)

            loop.set_postfix({
                "Loss": f"{metrics['loss']:.3f}",
                "CDiv": f"{metrics['cdiv']:.3f}",
                "Reg": f"{metrics['reg']:.3f}"
            })

        # --- Log average metrics for the epoch ---
        epoch_metrics = model.metrics()
        print(
            f"\nEpoch {epoch+1} Summary → "
            f"Loss: {epoch_metrics['loss']:.4f}, "
            f"CDiv: {epoch_metrics['cdiv']:.4f}, "
            f"Reg: {epoch_metrics['reg']:.4f}, "
            f"Real: {epoch_metrics['real']:.4f}, "
            f"Fake: {epoch_metrics['fake']:.4f}"
        )

        # --- Visualization every few epochs ---
        if (epoch + 1) % sample_interval == 0:
            print("Generating sample images...")
            with torch.no_grad():
                # Sample images from the replay buffer
                fake_imgs = model.buffer.sample_new_exmps(
                    steps=model.steps, step_size=model.step_size, noise=model.noise
                )
                plot_samples(fake_imgs, n=8)

    # --- Save model weights ---
    torch.save(model.model.state_dict(), "ebm_weights.pth")
    print("EBM training complete. Model saved as ebm_weights.pth")
