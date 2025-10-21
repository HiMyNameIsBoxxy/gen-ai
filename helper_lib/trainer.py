import torch
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
