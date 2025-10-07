import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
