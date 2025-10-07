import torch
import torch.nn as nn

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total, correct = 0, 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Compute loss for this batch
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Predictions
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute average loss and accuracy
    avg_loss = test_loss / len(test_loader)
    acc = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.2f}%")
    return avg_loss, acc
