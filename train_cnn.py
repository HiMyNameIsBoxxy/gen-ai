# train_cnn.py

from helper_lib.data_loader import get_data_loaders
from helper_lib.model import EnhancedCNN
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import get_device, set_seed

def main():
    # Reproducibility + device
    set_seed(42)
    device = get_device()

    # Get CIFAR-10 train/test data
    train_loader, test_loader = get_data_loaders(batch_size=32)

    # Initialize model
    model = EnhancedCNN()

    # Train and save weights
    train_model(model, train_loader, device, epochs=10, lr=0.0005)

    # Evaluate
    avg_loss, acc = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
