import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])

    # Downloads CIFAR-10 training split (50,000 images, 10 classes)
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    # Downloads CIFAR-10 test split (10,000 images, 10 classes)
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Data loaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# MNIST Data Loader
def get_mnist_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader