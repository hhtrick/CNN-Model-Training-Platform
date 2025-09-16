import os
import torch
from torchvision import datasets, transforms

# Create datasets directory if it doesn't exist
os.makedirs('datasets', exist_ok=True)

# Define transform
transform = transforms.ToTensor()

print("Downloading datasets...")

# Download MNIST dataset
print("Downloading MNIST...")
mnist_path = os.path.join('datasets', 'MNIST')
os.makedirs(mnist_path, exist_ok=True)
mnist_train = datasets.MNIST(root='datasets', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='datasets', train=False, download=True, transform=transform)
print(f"MNIST train set size: {len(mnist_train)}")
print(f"MNIST test set size: {len(mnist_test)}")

# Download FashionMNIST dataset
print("Downloading FashionMNIST...")
fashion_mnist_path = os.path.join('datasets', 'FashionMNIST')
os.makedirs(fashion_mnist_path, exist_ok=True)
fashion_mnist_train = datasets.FashionMNIST(root='datasets', train=True, download=True, transform=transform)
fashion_mnist_test = datasets.FashionMNIST(root='datasets', train=False, download=True, transform=transform)
print(f"FashionMNIST train set size: {len(fashion_mnist_train)}")
print(f"FashionMNIST test set size: {len(fashion_mnist_test)}")

# Download EMNIST datasets - it has multiple splits
print("Downloading EMNIST...")
emnist_path = os.path.join('datasets', 'EMNIST')
os.makedirs(emnist_path, exist_ok=True)

# EMNIST has 6 splits: byclass, bymerge, balanced, letters, digits, mnist
emnist_splits = ['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist']

for split in emnist_splits:
    print(f"Downloading EMNIST {split}...")
    
    try:
        emnist_train = datasets.EMNIST(root='datasets', split=split, train=True, download=True, transform=transform)
        emnist_test = datasets.EMNIST(root='datasets', split=split, train=False, download=True, transform=transform)
        print(f"EMNIST {split} train set size: {len(emnist_train)}")
        print(f"EMNIST {split} test set size: {len(emnist_test)}")
    except Exception as e:
        print(f"Failed to download EMNIST {split}: {e}")

print("Dataset download complete!")