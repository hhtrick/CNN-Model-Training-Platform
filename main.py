import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import sys
import pandas as pd
from PIL import Image

class ClassifyLeavesDataset(Dataset):
    """Custom dataset for classify-leaves"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied
        """
        print(f"Loading CSV file: {csv_file}")
        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        print("Creating label mappings...")
        # Create label to index mapping
        unique_labels = sorted(self.data_df['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        print(f"Dataset loaded with {len(self.data_df)} samples and {len(unique_labels)} classes")
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.data_df.iloc[idx, 0])
        
        try:
            # Use fast image loading with PIL optimizations
            image = Image.open(img_name)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a dummy image in case of error
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        label_str = self.data_df.iloc[idx, 1]
        label = self.label_to_idx[label_str]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataset(dataset_name, batch_size):
    """
    Load the specified dataset
    
    Args:
        dataset_name: Name of the dataset to load
        batch_size: Batch size for data loaders
        
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
        num_classes: Number of classes in the dataset
    """
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Dataset path
    data_path = os.path.join('datasets')
    
    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.startswith('EMNIST'):
        split = dataset_name.split('-')[1]
        train_dataset = datasets.EMNIST(root=data_path, split=split, train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root=data_path, split=split, train=False, download=True, transform=transform)
        # Get number of classes from the dataset
        num_classes = len(train_dataset.classes)
    elif dataset_name == 'classify-leaves':
        # Real classify-leaves dataset implementation
        data_root = os.path.join('datasets', 'classify-leaves')
        train_csv = os.path.join(data_root, 'train_split.csv')
        test_csv = os.path.join(data_root, 'test_split.csv')
        
        # Define optimized transforms for classify-leaves
        transform_leaves = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),  # Faster interpolation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Creating classify-leaves datasets...")
        train_dataset = ClassifyLeavesDataset(train_csv, data_root, transform=transform_leaves)
        test_dataset = ClassifyLeavesDataset(test_csv, data_root, transform=transform_leaves)
        num_classes = len(train_dataset.label_to_idx)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data loaders with optimizations
    if dataset_name == 'classify-leaves':
        # Use multiple workers for classify-leaves to speed up data loading
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=2, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True, persistent_workers=True)
    else:
        # Standard data loaders for other datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, num_classes

def main():
    parser = argparse.ArgumentParser(description='Train classic CNN models')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['LeNet', 'LeNet-ReLU', 'LeNet-ReLU-Dropout', 'LeNet-ReLU-Dropout-BN',
                                 'ResNet', 'ResNet18', 'ResNet34', 'AlexNet', 'VGG', 'NiN'],
                        help='Model to train')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['MNIST', 'FashionMNIST', 'EMNIST-byclass', 'EMNIST-bymerge',
                                 'EMNIST-balanced', 'EMNIST-letters', 'EMNIST-digits', 'EMNIST-mnist',
                                 'classify-leaves'],
                        help='Dataset to use')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Custom model name for saving')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_loader, test_loader, num_classes = get_dataset(args.dataset, args.batch_size)
    print(f"Dataset loaded. Number of classes: {num_classes}")
    
    # Set model name
    if args.model_name is None:
        model_name = f"{args.model}_{args.dataset}"
    else:
        model_name = args.model_name
    
    # Import backend and train model
    try:
        from backend import train_model
        
        print(f"Training {args.model} model...")
        print(f"Optimizer: {args.optimizer}")
        print(f"Learning Rate: {args.learning_rate}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print("-" * 50)
        
        model, train_losses, train_accuracies, test_accuracies = train_model(
            model_name=args.model,
            dataset=args.dataset,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer_type=args.optimizer,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            model_custom_name=model_name,
            batch_size=args.batch_size
        )
        
        # Print final results
        print("-" * 50)
        print("Training completed!")
        print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()