import torch
import importlib
import os
from torch.utils.data import DataLoader

def get_model_class(model_name):
    """
    Dynamically import and return the model class based on model name
    
    Args:
        model_name: Name of the model (e.g., 'LeNet', 'AlexNet')
        
    Returns:
        model_class: The model class
        train_module: The training module
    """
    try:
        # Convert model name to file name
        file_name = model_name.replace('-', '_')
        train_module = importlib.import_module(f'train_scripts.{file_name}')
        model_class = getattr(train_module, model_name.replace('-', ''))
        return model_class, train_module
    except Exception as e:
        raise ImportError(f"Could not import {model_name}: {str(e)}")

def train_model(model_name, dataset, train_loader, test_loader, 
                optimizer_type='Adam', learning_rate=0.001, epochs=10, 
                model_custom_name=None, progress_callback=None, batch_callback=None, **kwargs):
    """
    Train a specified model on a dataset
    
    Args:
        model_name: Name of the model to train
        dataset: Dataset name (for determining number of classes)
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer_type: Optimizer type ('Adam' or 'SGD')
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        model_custom_name: Custom name for the model
        progress_callback: Optional callback function for epoch progress updates
        batch_callback: Optional callback function for batch progress updates
        **kwargs: Additional arguments for model initialization
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        test_accuracies: List of test accuracies
    """
    # Determine number of classes and input channels based on dataset
    num_classes = get_num_classes(dataset)
    input_channels = get_input_channels(dataset)
    
    # Get model class and training module
    model_class, train_module = get_model_class(model_name)
    
    # Create model instance with input channels
    try:
        # Try to create model with input_channels parameter
        model = model_class(num_classes=num_classes, input_channels=input_channels, **kwargs)
    except TypeError:
        # Fallback for models that don't support input_channels parameter
        model = model_class(num_classes=num_classes, **kwargs)
    
    # Set model name
    if model_custom_name is None:
        model_custom_name = f"{model_name}_model"
    
    # Train model
    model, train_losses, train_accuracies, test_accuracies = train_module.train_model(
        model, train_loader, test_loader, optimizer_type, learning_rate, 
        epochs, model_custom_name, progress_callback=progress_callback,
        batch_callback=batch_callback
    )
    
    return model, train_losses, train_accuracies, test_accuracies

def get_num_classes(dataset):
    """
    Return the number of classes for a given dataset
    
    Args:
        dataset: Name of the dataset
        
    Returns:
        num_classes: Number of classes in the dataset
    """
    dataset_classes = {
        'MNIST': 10,
        'FashionMNIST': 10,
        'EMNIST-byclass': 62,
        'EMNIST-bymerge': 47,
        'EMNIST-balanced': 47,
        'EMNIST-letters': 26,
        'EMNIST-digits': 10,
        'EMNIST-mnist': 10,
        'classify-leaves': 176  # Based on the dataset documentation
    }
    
    # Handle generic EMNIST
    if dataset.startswith('EMNIST-'):
        return dataset_classes.get(dataset, 10)
    
    return dataset_classes.get(dataset, 10)

def get_input_channels(dataset):
    """
    Return the number of input channels for a given dataset
    
    Args:
        dataset: Name of the dataset
        
    Returns:
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
    """
    # Most datasets are grayscale (1 channel)
    grayscale_datasets = ['MNIST', 'FashionMNIST']
    
    # EMNIST datasets are also grayscale
    if dataset.startswith('EMNIST-'):
        return 1
    
    # classify-leaves is RGB (3 channels)
    if dataset == 'classify-leaves':
        return 3
    
    # Default to grayscale for other datasets
    return 1

def evaluate_model(model_name, model, test_loader):
    """
    Evaluate a trained model
    
    Args:
        model_name: Name of the model
        model: Trained model
        test_loader: Test data loader
        
    Returns:
        accuracy: Test accuracy
    """
    _, train_module = get_model_class(model_name)
    return train_module.evaluate_model(model, test_loader)

if __name__ == "__main__":
    print("Backend module for training classic CNN models")
    print("Available models:")
    models = ['LeNet', 'LeNet-ReLU', 'LeNet-ReLU-Dropout', 'LeNet-ReLU-Dropout-BN', 
              'ResNet', 'ResNet18', 'ResNet34', 'AlexNet', 'VGG', 'NiN']
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")