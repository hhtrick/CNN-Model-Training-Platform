import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

class LeNetReLUDropoutBN(nn.Module):
    """
    LeNet implementation with ReLU activation, Dropout layers, and Batch Normalization, adapted for various input sizes.
    """
    def __init__(self, num_classes=10, input_channels=1, dropout_rate=0.5):
        super(LeNetReLUDropoutBN, self).__init__()
        # First convolutional block with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)  # Dynamic input channels
        self.bn1 = nn.BatchNorm2d(6)
        
        # Second convolutional block with batch normalization
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.features = nn.Sequential(
            # First convolutional layer with batch normalization
            self.conv1,
            # Batch normalization
            self.bn1,
            # ReLU activation
            nn.ReLU(),
            # First pooling layer: 2x2 average pooling
            nn.AvgPool2d(2, stride=2),
            # Second convolutional layer with batch normalization
            self.conv2,
            # Batch normalization
            self.bn2,
            # ReLU activation
            nn.ReLU(),
            # Second pooling layer: 2x2 average pooling
            nn.AvgPool2d(2, stride=2)
        )
        
        # Calculate the size after convolutions for the linear layer
        # Input: 28x28 -> Conv(5x5) -> 24x24 -> AvgPool(2x2) -> 12x12
        # -> Conv(5x5) -> 8x8 -> AvgPool(2x2) -> 4x4
        self.classifier = nn.Sequential(
            # Fully connected layers
            nn.Linear(16 * 4 * 4, 120),  # Changed from 16*5*5 to 16*4*4
            # ReLU activation
            nn.ReLU(),
            # Dropout layer
            nn.Dropout(dropout_rate),
            nn.Linear(120, 84),
            # ReLU activation
            nn.ReLU(),
            # Dropout layer
            nn.Dropout(dropout_rate),
            nn.Linear(84, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def train_model(model, train_loader, test_loader, optimizer_type, learning_rate, 
                epochs, model_name, save_logs=True, progress_callback=None, batch_callback=None):
    """
    Train the LeNet-ReLU-Dropout-BN model
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer_type: Type of optimizer ('Adam' or 'SGD')
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        model_name: Name to save the model
        save_logs: Whether to save training logs
    
    Returns:
        model: Trained model
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        test_accuracies: List of test accuracies
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer type")
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        epoch_start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Batch-level callback for real-time visualization
            if batch_callback and i % 10 == 0:  # 每10个batch回调一次
                batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
                batch_callback(epoch + 1, i + 1, loss.item(), batch_accuracy)
            
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_train / total_train
        epoch_time = time.time() - epoch_start_time
        
        # Evaluate on test set
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        
        # Store metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Loss: {epoch_loss:.4f}, '
              f'Train Accuracy: {epoch_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%, '
              f'Time: {epoch_time:.2f}s')
        
        # Progress callback for epoch-level progress updates
        if progress_callback:
            progress_callback(epoch + 1, epoch_loss, epoch_accuracy, test_accuracy)
    
    # Save model - commented out for manual saving
    # model_path = os.path.join('models', 'classic_model', f'{model_name}.pth')
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")
    
    return model, train_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model
        test_loader: Test data loader
    
    Returns:
        accuracy: Test accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy