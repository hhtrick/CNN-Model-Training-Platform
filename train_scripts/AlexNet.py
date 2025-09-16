import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

class AlexNet(nn.Module):
    """
    AlexNet implementation based on the original paper, adapted for various input sizes.
    """
    def __init__(self, num_classes=10, input_channels=1, dropout_rate=0.5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # First convolutional layer - adapted for smaller input
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=1, padding=2),  # Dynamic input channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Second convolutional layer
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Third convolutional layer
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth convolutional layer
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth convolutional layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, test_loader, optimizer_type, learning_rate, 
                epochs, model_name, save_logs=True, progress_callback=None, batch_callback=None):
    """
    Train the AlexNet model
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer_type: Type of optimizer ('Adam' or 'SGD')
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        model_name: Name to save the model
        save_logs: Whether to save training logs
        progress_callback: Optional callback function for progress updates
        batch_callback: Optional callback function for batch-level updates
    
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
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
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
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(epoch + 1, epoch_loss, epoch_accuracy, test_accuracy)
    
    # 移除自动保存模型的代码，只有用户点击保存按钮时才保存
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