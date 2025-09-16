# CNN Model Training Platform

This project provides a comprehensive platform for training and experimenting with classic Convolutional Neural Network (CNN) models on various datasets. It is designed for machine learning researchers, students, and deep learning enthusiasts who want to understand and experiment with foundational CNN architectures.

## Features

### 10 Classic CNN Models
- **LeNet Variants**:
  - LeNet (original with Sigmoid/Tanh activation)
  - LeNet-ReLU (with ReLU activation)
  - LeNet-ReLU-Dropout (with ReLU and Dropout)
  - LeNet-ReLU-Dropout-BN (with ReLU, Dropout, and Batch Normalization)
- **ResNet Series**:
  - ResNet (configurable depth)
  - ResNet-18 (18-layer residual network)
  - ResNet-34 (34-layer residual network)
- **Other Classic Models**:
  - AlexNet (ImageNet winner 2012)
  - VGG (Visual Geometry Group network)
  - NiN (Network in Network)

### Multi-Dataset Support
- MNIST (handwritten digits)
- FashionMNIST (fashion items)
- EMNIST (extended with letters and digits)
  - byclass (62 classes)
  - bymerge (47 classes)
  - balanced (47 classes)
  - letters (26 classes)
  - digits (10 classes)
  - mnist (10 classes)
- classify-leaves (leaf classification dataset)

### Dual Interface
- **Command-Line Interface (CLI)**: For quick training and experimentation
- **Graphical User Interface (GUI)**: For interactive model training and visualization

### Visualization & Management
- Real-time training metrics visualization
- Model saving and loading capabilities
- Comprehensive logging system

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Note: The project requires Python 3.x and is optimized for GPU acceleration with CUDA.

## Usage

### Command-Line Interface

```bash
python main.py --model <model_name> --dataset <dataset_name> [--other_options]
```

Example:
```bash
python main.py --model LeNet --dataset MNIST --epochs 20 --learning_rate 0.001
```

Options:
- `--model`: Model to train (LeNet, LeNet-ReLU, LeNet-ReLU-Dropout, LeNet-ReLU-Dropout-BN, ResNet, ResNet18, ResNet34, AlexNet, VGG, NiN)
- `--dataset`: Dataset to use (MNIST, FashionMNIST, EMNIST-byclass, EMNIST-bymerge, EMNIST-balanced, EMNIST-letters, EMNIST-digits, EMNIST-mnist, classify-leaves)
- `--optimizer`: Optimizer to use (Adam, SGD) - default: Adam
- `--learning_rate`: Learning rate - default: 0.001
- `--epochs`: Number of epochs - default: 10
- `--batch_size`: Batch size - default: 32
- `--model_name`: Custom model name for saving - default: auto-generated

### GUI Interface

```bash
python main_gui.py
```

The GUI provides:
- Model selection dropdown
- Dataset selection (including EMNIST subsets)
- Hyperparameter configuration
- Training control buttons
- Real-time visualization charts for loss and accuracy
- Model saving and log saving capabilities

## Project Structure

```
├── train_scripts/          # Individual training scripts for each model
├── models/classic_model/   # Saved trained models
├── logs/classic_model/     # Training logs
├── datasets/               # Downloaded datasets
├── backend.py              # Unified backend for training management
├── main.py                 # Command-line interface
├── main_gui.py             # GUI interface
├── requirements.txt        # Project dependencies
├── dataset_download.py     # Dataset downloading script
├── split_classify_leaves.py # Dataset splitting script
├── dataset_documentation.md # Dataset documentation (English)
├── dataset_documentation_zh.md # Dataset documentation (Chinese)
├── project_description.md  # Project description (English)
└── 项目说明.md             # Project description (Chinese)
```

## Models

All models have been adapted to work with grayscale images (28x28) such as MNIST and FashionMNIST. The architectures have been modified from their original versions to properly handle these smaller input sizes.

For detailed architecture descriptions, please refer to [project_description.md](project_description.md).

## Datasets

The project supports the following datasets:
- MNIST: Handwritten digit recognition
- FashionMNIST: Fashion item recognition
- EMNIST: Extended MNIST with letters and digits
- classify-leaves: Leaf classification dataset

See [dataset_documentation.md](dataset_documentation.md) for detailed information about each dataset.

## Requirements

- Python 3.x
- PyTorch 2.8.0+cu128
- TorchVision 0.23.0+cu128
- PySide6 6.9.2
- matplotlib 3.10.6
- pandas 2.3.2
- scikit-learn 1.7.2

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the platform.

## Contact

If you have any questions, suggestions, or feedback, please feel free to:
- Email: hhtrick@outlook.com
- Open an issue on the project's GitHub repository

## License

This project is for educational and research purposes.