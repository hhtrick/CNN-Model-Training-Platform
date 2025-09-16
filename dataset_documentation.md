# Dataset Documentation

This document provides an overview of the datasets downloaded for deep learning experiments: MNIST, FashionMNIST, and EMNIST.

## 1. MNIST Dataset

MNIST (Modified National Institute of Standards and Technology) is a large database of handwritten digits that is commonly used for training various image processing systems. It is created by "re-mixing" the samples from the original NIST dataset.

### Key Features:
- Contains 70,000 images of handwritten digits (0-9)
- Each image is a 28x28 pixel grayscale image
- Training set: 60,000 images
- Test set: 10,000 images
- Images are size-normalized and centered
- Widely used as a benchmark for image classification algorithms

### Structure:
- Each image is labeled with its corresponding digit (0-9)
- Data is divided into training and test sets
- Images have been size-normalized and centered in a 28x28 pixel area

## 2. FashionMNIST Dataset

Fashion-MNIST is a dataset of Zalando's article images consisting of 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

### Key Features:
- Created as a direct drop-in replacement for the original MNIST dataset
- Same image size and format as MNIST
- Each image is a 28x28 pixel grayscale image
- Training set: 60,000 images
- Test set: 10,000 images

### Categories:
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

### Usage:
- Provides a more challenging classification problem than MNIST
- Still simple enough for testing and debugging code
- Same image dimensions and structure as MNIST for easy model performance comparison

## 3. EMNIST Dataset

EMNIST (Extended MNIST) is an extension of the MNIST database that includes not only digits but also letters. It contains multiple subsets, each with different character combinations.

### Key Features:
- Extension of MNIST including letters and digits
- Maintains the same image format as MNIST (28x28 grayscale images)
- Provides multiple splits to accommodate different use cases

### EMNIST Splits:

#### 1. byclass
- Contains all available characters (digits and letters)
- 62 classes (10 digits + 26 lowercase letters + 26 uppercase letters)
- Training set: 697,932 images
- Test set: 116,323 images

#### 2. bymerge
- Similar to byclass but merges some characters (e.g., 'O' and '0')
- 47 classes
- Training set: 697,932 images
- Test set: 116,323 images

#### 3. balanced
- Balanced dataset with equal number of samples per class
- 47 classes
- Training set: 112,800 images (800 per class)
- Test set: 18,800 images (800 per class)

#### 4. letters
- Contains only letters (uppercase and lowercase)
- 26 classes (letters 'a' to 'z')
- Training set: 124,800 images
- Test set: 20,800 images

#### 5. digits
- Contains only digits (same as MNIST but with different splits)
- 10 classes (digits 0-9)
- Training set: 240,000 images
- Test set: 40,000 images

#### 6. mnist
- Replicates the original MNIST dataset
- 10 classes (digits 0-9)
- Training set: 60,000 images
- Test set: 10,000 images

## 4. classify-leaves Dataset

This dataset comes from the second part completion competition dataset of the "Learn Deep Learning with PyTorch" video series by Bilibili UP "Learn AI with Li Mu". The original data is from a Kaggle competition, but the original test.csv and sample_submission.csv files have been removed.

### Key Features:
- Contains 18,353 leaf images
- Covers 176 different leaf categories
- Each image is a color image with varying dimensions
- Data has been split into training and test sets (80% training, 20% testing)
- Image files are stored in the images/ directory

### Dataset Structure:
- train_split.csv: Training set containing 14,682 images
- test_split.csv: Test set containing 3,671 images
- images/: Directory containing all image files

### Usage Notes:
- Image dimensions vary and need preprocessing (such as resizing) for training
- Large number of categories makes this a challenging fine-grained classification task
- Dataset has been split using stratified sampling to ensure consistent category proportions in training and test sets

## Directory Structure

After running the download script, datasets are organized as follows:

```
datasets/
├── MNIST/
│   ├── raw/
│   └── processed/
├── FashionMNIST/
│   ├── raw/
│   └── processed/
├── EMNIST/
│   ├── byclass/
│   ├── bymerge/
│   ├── balanced/
│   ├── letters/
│   ├── digits/
│   └── mnist/
└── classify-leaves/
    ├── images/
    ├── train.csv
    ├── train_split.csv
    └── test_split.csv
```

Each dataset contains both raw and processed data. The processed data can be used directly with PyTorch's DataLoader.

## Usage Instructions

- All datasets follow the same format and can be easily loaded using PyTorch's torchvision.datasets module
- Images are 28x28 grayscale images suitable for convolutional neural networks
- Datasets are already divided into training and test sets
- Labels are provided for supervised learning tasks

## Dataset Download

Running the [dataset_download.py](dataset_download.py) script will automatically download the MNIST, FashionMNIST, and EMNIST datasets to their respective directories, which will be automatically loaded during training. For the classify-leaves dataset, please visit https://www.kaggle.com/competitions/classify-leaves to download, and extract the downloaded dataset to the [datasets\classify-leaves](datasets\classify-leaves) folder. The program will automatically read it.

After downloading the classify-leaves dataset from https://www.kaggle.com/competitions/classify-leaves and placing it in the datasets folder, please execute the split_classify_leaves.py script to perform the dataset splitting operation.

---

This document was generated by the qwen3-coder model in the Tongyi Lingma VSCode plugin.
