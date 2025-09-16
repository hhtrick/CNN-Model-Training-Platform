# CNN Model Training Platform Project Description

This is a project designed for experiencing deep learning, primarily focused on experimenting with convolutional neural network models and tuning parameters. The following is the project description:

## 1. Datasets:
1. MNIST handwritten digit dataset
2. EMNIST dataset containing handwritten digits and letters
3. FashionMNIST clothing dataset
4. classify-leaves dataset: Taken from the second part completion competition dataset of the "Learn Deep Learning with PyTorch" video series by Bilibili UP "Learn AI with Li Mu". Since the Kaggle competition has ended, the original test.csv and sample_submission.csv files have been removed, and the samples in train.csv have been split to obtain training and test sets.

For detailed introduction to datasets, please refer to the dataset_documentation.md document.

## 2. Convolutional Neural Network Architecture:

### Classic Models
This section includes some classic convolutional neural network models.

#### 1.1 LeNet
LeNet is a convolutional neural network proposed by Yann LeCun et al. in 1998 for handwritten digit recognition. It is one of the earliest convolutional neural networks.
- Layer 1: Convolutional layer (C1), using 6 5×5 convolutional kernels, output feature map size 28×28×6
- Layer 2: Pooling layer (S2), using 2×2 average pooling, stride 2, output feature map size 14×14×6
- Layer 3: Convolutional layer (C3), using 16 5×5 convolutional kernels, output feature map size 10×10×16
- Layer 4: Pooling layer (S4), using 2×2 average pooling, stride 2, output feature map size 5×5×16
- Layer 5: Fully connected layer (F5), 16×5×5 input, 120 neuron output
- Layer 6: Fully connected layer (F6), 120 neuron input, 84 neuron output
- Layer 7: Output layer, 84 neuron input, output corresponding to number of classes
Activation function: Sigmoid or Tanh

#### 1.2 LeNet-ReLU
LeNet-ReLU is an improved version of LeNet, replacing the original Sigmoid/Tanh activation functions with ReLU activation functions, which can alleviate the gradient vanishing problem and speed up training.
- Layer 1: Convolutional layer (C1), using 6 5×5 convolutional kernels, output feature map size 28×28×6
- Layer 2: ReLU activation function
- Layer 3: Pooling layer (S2), using 2×2 average pooling, stride 2, output feature map size 14×14×6
- Layer 4: Convolutional layer (C3), using 16 5×5 convolutional kernels, output feature map size 10×10×16
- Layer 5: ReLU activation function
- Layer 6: Pooling layer (S4), using 2×2 average pooling, stride 2, output feature map size 5×5×16
- Layer 7: Fully connected layer (F5), 16×5×5 input, 120 neuron output
- Layer 8: ReLU activation function
- Layer 9: Fully connected layer (F6), 120 neuron input, 84 neuron output
- Layer 10: ReLU activation function
- Layer 11: Output layer, 84 neuron input, output corresponding to number of classes

#### 1.3 LeNet-ReLU-Dropout
Based on LeNet-ReLU, a Dropout layer is added between the fully connected layers to prevent overfitting and improve the model's generalization ability.
- Layer 1: Convolutional layer (C1), using 6 5×5 convolutional kernels, output feature map size 28×28×6
- Layer 2: ReLU activation function
- Layer 3: Pooling layer (S2), using 2×2 average pooling, stride 2, output feature map size 14×14×6
- Layer 4: Convolutional layer (C3), using 16 5×5 convolutional kernels, output feature map size 10×10×16
- Layer 5: ReLU activation function
- Layer 6: Pooling layer (S4), using 2×2 average pooling, stride 2, output feature map size 5×5×16
- Layer 7: Fully connected layer (F5), 16×5×5 input, 120 neuron output
- Layer 8: ReLU activation function
- Layer 9: Dropout layer, dropout rate default 0.5
- Layer 10: Fully connected layer (F6), 120 neuron input, 84 neuron output
- Layer 11: ReLU activation function
- Layer 12: Dropout layer, dropout rate default 0.5
- Layer 13: Output layer, 84 neuron input, output corresponding to number of classes

#### 1.4 LeNet-ReLU-Dropout-BN
Based on LeNet-ReLU-Dropout, Batch Normalization layers are added, which can accelerate the training process and improve model stability.
- Layer 1: Convolutional layer (C1), using 6 5×5 convolutional kernels, output feature map size 28×28×6
- Layer 2: Batch normalization layer (BN)
- Layer 3: ReLU activation function
- Layer 4: Pooling layer (S2), using 2×2 average pooling, stride 2, output feature map size 14×14×6
- Layer 5: Convolutional layer (C3), using 16 5×5 convolutional kernels, output feature map size 10×10×16
- Layer 6: Batch normalization layer (BN)
- Layer 7: ReLU activation function
- Layer 8: Pooling layer (S4), using 2×2 average pooling, stride 2, output feature map size 5×5×16
- Layer 9: Fully connected layer (F5), 16×5×5 input, 120 neuron output
- Layer 10: ReLU activation function
- Layer 11: Dropout layer, dropout rate default 0.5
- Layer 12: Fully connected layer (F6), 120 neuron input, 84 neuron output
- Layer 13: ReLU activation function
- Layer 14: Dropout layer, dropout rate default 0.5
- Layer 15: Output layer, 84 neuron input, output corresponding to number of classes

#### 1.5 ResNet
ResNet (Residual Network) is a deep residual network proposed by Kaiming He et al. at Microsoft Research in 2015. It solved the problem of training difficulties in deep networks by introducing residual connections.
- Layer 1: Convolutional layer, using 7×7 convolutional kernel, 64 output channels, stride 2, padding 3
- Layer 2: Batch normalization layer (BN)
- Layer 3: ReLU activation function
- Layer 4: Max pooling layer, 3×3 pooling window, stride 2, padding 1
- Layer 5: Residual block layer (layer1), containing a specified number of residual blocks
- Layer 6: Residual block layer (layer2), containing a specified number of residual blocks, with downsampling
- Layer 7: Residual block layer (layer3), containing a specified number of residual blocks, with downsampling
- Layer 8: Residual block layer (layer4), containing a specified number of residual blocks, with downsampling
- Layer 9: Global average pooling layer, output size 1×1
- Layer 10: Fully connected layer, output corresponding to number of classes

Residual block structure:
- Main path: Convolutional layer → Batch normalization layer → ReLU activation function → Convolutional layer → Batch normalization layer
- Shortcut connection: If input and output dimensions are inconsistent, use 1×1 convolution to adjust dimensions
- Final output: Main path output and shortcut connection output are added together and then passed through ReLU activation function

#### 1.6 ResNet-18
ResNet-18 is one of the ResNet series, containing 18 layers with weights, built using basic residual blocks (BasicBlock). The structure is relatively simple but effective.
- Layer 1: Convolutional layer, using 7×7 convolutional kernel, 64 output channels, stride 2, padding 3
- Layer 2: Batch normalization layer (BN)
- Layer 3: ReLU activation function
- Layer 4: Max pooling layer, 3×3 pooling window, stride 2, padding 1
- Layer 5: Residual block layer (layer1), containing 2 residual blocks, each block has 2 3×3 convolutional layers, output channels 64
- Layer 6: Residual block layer (layer2), containing 2 residual blocks, each block has 2 3×3 convolutional layers, output channels 128, with downsampling
- Layer 7: Residual block layer (layer3), containing 2 residual blocks, each block has 2 3×3 convolutional layers, output channels 256, with downsampling
- Layer 8: Residual block layer (layer4), containing 2 residual blocks, each block has 2 3×3 convolutional layers, output channels 512, with downsampling
- Layer 9: Global average pooling layer, output size 1×1
- Layer 10: Fully connected layer, output corresponding to number of classes (512→num_classes)

#### 1.7 ResNet-34
ResNet-34 is another in the ResNet series, containing 34 layers with weights, also built using basic residual blocks, deeper than ResNet-18.
- Layer 1: Convolutional layer, using 7×7 convolutional kernel, 64 output channels, stride 2, padding 3
- Layer 2: Batch normalization layer (BN)
- Layer 3: ReLU activation function
- Layer 4: Max pooling layer, 3×3 pooling window, stride 2, padding 1
- Layer 5: Residual block layer (layer1), containing 3 residual blocks, each block has 2 3×3 convolutional layers, output channels 64
- Layer 6: Residual block layer (layer2), containing 4 residual blocks, each block has 2 3×3 convolutional layers, output channels 128, with downsampling
- Layer 7: Residual block layer (layer3), containing 6 residual blocks, each block has 2 3×3 convolutional layers, output channels 256, with downsampling
- Layer 8: Residual block layer (layer4), containing 3 residual blocks, each block has 2 3×3 convolutional layers, output channels 512, with downsampling
- Layer 9: Global average pooling layer, output size 1×1
- Layer 10: Fully connected layer, output corresponding to number of classes (512→num_classes)

#### 1.8 AlexNet
AlexNet is a deep convolutional neural network proposed by Alex Krizhevsky et al. in 2012, which achieved breakthrough results in the ImageNet competition. It contains 5 convolutional layers and 3 fully connected layers, using ReLU activation functions and Dropout techniques.
- Layer 1: Convolutional layer, using 96 11×11 convolutional kernels, stride 4, padding 2
- Layer 2: ReLU activation function
- Layer 3: Max pooling layer, 3×3 pooling window, stride 2
- Layer 4: Convolutional layer, using 256 5×5 convolutional kernels, padding 2
- Layer 5: ReLU activation function
- Layer 6: Max pooling layer, 3×3 pooling window, stride 2
- Layer 7: Convolutional layer, using 384 3×3 convolutional kernels, padding 1
- Layer 8: ReLU activation function
- Layer 9: Convolutional layer, using 384 3×3 convolutional kernels, padding 1
- Layer 10: ReLU activation function
- Layer 11: Convolutional layer, using 256 3×3 convolutional kernels, padding 1
- Layer 12: ReLU activation function
- Layer 13: Max pooling layer, 3×3 pooling window, stride 2
- Layer 14: Adaptive average pooling layer, output size 6×6
- Layer 15: Dropout layer, dropout rate 0.5
- Layer 16: Fully connected layer, output 4096 neurons
- Layer 17: ReLU activation function
- Layer 18: Dropout layer, dropout rate 0.5
- Layer 19: Fully connected layer, output 4096 neurons
- Layer 20: ReLU activation function
- Layer 21: Fully connected layer, output corresponding to number of classes

#### 1.9 VGG
VGG is a convolutional neural network proposed by the Visual Geometry Group at Oxford University, famous for its simple and unified structure. VGG deepens network depth by stacking multiple small convolutional kernels (3×3) instead of large convolutional kernels.
- Group 1: Convolutional layer (64 3×3 kernels) → ReLU → Convolutional layer (64 3×3 kernels) → ReLU → Max pooling layer (2×2)
- Group 2: Convolutional layer (128 3×3 kernels) → ReLU → Convolutional layer (128 3×3 kernels) → ReLU → Max pooling layer (2×2)
- Group 3: Convolutional layer (256 3×3 kernels) → ReLU → Convolutional layer (256 3×3 kernels) → ReLU → Convolutional layer (256 3×3 kernels) → ReLU → Max pooling layer (2×2)
- Group 4: Convolutional layer (512 3×3 kernels) → ReLU → Convolutional layer (512 3×3 kernels) → ReLU → Convolutional layer (512 3×3 kernels) → ReLU → Max pooling layer (2×2)
- Group 5: Convolutional layer (512 3×3 kernels) → ReLU → Convolutional layer (512 3×3 kernels) → ReLU → Convolutional layer (512 3×3 kernels) → ReLU → Max pooling layer (2×2)
- Adaptive average pooling layer, output size 7×7
- Fully connected layer (512×7×7→4096) → ReLU → Dropout → Fully connected layer (4096→4096) → ReLU → Dropout → Fully connected layer (4096→num_classes)

#### 1.10 NiN
NiN (Network in Network) is a network structure proposed by Min Lin et al. in 2013. It innovatively uses multilayer perceptrons (MLP) inside convolutional layers to replace traditional convolution operations, and uses global average pooling instead of fully connected layers to reduce parameters.
- Layer 1: NiN block, input channels 1, output channels 96, convolutional kernel 11×11, stride 4, no padding → Max pooling layer (3×3, stride 2)
- Layer 2: NiN block, input channels 96, output channels 256, convolutional kernel 5×5, stride 1, padding 2 → Max pooling layer (3×3, stride 2)
- Layer 3: NiN block, input channels 256, output channels 384, convolutional kernel 3×3, stride 1, padding 1 → Max pooling layer (3×3, stride 2) → Dropout layer (0.5)
- Layer 4: NiN block, input channels 384, output channels num_classes, convolutional kernel 3×3, stride 1, padding 1
- Global average pooling layer, output size 1×1
- Flatten layer, output number of classes

NiN block structure:
- Convolutional layer → ReLU → 1×1 convolutional layer → ReLU → 1×1 convolutional layer → ReLU

### Special Note
The "## 1. Classic Models" section documentation was generated by the qwen3-coder model in the Tongyi Lingma VSCode plugin.

## 2. Custom Models

remain to develop
