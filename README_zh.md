# CNN 模型训练平台

本项目提供了一个全面的平台，用于在各种数据集上训练和实验经典的卷积神经网络(CNN)模型。该项目专为机器学习研究人员、学生和深度学习爱好者设计，帮助他们理解和实验基础的CNN架构。

## 功能特点

### 10种经典CNN模型
- **LeNet系列**:
  - LeNet (原始版本，使用Sigmoid/Tanh激活函数)
  - LeNet-ReLU (使用ReLU激活函数)
  - LeNet-ReLU-Dropout (使用ReLU和Dropout)
  - LeNet-ReLU-Dropout-BN (使用ReLU、Dropout和批归一化)
- **ResNet系列**:
  - ResNet (可配置深度)
  - ResNet-18 (18层残差网络)
  - ResNet-34 (34层残差网络)
- **其他经典模型**:
  - AlexNet (2012年ImageNet竞赛冠军)
  - VGG (视觉几何组网络)
  - NiN (网络中的网络)

### 多数据集支持
- MNIST (手写数字)
- FashionMNIST (时尚物品)
- EMNIST (扩展版，包含字母和数字)
  - byclass (62个类别)
  - bymerge (47个类别)
  - balanced (47个类别)
  - letters (26个类别)
  - digits (10个类别)
  - mnist (10个类别)
- classify-leaves (叶子分类数据集)

### 双界面操作
- **命令行界面(CLI)**: 快速训练和实验
- **图形用户界面(GUI)**: 交互式模型训练和可视化

### 可视化与管理
- 实时训练指标可视化
- 模型保存和加载功能
- 全面的日志系统

## 安装说明

1. 安装所需依赖:
```bash
pip install -r requirements.txt
```

注意: 项目需要Python 3.x，并针对CUDA优化了GPU加速。

## 使用方法

### 命令行界面

```bash
python main.py --model <模型名称> --dataset <数据集名称> [--其他选项]
```

示例:
```bash
python main.py --model LeNet --dataset MNIST --epochs 20 --learning_rate 0.001
```

选项:
- `--model`: 要训练的模型 (LeNet, LeNet-ReLU, LeNet-ReLU-Dropout, LeNet-ReLU-Dropout-BN, ResNet, ResNet18, ResNet34, AlexNet, VGG, NiN)
- `--dataset`: 使用的数据集 (MNIST, FashionMNIST, EMNIST-byclass, EMNIST-bymerge, EMNIST-balanced, EMNIST-letters, EMNIST-digits, EMNIST-mnist, classify-leaves)
- `--optimizer`: 使用的优化器 (Adam, SGD) - 默认: Adam
- `--learning_rate`: 学习率 - 默认: 0.001
- `--epochs`: 训练轮数 - 默认: 10
- `--batch_size`: 批次大小 - 默认: 32
- `--model_name`: 自定义模型保存名称 - 默认: 自动生成

### 图形界面

```bash
python main_gui.py
```

图形界面提供以下功能:
- 模型选择下拉框
- 数据集选择 (包括EMNIST子集)
- 超参数配置
- 训练控制按钮
- 损失和准确率的实时可视化图表
- 模型和日志保存功能

## 项目结构

```
├── train_scripts/          # 各个模型的独立训练脚本
├── models/classic_model/   # 保存训练好的模型
├── logs/classic_model/     # 训练日志
├── datasets/               # 下载的数据集
├── backend.py              # 统一的训练管理后端
├── main.py                 # 命令行接口
├── main_gui.py             # 图形用户界面
├── requirements.txt        # 项目依赖
├── dataset_download.py     # 数据集下载脚本
├── split_classify_leaves.py # 数据集分割脚本
├── dataset_documentation.md # 数据集文档 (英文)
├── dataset_documentation_zh.md # 数据集文档 (中文)
├── project_description.md  # 项目说明 (英文)
└── 项目说明.md             # 项目说明 (中文)
```

## 模型说明

所有模型都经过调整，可以处理MNIST和FashionMNIST等灰度图像(28x28)。这些架构是从原始版本修改而来，以适配较小的输入尺寸。

详细的架构描述，请参考 [项目说明.md](项目说明.md)。

## 数据集

项目支持以下数据集:
- MNIST: 手写数字识别
- FashionMNIST: 时尚物品识别
- EMNIST: 扩展的MNIST，包含字母和数字
- classify-leaves: 叶子分类数据集

有关每个数据集的详细信息，请参见 [dataset_documentation_zh.md](dataset_documentation_zh.md)。

## 系统要求

- Python 3.x
- PyTorch 2.8.0+cu128
- TorchVision 0.23.0+cu128
- PySide6 6.9.2
- matplotlib 3.10.6
- pandas 2.3.2
- scikit-learn 1.7.2

## 贡献

欢迎贡献！请随时提交问题或拉取请求来改进平台。

## 联系方式

如果您有任何问题、建议或反馈，请随时：
- 发送邮件至: hhtrick@outlook.com
- 在项目的GitHub仓库中提交issue

## 许可证

本项目仅供教育和研究使用。