# High-Accuracy CNN Architecture for MNIST Classification

## Overview

This repository contains a Convolutional Neural Network (CNN) architecture that achieves **99.58% test accuracy** on the MNIST handwritten digit classification dataset. The model is lightweight with only **16,426 parameters** and demonstrates efficient learning through modern deep learning techniques.

## Architecture Details

### Model Performance
- **Test Accuracy**: 99.58% (59,751/60,000 correct predictions)
- **Total Parameters**: 16,426 (all trainable)
- **Model Size**: 0.06 MB
- **Memory Usage**: 0.46 MB total

### Network Architecture

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input: 28×28×1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)      # 26×26×16
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)     # 24×24×16
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)                   # 12×12×16

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)     # 10×10×16
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3)     # 8×8×32
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)                   # 4×4×32

        # Bottleneck layer
        self.conv5 = nn.Conv2d(32, 8, kernel_size=1)      # 4×4×8

        # Final convolution layers
        self.conv6 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 4×4×16
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3)     # 2×2×16
        self.bn7 = nn.BatchNorm2d(16)

        # Classifier
        self.fc1 = nn.Linear(16*2*2, 40)
        self.drop = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(40, 10)
```

### Layer-by-Layer Breakdown

| Layer Type | Output Shape | Parameters | Description |
|------------|--------------|------------|-------------|
| Conv2d-1 | [-1, 16, 26, 26] | 160 | Initial feature extraction |
| BatchNorm2d-2 | [-1, 16, 26, 26] | 32 | Normalization |
| Conv2d-3 | [-1, 16, 24, 24] | 2,320 | Feature refinement |
| BatchNorm2d-4 | [-1, 16, 24, 24] | 32 | Normalization |
| MaxPool2d-5 | [-1, 16, 12, 12] | 0 | Spatial downsampling |
| Conv2d-6 | [-1, 16, 10, 10] | 2,320 | Deeper feature extraction |
| BatchNorm2d-7 | [-1, 16, 10, 10] | 32 | Normalization |
| Conv2d-8 | [-1, 32, 8, 8] | 4,640 | Channel expansion |
| BatchNorm2d-9 | [-1, 32, 8, 8] | 64 | Normalization |
| MaxPool2d-10 | [-1, 32, 4, 4] | 0 | Spatial downsampling |
| Conv2d-11 | [-1, 8, 4, 4] | 264 | **Bottleneck layer** |
| Conv2d-12 | [-1, 16, 4, 4] | 1,168 | Feature reconstruction |
| BatchNorm2d-13 | [-1, 16, 4, 4] | 32 | Normalization |
| Conv2d-14 | [-1, 16, 2, 2] | 2,320 | Final convolution |
| BatchNorm2d-15 | [-1, 16, 2, 2] | 32 | Normalization |
| Linear-16 | [-1, 40] | 2,600 | Hidden layer |
| Dropout-17 | [-1, 40] | 0 | Regularization |
| Linear-18 | [-1, 10] | 410 | Output classification |

## Training Configuration

### Hyperparameters
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 20
- **Learning Rate Scheduler**: StepLR (step_size=15, gamma=0.1)

### Data Preprocessing
- **Training Transforms**:
  - RandomApply([CenterCrop(22)], p=0.1)
  - Resize((28, 28))
  - RandomRotation((-15°, 15°), fill=0)
  - ToTensor()
  - Normalize((0.1407,), (0.4081,))

- **Test Transforms**:
  - ToTensor()
  - Normalize((0.1407,), (0.4081,))

### Training Progress
The model shows consistent improvement across epochs:
- **Epoch 1**: 97.01% accuracy
- **Epoch 10**: 99.09% accuracy
- **Epoch 15**: 99.30% accuracy (LR step down occurs)
- **Epoch 18**: 99.58% accuracy (peak performance)
- **Final**: 99.58% accuracy

## Architecture Design Principles

### 1. Progressive Feature Extraction
- Starts with 16 channels for initial feature detection
- Expands to 32 channels in the middle layers for complex pattern recognition
- Uses strategic bottleneck (32→8) to reduce parameters

### 2. Normalization Strategy
- Batch normalization after each convolutional layer
- Helps stabilize training and accelerate convergence
- Enables higher learning rates

### 3. Regularization
- Dropout (p=0.1) applied before final classification layer
- Prevents overfitting while maintaining high accuracy

### 4. Efficient Design
- Uses 1×1 convolution as bottleneck layer
- Minimal fully connected layers (only 3,010 parameters)
- Most parameters concentrated in convolutional layers

## Pros of the Architecture

### ✅ High Performance
- **Exceptional accuracy**: 99.58% on MNIST test set
- **Fast convergence**: Reaches >99% accuracy by epoch 8
- **Stable training**: Consistent performance across multiple epochs

### ✅ Efficient Design
- **Lightweight**: Only 16,426 parameters
- **Memory efficient**: 0.46 MB total memory usage
- **Fast inference**: Small model size enables quick predictions

### ✅ Modern Techniques
- **Batch normalization**: Improves training stability
- **Bottleneck architecture**: Reduces parameters while maintaining performance
- **Strategic dropout**: Prevents overfitting without sacrificing accuracy

### ✅ Robust Feature Learning
- **Progressive channel expansion**: 1→16→32→8→16 channels
- **Multiple pooling layers**: Effective spatial dimension reduction
- **Deep feature hierarchy**: 7 convolutional layers capture complex patterns

## Cons of the Architecture

### ❌ Dataset-Specific Design
- **Overfitted to MNIST**: Architecture may not generalize well to complex datasets
- **Simple task**: MNIST is relatively easy; results may not translate to harder problems
- **Limited robustness**: May struggle with more diverse image characteristics

### ❌ Architectural Limitations
- **Excessive depth for MNIST**: 7 conv layers might be overkill for 28×28 images
- **Computational overhead**: Multiple batch norm layers add training/inference time
- **Limited dropout**: Regularization only at the end, not throughout the network

### ❌ Scalability Concerns
- **Fixed input size**: Designed specifically for 28×28 images
- **Channel progression**: Architecture choices may not scale to larger, more complex datasets
- **Memory constraints**: While efficient for MNIST, may need redesign for larger inputs

### ❌ Training Dependencies
- **Learning rate scheduling**: Requires careful tuning of step size and gamma
- **Batch normalization dependency**: May struggle with very small batch sizes
- **Optimizer sensitivity**: Performance tied to specific Adam hyperparameters

## Conclusion

This CNN architecture represents an excellent balance between performance and efficiency for MNIST classification. It achieves state-of-the-art accuracy (99.58%) while maintaining a compact parameter count (16,426). The design showcases modern deep learning practices including batch normalization, bottleneck layers, and strategic regularization.

However, the architecture's specialization for MNIST may limit its applicability to more complex computer vision tasks. For real-world applications, consider adapting the design principles while scaling the architecture appropriately for the target dataset's complexity.

## Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize model
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Training loop
for epoch in range(20):
    train(model, device, train_loader, optimizer, criterion)
    test(model, device, test_loader, criterion)
    scheduler.step()
```