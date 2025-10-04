# CIFAR-10 CNN Model - Session 7 Assignment

This repository contains a Convolutional Neural Network (CNN) implementation for CIFAR-10 image classification with advanced techniques including dilated convolution, depthwise separable convolution, and data augmentation.

## Model Architecture

### Overview
The model is designed with 4 main convolution blocks (C1-C4) and an output layer, achieving a total of **193,280 trainable parameters** with a final receptive field of **44**.

### Architecture Details

| Block | Layer Type | Input Size | In Channels | Kernel | Stride | Padding | Dilation | Output Size | Out Channels | RF |
|-------|-----------|-----------|-------------|--------|--------|---------|----------|-------------|--------------|-----|
| **C1** | Conv2d | 32×32 | 3 | 3×3 | 1 | 0 | 1 | 30×30 | 32 | 3 |
| C1 | Conv2d | 30×30 | 32 | 3×3 | 1 | 0 | 1 | 28×28 | 64 | 5 |
| C1 | Conv2d (Dilated) | 28×28 | 64 | 3×3 | 1 | 0 | 2 | 24×24 | 64 | 9 |
| **C2** | Conv2d (1×1) | 24×24 | 64 | 1×1 | 1 | 0 | 1 | 24×24 | 32 | 9 |
| C2 | Conv2d (Depthwise) | 24×24 | 32 | 3×3 | 1 | 1 | 1 | 24×24 | 32 | 11 |
| C2 | Conv2d (Pointwise) | 24×24 | 32 | 1×1 | 1 | 0 | 1 | 24×24 | 64 | 11 |
| C2 | Conv2d | 24×24 | 64 | 3×3 | 1 | 0 | 1 | 22×22 | 64 | 13 |
| **C3** | Conv2d (Strided) | 22×22 | 64 | 3×3 | 2 | 0 | 1 | 10×10 | 64 | 17 |
| C3 | Conv2d (1×1) | 10×10 | 64 | 1×1 | 1 | 0 | 1 | 10×10 | 32 | 17 |
| C3 | AvgPool2d | 10×10 | 32 | 2×2 | 2 | 0 | - | 5×5 | 32 | 21 |
| **C4** | Conv2d | 5×5 | 32 | 3×3 | 1 | 1 | 1 | 5×5 | 64 | 25 |
| C4 | Conv2d | 5×5 | 64 | 3×3 | 1 | 1 | 1 | 5×5 | 64 | 29 |
| C4 | AvgPool2d | 5×5 | 64 | 5×5 | - | 0 | - | 1×1 | 64 | 44 |
| **Output** | Conv2d (1×1) | 1×1 | 64 | 1×1 | 1 | 0 | 1 | 1×1 | 10 | 44 |

### Key Architectural Features

1. **Dilated Convolution (C1)**: Uses dilation=2 to expand receptive field without increasing parameters
2. **Depthwise Separable Convolution (C2)**: Reduces computational cost while maintaining performance
   - Depthwise convolution (groups=32)
   - Pointwise convolution (1×1)
3. **Strided Convolution (C3)**: Reduces spatial dimensions efficiently (stride=2)
4. **Global Average Pooling**: Final 5×5 pooling reduces to 1×1
5. **Regularization**: Batch Normalization + Dropout (0.05) after each convolution

### Model Summary
```
Total params: 193,280
Trainable params: 193,280
Non-trainable params: 0
Input size (MB): 0.01
Forward/backward pass size (MB): 6.49
Params size (MB): 0.74
Estimated Total Size (MB): 7.24
```

## Training Configuration

### Hyperparameters
- **Batch Size**: 128
- **Epochs**: 30
- **Optimizer**: SGD
  - Learning Rate: 0.01
  - Momentum: 0.9
  - Weight Decay: 1e-4
- **Scheduler**: OneCycleLR
  - Max LR: 0.05
  - Steps per epoch: 391

### Data Augmentation
The model uses Albumentations library for advanced augmentation:
- **Horizontal Flip** (p=0.5)
- **ShiftScaleRotate** (shift=0.0625, scale=0.1, rotate=7°, p=0.5)
- **CoarseDropout** (Cutout - 16×16 holes, p=0.5)
- **Normalization** (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

## Training Results

### Epoch-by-Epoch Performance

| Epoch | Train Loss | Train Acc (%) | Test Loss | Test Acc (%) |
|-------|-----------|---------------|-----------|--------------|
| 1 | 1.3913 | 40.41 | 1.3186 | 52.44 |
| 2 | 1.2331 | 56.97 | 1.0554 | 61.49 |
| 3 | 0.8808 | 63.13 | 0.9094 | 67.60 |
| 4 | 0.8719 | 67.12 | 0.8497 | 70.52 |
| 5 | 0.8952 | 69.96 | 0.7691 | 73.23 |
| 6 | 0.7794 | 72.15 | 0.7395 | 74.18 |
| 7 | 0.6152 | 74.20 | 0.6706 | 76.95 |
| 8 | 0.7819 | 75.54 | 0.6531 | 76.91 |
| 9 | 0.5014 | 76.57 | 0.6137 | 78.84 |
| 10 | 0.4678 | 77.93 | 0.6047 | 79.14 |
| 11 | 0.5623 | 78.72 | 0.5757 | 80.00 |
| 12 | 0.4634 | 79.54 | 0.5469 | 81.13 |
| 13 | 0.6944 | 80.23 | 0.5461 | 81.15 |
| 14 | 0.6877 | 81.02 | 0.5380 | 81.62 |
| 15 | 0.6356 | 81.60 | 0.5104 | 82.40 |
| 16 | 0.6250 | 82.57 | 0.5060 | 82.63 |
| 17 | 0.4750 | 82.70 | 0.5177 | 82.04 |
| 18 | 0.7391 | 83.17 | 0.4809 | 83.31 |
| 19 | 0.5653 | 83.86 | 0.4954 | 82.93 |
| 20 | 0.5299 | 84.06 | 0.4788 | 83.57 |
| 21 | 0.4953 | 84.64 | 0.4510 | 84.69 |
| 22 | 0.3112 | 85.05 | 0.4607 | 84.57 |
| 23 | 0.3844 | 85.98 | 0.4402 | 85.05 |
| 24 | 0.3188 | 86.47 | 0.4304 | 85.37 |
| 25 | 0.3145 | 86.90 | 0.4145 | 86.09 |
| 26 | 0.3516 | 87.85 | 0.4023 | 86.48 |
| 27 | 0.3470 | 88.51 | 0.3902 | 86.95 |
| 28 | 0.3746 | 88.85 | 0.3882 | 87.16 |
| 29 | 0.2913 | 89.14 | 0.3877 | 87.37 |
| 30 | 0.4142 | 89.29 | 0.3889 | **87.28** |

### Final Performance
- **Final Test Accuracy**: 87.28%
- **Best Test Accuracy**: 87.37% (Epoch 29)
- **Final Train Accuracy**: 89.29%

## Results Analysis

### Learning Progression

1. **Initial Phase (Epochs 1-5)**: Rapid learning
   - Test accuracy improved from 52.44% to 73.23%
   - Sharp decrease in both train and test loss
   - Model quickly learns basic patterns

2. **Growth Phase (Epochs 6-15)**: Steady improvement
   - Test accuracy: 74.18% → 82.40%
   - Consistent reduction in test loss
   - Learning rate increases during OneCycleLR ramp-up

3. **Refinement Phase (Epochs 16-25)**: Gradual gains
   - Test accuracy: 82.63% → 86.09%
   - Smaller incremental improvements
   - Model fine-tunes learned features

4. **Plateau Phase (Epochs 26-30)**: Convergence
   - Test accuracy stabilizes around 86-87%
   - Minor fluctuations in performance
   - Best performance at epoch 29 (87.37%)

### Key Observations

#### Strengths
- **Strong Generalization**: Small gap between train (89.29%) and test (87.28%) accuracy indicates good generalization
- **Effective Regularization**: Dropout and data augmentation prevent overfitting
- **Optimal Architecture**: 193K parameters achieve competitive performance without being overparameterized
- **Stable Training**: OneCycleLR provides smooth convergence

#### Performance Characteristics
- **Training Accuracy**: Reaches 89.29% by epoch 30
- **Test Accuracy**: Peaks at 87.37% (epoch 29), final 87.28%
- **Generalization Gap**: ~2% difference between train and test accuracy
- **Loss Trends**: Both training and test losses decrease consistently

#### Potential Improvements
1. **Extended Training**: Could potentially reach higher accuracy with more epochs
2. **Learning Rate Tuning**: Fine-tune max_lr in OneCycleLR scheduler
3. **Architecture Tweaks**: Experiment with channel sizes or additional layers
4. **Augmentation**: Try additional augmentation techniques (MixUp, CutMix)

## Training Visualizations

### Loss Curves
The training process shows consistent decrease in both training and testing loss:
- **Training Loss**: 1.39 → 0.41 (drops ~70%)
- **Testing Loss**: 1.32 → 0.39 (drops ~70%)

Both curves follow similar trajectories, indicating healthy learning without severe overfitting.

### Accuracy Curves
Accuracy steadily improves across both sets:
- **Training Accuracy**: 40.41% → 89.29% (+48.88%)
- **Testing Accuracy**: 52.44% → 87.28% (+34.84%)

The gap between training and testing accuracy remains small throughout training, demonstrating effective regularization.

> **Note**: Detailed ASCII visualizations of loss and accuracy curves are available in the training notebook output. For graphical plots, run the training script which generates `training_metrics_model.png`.

## Hardware & Training Time
- **Device**: CUDA (Tesla T4 GPU on Google Colab)
- **Training Speed**: ~11.5 iterations/second
- **Time per Epoch**: ~34 seconds
- **Total Training Time**: ~17 minutes (30 epochs)

## Repository Structure
```
session7-ass/
├── model.py              # Model architecture and data augmentation
├── train.py              # Training script
├── train.ipynb           # Jupyter notebook for Colab training
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

### Training
```python
from train import CIFARTrainer

trainer = CIFARTrainer(
    model_module_name='model',
    epochs=30,
    batch_size=128
)
final_accuracy = trainer.run()
```

### Model Loading
```python
from model import Net

model = Net()
model.load_state_dict(torch.load('model.pth'))
```

## Requirements
```
torch>=2.0.0
torchvision
albumentations
numpy
tqdm
torchsummary
plotille
```

## Conclusion

This CIFAR-10 CNN model demonstrates effective use of advanced convolution techniques (dilated, depthwise separable, strided) combined with strong regularization strategies. The model achieves **87.28% test accuracy** with only **193K parameters**, showing excellent parameter efficiency.

The training process exhibits:
- Healthy learning curves with consistent improvement
- Minimal overfitting (2% generalization gap)
- Stable convergence with OneCycleLR scheduling
- Effective data augmentation strategies

This architecture serves as a strong baseline for CIFAR-10 classification while maintaining computational efficiency.
