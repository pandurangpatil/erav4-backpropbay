# MNIST CNN Models - Session 6 Assignment

This repository contains three CNN models for MNIST digit classification, each designed to explore different architectural concepts and training techniques to achieve 99.4% test accuracy with less than 8K parameters.

## Overview

| Model | Parameters | Best Train Acc | Best Test Acc | Key Features |
|-------|------------|----------------|---------------|--------------|
| Model 1 | 7,108 | 99.63% | 99.06% | Basic CNN + BatchNorm |
| Model 2 | ~6,000 | 99.00% | 98.99% | GAP + Dropout Regularization |
| Model 3 | ~7,000 | 99.06% | 99.50% | Data Aug + OneCycleLR + Weight Decay |

---

## Model 1: Baseline CNN

### Files
- **Model Definition**: [model1.py](./model1.py)
- **Training Notebook**: [train_model1.ipynb](./train_model1.ipynb)
- **Google Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pandurangpatil/erav4-backpropbay/blob/main/session6-ass/train_model1.ipynb)

### Target
1. Basic modular code structure with .py files
2. NN with parameters less than 8K
3. Check with Batch Normalization if we can hit 99.4% test accuracy

### Architecture
```
Input (1x28x28) → Conv2d(1→4, 3x3) → BN → ReLU → [26x26x4]
                ↓
Conv2d(4→4, 3x3) → BN → ReLU → [24x24x4]
                ↓
Conv2d(4→8, 3x3) → BN → ReLU → [22x22x8]
                ↓
MaxPool2d(2x2) → [11x11x8]
                ↓
Conv2d(8→4, 1x1) → BN → ReLU → [11x11x4]  # Transition
                ↓
Conv2d(4→8, 3x3) → BN → ReLU → [9x9x8]
                ↓
Conv2d(8→16, 3x3) → BN → ReLU → [7x7x16]
                ↓
Conv2d(16→10, 1x1) → BN → ReLU → [7x7x10]  # Channel reduction
                ↓
Conv2d(10→10, 7x7) → [1x1x10]  # Final conv acts as FC
                ↓
LogSoftmax → Output
```

**Total Parameters**: 7,108

### Training Configuration
- **Optimizer**: SGD (lr=0.01, momentum=0.9)
- **Scheduler**: None
- **Data Augmentation**: None (basic normalization only)
- **Epochs**: 20
- **Batch Size**: 128

### Results
- **Parameters**: 7,108
- **Best Training Accuracy**: 99.63%
- **Best Test Accuracy**: 99.06%

### Analysis
1. **Good Foundation**: Model demonstrates solid CNN architecture principles with proper use of BatchNorm
2. **Overfitting Issue**: Clear gap between training (99.63%) and test (99.06%) accuracy
3. **Plateau Effect**: Test accuracy consistently stays below 99.1%, indicating model limitations
4. **Training Behavior**: Model consistently hits 99%+ training accuracy after epoch 7, but test accuracy plateaus
5. **Architecture Efficiency**: Uses 1x1 convolutions for channel reduction and transition blocks
6. **Missing Regularization**: No dropout or data augmentation leads to overfitting

---

## Model 2: Regularization Focus

### Files
- **Model Definition**: [model2.py](./model2.py)
- **Training Notebook**: [train_model2.ipynb](./train_model2.ipynb)
- **Google Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pandurangpatil/erav4-backpropbay/blob/main/session6-ass/train_model2.ipynb)

### Target
1. Use regularization to reduce/remove overfitting from previous run
2. Use GAP (Global Average Pooling) to reduce number of parameters
3. Add more layers to increase capacity while maintaining parameter efficiency

### Architecture
```
Input (1x28x28) → Conv2d(1→8, 3x3) → BN → ReLU → [26x26x8]
                ↓
Conv2d(8→8, 3x3) → BN → ReLU → [24x24x8]
                ↓
Conv2d(8→16, 3x3) → BN → ReLU → [22x22x16]
                ↓
Dropout(0.25) → MaxPool2d(2x2) → [11x11x16]
                ↓
Conv2d(16→8, 1x1) → BN → ReLU → [11x11x8]  # Transition
                ↓
Conv2d(8→8, 3x3) → BN → ReLU → [9x9x8]
                ↓
Conv2d(8→16, 3x3) → BN → ReLU → [7x7x16]
                ↓
Dropout(0.25) → Conv2d(16→16, 3x3) → BN → ReLU → [5x5x16]
                ↓
Conv2d(16→10, 1x1) → [5x5x10]  # Channel reduction
                ↓
AvgPool2d(5x5) → [1x1x10]  # GAP equivalent
                ↓
LogSoftmax → Output
```

**Total Parameters**: ~6,000

### Training Configuration
- **Optimizer**: SGD (lr=0.01, momentum=0.9)
- **Scheduler**: None
- **Regularization**: Dropout (0.25)
- **Data Augmentation**: None
- **Epochs**: 15
- **Batch Size**: 128

### Results
- **Parameters**: ~6,000
- **Best Training Accuracy**: 99.00%
- **Best Test Accuracy**: 98.99%

### Analysis
1. **Successful Overfitting Reduction**: Training and test accuracy are now very close (99.00% vs 98.99%)
2. **GAP Implementation**: Global Average Pooling reduces parameters while maintaining spatial information
3. **Effective Regularization**: Dropout layers prevent overfitting without significant accuracy loss
4. **Parameter Efficiency**: Reduced parameters by ~1K while maintaining competitive performance
5. **Balanced Learning**: No significant gap between training and validation performance
6. **Architecture Trade-off**: More regularization leads to slightly lower peak accuracy but better generalization

---

## Model 3: Optimized Performance

### Files
- **Model Definition**: [model3.py](./model3.py)
- **Training Notebook**: [train_model3.ipynb](./train_model3.ipynb)
- **Google Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pandurangpatil/erav4-backpropbay/blob/main/session6-ass/train_model3.ipynb)

### Target
1. Hit target of achieving 99.4% test accuracy with parameters less than 8K in 15 epochs
2. Implement advanced training techniques for optimal performance

### Architecture
```
Input (1x28x28) → Conv2d(1→8, 3x3) → ReLU → BN → Dropout(0.05) → [26x26x8]
                ↓
Conv2d(8→16, 3x3) → ReLU → BN → [24x24x16]
                ↓
Conv2d(16→8, 1x1) → [24x24x8]  # Transition
                ↓
MaxPool2d(2x2) → [12x12x8]
                ↓
Conv2d(8→16, 3x3) → ReLU → BN → [10x10x16]
                ↓
Conv2d(16→16, 3x3) → ReLU → BN → [8x8x16]
                ↓
Conv2d(16→16, 3x3, pad=1) → ReLU → BN → [8x8x16]  # Maintain spatial size
                ↓
AdaptiveAvgPool2d(1) → [1x1x16]  # GAP
                ↓
Conv2d(16→10, 1x1) → [1x1x10]
                ↓
LogSoftmax → Output
```

**Total Parameters**: ~7,000

### Training Configuration
- **Optimizer**: SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)
- **Scheduler**: OneCycleLR (max_lr=0.05, 20 epochs)
- **Data Augmentation**:
  - RandomRotation(±7°)
  - RandomAffine (translate=0.1)
- **Regularization**: Minimal dropout (0.05), Weight decay
- **Epochs**: 20
- **Batch Size**: 128

### Results
- **Parameters**: ~7,000
- **Best Training Accuracy**: 99.06%
- **Best Test Accuracy**: 99.50%

### Analysis
1. **Target Achievement**: Successfully achieved 99.4%+ test accuracy (99.50%)
2. **Optimal Regularization**: Balanced approach with minimal dropout + weight decay + data augmentation
3. **Advanced Training**: OneCycleLR scheduler provides better convergence than fixed learning rate
4. **Data Augmentation Impact**: RandomRotation and RandomAffine significantly improve generalization
5. **Architecture Refinement**: AdaptiveAvgPool2d provides more flexible spatial pooling
6. **Training Efficiency**: Achieves target accuracy within 15-20 epochs as required
7. **Generalization**: Excellent balance between training and test performance
8. **Parameter Efficiency**: Maintains parameter count under 8K while achieving best performance

---

## Key Learnings

### Progressive Improvements
1. **Model 1**: Established strong baseline architecture but suffered from overfitting
2. **Model 2**: Successfully addressed overfitting through regularization but plateau'd at lower accuracy
3. **Model 3**: Achieved optimal performance through balanced regularization and advanced training techniques

### Critical Success Factors
- **Data Augmentation**: Proved essential for achieving 99.4%+ accuracy
- **Learning Rate Scheduling**: OneCycleLR significantly improved convergence
- **Balanced Regularization**: Combination of weight decay + minimal dropout + data augmentation
- **Architecture Efficiency**: Proper use of 1x1 convolutions and Global Average Pooling

### Technical Insights
- Overfitting can be addressed through multiple complementary approaches
- Parameter efficiency doesn't always correlate with better performance
- Advanced training techniques (scheduling, augmentation) can provide significant gains
- Architecture design should balance capacity and regularization