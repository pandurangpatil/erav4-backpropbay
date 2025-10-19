# CIFAR-100 Training with WideResNet-28-10 and ResNet50

A comprehensive PyTorch implementation for training WideResNet-28-10 and ResNet50 on CIFAR-100 dataset with state-of-the-art training techniques including MixUp augmentation, label smoothing, mixed precision training, and automatic HuggingFace Hub integration.

## Features

- **Multiple Model Architectures**:
  - WideResNet-28-10 with 36.5M parameters
  - ResNet50 with 23.7M parameters (custom implementation)
- **CLI Model Selection**: Choose your model via command-line argument
- **State-of-the-art Augmentation**: Albumentations-based transforms with Cutout, ColorJitter, and geometric transforms
- **MixUp Augmentation**: Data mixing with configurable alpha parameter
- **Label Smoothing**: Reduces overfitting and improves generalization
- **Mixed Precision Training**: Faster training with automatic mixed precision (AMP)
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Warmup**: Smooth warmup followed by cosine annealing with warm restarts
- **Smart Checkpointing**: Local checkpoint saves with optional HuggingFace Hub upload
- **Early Stopping**: Automatic stopping with configurable patience
- **Rich Visualizations**: Training curves, console plots, and comprehensive metrics
- **HuggingFace Integration**: Automatic model card generation and upload

## Model Architecture

### WideResNet-28-10

- **Architecture**: Wide Residual Network
- **Depth**: 28 layers
- **Width Factor**: 10
- **Parameters**: ~36.5M
- **Dropout**: 0.3
- **Input Size**: 32×32×3 (CIFAR-100 images)
- **Output**: 100 classes

**Components**:
- **BasicBlock**: Residual block with batch normalization, ReLU, and dropout
- **NetworkBlock**: Stack of basic blocks with configurable depth
- **Global Average Pooling**: Reduces spatial dimensions before classification
- **Fully Connected Layer**: Final classification layer

### ResNet50

- **Architecture**: Deep Residual Network with Bottleneck blocks
- **Depth**: 50 layers
- **Parameters**: ~23.7M
- **Input Size**: 32×32×3 (CIFAR-100 images, adapted from ImageNet)
- **Output**: 100 classes
- **Layer Structure**: [3, 4, 6, 3] blocks per stage

**Components**:
- **Bottleneck Block**: 1×1 → 3×3 → 1×1 convolution with expansion factor 4
- **Initial Conv**: 3×3 conv (adapted for CIFAR, not 7×7 as in ImageNet)
- **4 Residual Stages**: Progressive feature extraction with downsampling
- **Adaptive Average Pooling**: Reduces spatial dimensions before classification
- **Fully Connected Layer**: Final classification layer

**CIFAR Adaptations**:
- Replaced 7×7 conv (stride 2) with 3×3 conv (stride 1)
- Removed max pooling layer (too aggressive for 32×32 images)
- Optimized for smaller input resolution

### Model Comparison

| Feature | WideResNet-28-10 | ResNet50 |
|---------|------------------|----------|
| **Parameters** | 36.5M | 23.7M |
| **Depth** | 28 layers | 50 layers |
| **Width** | Very wide (10×) | Standard bottleneck |
| **Block Type** | BasicBlock with dropout | Bottleneck (1×1→3×3→1×1) |
| **Memory Usage** | Higher | Lower |
| **Training Speed** | Slower | Faster |
| **Typical CIFAR-100 Accuracy** | 70-74% | 68-72% |

**When to use WideResNet-28-10**:
- Maximum accuracy is the priority
- Sufficient GPU memory available (8GB+)
- Not concerned about model size
- Research and benchmarking

**When to use ResNet50**:
- Need faster training/inference
- Limited GPU memory (4-6GB)
- Want smaller model size for deployment
- Good balance of accuracy and efficiency

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

**Dependencies**:
- `torch` (PyTorch)
- `torchvision`
- `torchsummary`
- `albumentations`
- `numpy`
- `tqdm`
- `matplotlib`
- `plotille`
- `huggingface_hub` (optional, for model upload)

### Quick Install

```bash
pip install torch torchvision torchsummary albumentations numpy tqdm matplotlib plotille huggingface_hub
```

## Usage

### 1. Command-Line Training

#### Basic Training with WideResNet (default)

```bash
python train.py
# or explicitly specify
python train.py --model model
```

#### Train with ResNet50

```bash
python train.py --model resnet50
```

#### Training with Custom Parameters

```bash
# WideResNet with custom parameters
python train.py --model model --epochs 100 --batch-size 256

# ResNet50 with custom parameters
python train.py --model resnet50 --epochs 100 --batch-size 256
```

#### Training with HuggingFace Upload

```bash
# WideResNet
python train.py \
  --model model \
  --epochs 100 \
  --batch-size 256 \
  --hf-token YOUR_HUGGINGFACE_TOKEN \
  --hf-repo username/cifar100-wideresnet

# ResNet50
python train.py \
  --model resnet50 \
  --epochs 100 \
  --batch-size 256 \
  --hf-token YOUR_HUGGINGFACE_TOKEN \
  --hf-repo username/cifar100-resnet50
```

#### All Command-Line Options

```bash
python train.py \
  --model MODEL_MODULE_NAME \    # Model: 'model' (WideResNet) or 'resnet50' (default: 'model')
  --epochs NUM_EPOCHS \           # Number of training epochs (default: 100)
  --batch-size BATCH_SIZE \       # Batch size (default: 256)
  --device DEVICE \               # Device: 'cuda', 'mps', 'cpu', or None for auto-detect (default: None)
  --hf-token HF_TOKEN \           # HuggingFace API token (optional)
  --hf-repo HF_REPO_ID            # HuggingFace repo ID like 'username/repo' (optional)
```

### 2. Google Colab Training

Open `train.ipynb` in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pandurangpatil/erav4-backpropbay/blob/main/session8-more-trials/train.ipynb)

**Steps**:
1. Open the notebook in Colab
2. Set runtime to GPU (Runtime → Change runtime type → GPU)
3. (Optional) Add HuggingFace token to Colab secrets:
   - Click 🔑 icon in left sidebar
   - Add secret: `HF_TOKEN` with your token
   - Update `HF_REPO_ID` in the notebook
4. Run all cells

### 3. Python API Usage

```python
from train import CIFARTrainer

# Train WideResNet-28-10
trainer = CIFARTrainer(
    model_module_name='model',       # WideResNet-28-10
    epochs=100,                      # Number of epochs
    batch_size=256,                  # Batch size
    use_mixup=True,                  # Enable MixUp augmentation
    mixup_alpha=0.2,                 # MixUp alpha parameter
    label_smoothing=0.1,             # Label smoothing factor
    use_amp=True,                    # Use automatic mixed precision
    gradient_clip=1.0,               # Gradient clipping max norm
    warmup_epochs=5,                 # Number of warmup epochs
    checkpoint_epochs=[10, 25, 50, 75],  # Epochs to save checkpoints
    hf_token='your_token',           # HuggingFace token (optional)
    hf_repo_id='username/repo'       # HuggingFace repo ID (optional)
)
best_accuracy = trainer.run()

# Or train ResNet50
trainer_resnet = CIFARTrainer(
    model_module_name='resnet50',    # ResNet50
    epochs=100,
    batch_size=256,
    # ... other parameters same as above
)
best_accuracy = trainer_resnet.run()

print(f"Best test accuracy: {best_accuracy:.2f}%")
```

## Training Configuration

### CIFARTrainer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_module_name` | str | `'model'` | Name of the module containing the model |
| `epochs` | int | `100` | Number of training epochs |
| `batch_size` | int | `256` | Batch size for training and testing |
| `use_mixup` | bool | `True` | Enable MixUp data augmentation |
| `mixup_alpha` | float | `0.2` | MixUp alpha parameter (beta distribution) |
| `label_smoothing` | float | `0.1` | Label smoothing factor (0.0 = no smoothing) |
| `use_amp` | bool | `True` | Use automatic mixed precision training |
| `gradient_clip` | float | `1.0` | Maximum gradient norm for clipping |
| `warmup_epochs` | int | `5` | Number of warmup epochs (0.01 → 0.1) |
| `checkpoint_epochs` | list | `[10, 20, 25, 30, 40, 50, 60, 75, 90]` | Epochs to save checkpoints |
| `hf_token` | str | `None` | HuggingFace API token for model upload |
| `hf_repo_id` | str | `None` | HuggingFace repository ID (e.g., 'username/repo-name') |

### Default Training Configuration

```python
# Optimizer
optimizer = SGD(
    lr=0.01,              # Initial learning rate
    momentum=0.9,
    weight_decay=1e-3     # L2 regularization
)

# Learning Rate Schedule
# 1. Warmup: 5 epochs (0.01 → 0.1)
# 2. Cosine Annealing with Warm Restarts
#    - T_0 = 25 (first cycle length)
#    - T_mult = 1 (cycle length stays same)
#    - eta_min = 1e-4 (minimum LR)

# Early Stopping
patience = 15  # Stop if no improvement for 15 epochs
```

## Data Augmentation

### Training Augmentation (Albumentations)

```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.0625,      # ±6.25% translation
        scale_limit=0.1,         # ±10% scaling
        rotate_limit=15,         # ±15° rotation
        p=0.5
    ),
    A.CoarseDropout(            # Cutout augmentation
        max_holes=1,
        max_height=8,
        max_width=8,
        p=0.5,
        fill_value=0
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10,
        p=0.3
    ),
    A.Normalize(
        mean=(0.5071, 0.4865, 0.4409),  # CIFAR-100 mean
        std=(0.2673, 0.2564, 0.2761)     # CIFAR-100 std
    ),
    ToTensorV2()
])
```

### Test Augmentation

```python
A.Compose([
    A.Normalize(
        mean=(0.5071, 0.4865, 0.4409),
        std=(0.2673, 0.2564, 0.2761)
    ),
    ToTensorV2()
])
```

## Checkpoint Management

### Local Checkpoints

**Saved at specified epochs** (default: 10, 20, 25, 30, 40, 50, 60, 75, 90):
- Location: `./checkpoints/checkpoint_epoch{N}.pth`
- Contains: model state, optimizer state, metrics, config

**Best model** (saved when test accuracy improves):
- Location: `./checkpoints/best_model.pth`
- Automatically updated during training

**Checkpoint Contents**:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'train_accuracy': float,
    'test_accuracy': float,
    'train_loss': float,
    'test_loss': float,
    'timestamp': str,
    'config': dict
}
```

### HuggingFace Hub Upload

**Uploads after training completes** (if token and repo provided):
- `best_model.pth` - Best performing checkpoint
- `metrics.json` - Complete training history
- `training_curves.png` - Visualization plots
- `config.json` - Hyperparameter configuration
- `README.md` - Auto-generated model card

**Model Card Includes**:
- Model architecture details
- Training configuration
- Performance metrics
- Usage examples
- Citation

## Results and Outputs

### Console Output

During training, you'll see:
- Real-time progress bars with loss and accuracy
- Current learning rate
- Test set evaluation after each epoch
- Best model notifications
- Checkpoint saves
- Early stopping warnings

### Saved Files

```
./checkpoints/
├── best_model.pth                  # Best performing model
├── checkpoint_epoch10.pth          # Checkpoint at epoch 10
├── checkpoint_epoch25.pth          # Checkpoint at epoch 25
├── checkpoint_epoch50.pth          # Checkpoint at epoch 50
├── ...                             # Other configured checkpoints
├── training_curves.png             # 4-panel training visualization
├── metrics.json                    # Complete training history
├── config.json                     # Final configuration
└── README.md                       # Auto-generated model card
```

### Visualizations

**Training Curves Plot** (`training_curves.png`):
1. **Loss**: Train vs Test loss over epochs
2. **Accuracy**: Train vs Test accuracy with 74% target line
3. **Learning Rate**: LR schedule visualization
4. **Overfitting Gap**: Train-Test accuracy difference

**Console Plots** (via plotille):
- Colored ASCII plots in terminal
- Real-time training monitoring
- No GUI required

### Metrics JSON

```json
{
  "epochs": [1, 2, 3, ...],
  "train_losses": [...],
  "test_losses": [...],
  "train_accuracies": [...],
  "test_accuracies": [...],
  "learning_rates": [...],
  "best_test_accuracy": 74.5,
  "best_epoch": 85
}
```

## Examples

### Example 1: Quick Start

```bash
# Train WideResNet with defaults (no HuggingFace upload)
python train.py

# Train ResNet50 with defaults
python train.py --model resnet50

# Expected output:
# - 100 epochs of training
# - Checkpoints in ./checkpoint_N/
# - Training curves and metrics
# - Best model saved
```

### Example 2: Custom Training Configuration

```python
from train import CIFARTrainer

# Faster training for testing
trainer = CIFARTrainer(
    epochs=30,              # Fewer epochs
    batch_size=512,         # Larger batch size
    use_mixup=False,        # Disable MixUp for speed
    warmup_epochs=2,        # Shorter warmup
    checkpoint_epochs=[10, 20]  # Only save 2 checkpoints
)

best_acc = trainer.run()
```

### Example 3: Training with HuggingFace Upload

```python
from train import CIFARTrainer

trainer = CIFARTrainer(
    epochs=100,
    batch_size=256,
    hf_token='hf_...',                          # Your HF token
    hf_repo_id='username/cifar100-wideresnet'  # Your HF repo
)

# Train and auto-upload to HuggingFace
best_acc = trainer.run()

# Access your model at:
# https://huggingface.co/username/cifar100-wideresnet
```

### Example 4: Resume from Checkpoint

```python
import torch
from model import WideResNet
from resnet50 import ResNet50

# Load WideResNet checkpoint
checkpoint = torch.load('./checkpoint_1/checkpoint_epoch50.pth')
model = WideResNet(depth=28, widen_factor=10, num_classes=100)
model.load_state_dict(checkpoint['model_state_dict'])

# Or load ResNet50 checkpoint
checkpoint_resnet = torch.load('./checkpoint_2/checkpoint_epoch50.pth')
model_resnet = ResNet50(num_classes=100)
model_resnet.load_state_dict(checkpoint_resnet['model_state_dict'])

# Check checkpoint info
print(f"Epoch: {checkpoint['epoch']}")
print(f"Test Accuracy: {checkpoint['test_accuracy']:.2f}%")
print(f"Timestamp: {checkpoint['timestamp']}")
```

### Example 5: Load Model from HuggingFace

```python
import torch
from huggingface_hub import hf_hub_download
from model import WideResNet
from resnet50 import ResNet50

# Download WideResNet from HuggingFace
checkpoint_path = hf_hub_download(
    repo_id="username/cifar100-wideresnet",
    filename="best_model.pth"
)
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model = WideResNet(depth=28, widen_factor=10, num_classes=100)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Or download ResNet50 from HuggingFace
checkpoint_path_resnet = hf_hub_download(
    repo_id="username/cifar100-resnet50",
    filename="best_model.pth"
)
checkpoint_resnet = torch.load(checkpoint_path_resnet, map_location='cpu')
model_resnet = ResNet50(num_classes=100)
model_resnet.load_state_dict(checkpoint_resnet['model_state_dict'])
model_resnet.eval()

# Use for inference
with torch.no_grad():
    output = model(input_tensor)
    predictions = output.argmax(dim=1)
```

## Project Structure

```
session8-more-trials/
├── model.py              # WideResNet-28-10 architecture and transforms
├── resnet50.py           # ResNet50 architecture and transforms (custom implementation)
├── train.py              # CIFARTrainer class and training logic
├── train.ipynb           # Google Colab training notebook
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── checkpoint_*/         # Training outputs (created automatically per run)
    ├── best_model.pth
    ├── checkpoint_*.pth
    ├── metrics.json
    ├── config.json
    ├── training_curves.png
    └── README.md
```

### File Descriptions

**`model.py`**:
- WideResNet-28-10 architecture (BasicBlock, NetworkBlock, WideResNet)
- Albumentations transforms for train/test
- CIFAR-100 normalization parameters
- Optimizer configuration (SGD with momentum and weight decay)
- Scheduler configuration (CosineAnnealingWarmRestarts)

**`resnet50.py`**:
- ResNet50 architecture (Bottleneck, ResNet50) - custom implementation from scratch
- Same Albumentations transforms as WideResNet
- CIFAR-100 optimized (3×3 initial conv, no max pooling)
- Same optimizer and scheduler configuration as WideResNet
- Compatible with the same training pipeline

**`train.py`**:
- `CIFARTrainer` class with all training logic
- MixUp augmentation implementation
- WarmupScheduler for learning rate warmup
- Checkpoint saving and loading
- HuggingFace Hub integration
- Visualization and plotting functions
- Command-line interface

**`train.ipynb`**:
- Google Colab-ready notebook
- Step-by-step training guide
- HuggingFace token setup (via Colab secrets)
- Configuration display
- Training execution
- Results summary

## Advanced Features

### MixUp Augmentation

MixUp creates virtual training examples by mixing pairs of images and their labels:

```python
# Mix two samples
mixed_image = λ * image_a + (1-λ) * image_b
mixed_label = λ * label_a + (1-λ) * label_b

# where λ ~ Beta(α, α)
```

**Benefits**:
- Improves generalization
- Reduces overfitting
- More robust to label noise

### Label Smoothing

Prevents the model from becoming overconfident:

```python
# Hard labels: [0, 0, 1, 0, 0]
# Soft labels: [0.025, 0.025, 0.9, 0.025, 0.025]  (smoothing=0.1)
```

**Benefits**:
- Better calibration
- Improved generalization
- Reduced overfitting

### Mixed Precision Training

Uses FP16 for faster training while maintaining accuracy:

```python
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- 2-3x faster training
- Reduced memory usage
- Maintained accuracy

### Gradient Clipping

Prevents exploding gradients by limiting gradient magnitude:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Benefits**:
- More stable training
- Better convergence
- Prevents NaN losses

### Learning Rate Warmup

Gradually increases learning rate at the start:

```
Epochs 1-5: Linear warmup (0.01 → 0.1)
Epochs 6+:  Cosine annealing with restarts
```

**Benefits**:
- Stable early training
- Better final performance
- Reduced sensitivity to LR

### Early Stopping

Automatically stops training when no improvement:

```python
patience = 15  # Stop after 15 epochs without improvement
```

**Benefits**:
- Saves computation time
- Prevents overfitting
- Optimal model selection

## Tips and Best Practices

### 1. GPU Memory

If you encounter OOM (Out of Memory) errors:

```python
# Reduce batch size
trainer = CIFARTrainer(batch_size=128)  # Instead of 256

# Or disable mixed precision
trainer = CIFARTrainer(use_amp=False)
```

### 2. Faster Training

For quicker iterations during development:

```python
trainer = CIFARTrainer(
    epochs=30,              # Fewer epochs
    batch_size=512,         # Larger batches
    use_mixup=False,        # Faster (but lower accuracy)
    checkpoint_epochs=[10, 20]  # Fewer checkpoints
)
```

### 3. Maximum Accuracy

For best possible results:

```python
trainer = CIFARTrainer(
    epochs=200,             # More epochs
    batch_size=256,         # Optimal batch size
    use_mixup=True,         # Data augmentation
    label_smoothing=0.1,    # Regularization
    use_amp=True,           # Speed + accuracy
    warmup_epochs=10        # Longer warmup
)
```

### 4. Monitoring Training

Watch for these signs:

- **Overfitting**: Large gap between train and test accuracy
  - Solution: Increase regularization, more augmentation
- **Underfitting**: Low train accuracy
  - Solution: Train longer, reduce regularization
- **Unstable training**: Loss fluctuates wildly
  - Solution: Lower learning rate, gradient clipping

### 5. HuggingFace Upload

To enable model sharing:

1. Create account at [huggingface.co](https://huggingface.co)
2. Get API token from [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Create repository or use existing one
4. Pass token and repo to trainer:

```python
trainer = CIFARTrainer(
    hf_token='hf_...',
    hf_repo_id='username/cifar100-wideresnet'
)
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or disable mixed precision:
```python
trainer = CIFARTrainer(batch_size=128, use_amp=False)
```

### Issue: "albumentations not found"

**Solution**: Install albumentations:
```bash
pip install albumentations
```

### Issue: "HuggingFace upload failed"

**Solutions**:
1. Check token is valid: `huggingface-cli login`
2. Verify repo exists on HuggingFace Hub
3. Check internet connection
4. Training will continue without upload

### Issue: Training too slow

**Solutions**:
1. Ensure GPU is being used (check `Device: cuda`)
2. Increase batch size if GPU memory allows
3. Reduce number of workers in data loader
4. Check for CPU bottlenecks

### Issue: Poor accuracy

**Potential causes**:
1. Training too short - increase epochs
2. Learning rate too high/low - adjust scheduler
3. Insufficient augmentation - check transforms
4. Model not converging - check loss curve

## Performance Expectations

### Expected Results

With default configuration:

| Metric | Value |
|--------|-------|
| **Best Test Accuracy** | 70-74% |
| **Training Time** | ~6-8 hours (100 epochs on T4 GPU) |
| **Time per Epoch** | ~4 minutes |
| **GPU Memory** | ~6-8 GB |
| **Best Epoch** | Usually 75-95 |

### Benchmark Results

Training on Google Colab (T4 GPU):

```
Epoch 1:   Train 23.5% | Test 21.2% | Time 4m 12s
Epoch 25:  Train 75.3% | Test 62.8% | Time 4m 05s
Epoch 50:  Train 89.2% | Test 68.5% | Time 4m 03s
Epoch 75:  Train 94.1% | Test 71.8% | Time 4m 02s
Epoch 100: Train 96.5% | Test 71.2% | Time 4m 01s

Best: 71.8% @ Epoch 75
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cifar100-wideresnet-training,
  author = {Your Name},
  title = {CIFAR-100 Training with WideResNet-28-10},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/pandurangpatil/erav4-backpropbay/tree/main/session8-more-trials}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by He et al.
- **WideResNet**: [Wide Residual Networks](https://arxiv.org/abs/1605.07146) by Zagoruyko & Komodakis
- **MixUp**: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) by Zhang et al.
- **Cutout**: [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552) by DeVries & Taylor
- **CIFAR-100**: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) by Krizhevsky

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Contribute via pull requests

---

**Happy Training!** 🚀
