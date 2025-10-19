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

## Learning Rate Schedulers

This implementation supports two state-of-the-art learning rate schedulers:

### Scheduler Options

#### 1. CosineAnnealingWarmRestarts (Default)
- **Type**: Cosine annealing with warm restarts
- **Configuration**: T_0=25, T_mult=1, eta_min=1e-4
- **Warmup**: Custom 5-epoch linear warmup (0.01 → 0.1)
- **Best for**: Stable, predictable training with periodic LR resets

#### 2. OneCycleLR
- **Type**: One Cycle Policy (Smith, 2018)
- **Configuration**: Loaded from `config.json`
- **Warmup**: Built-in (no separate warmup phase needed)
- **Best for**: Fast convergence, super-convergence training

### Scheduler Comparison

| Feature | CosineAnnealingWarmRestarts | OneCycleLR |
|---------|----------------------------|------------|
| **Learning Rate Pattern** | Cosine decay with restarts | Single cycle (warmup → decay) |
| **Warmup** | External (5 epochs) | Built-in (configurable %) |
| **Max LR** | 0.1 | Configurable (default: 0.1) |
| **Min LR** | 1e-4 | initial_lr/final_div_factor |
| **Training Speed** | Moderate | Fast (super-convergence) |
| **Stability** | Very stable | Can be aggressive |
| **Configuration** | Hardcoded | JSON file |
| **When to Use** | Research, benchmarking | Fast prototyping, competitions |

### OneCycleLR Configuration

When using `--scheduler onecycle`, the scheduler reads parameters from a JSON configuration file.

#### Configuration File Format

Create a `config.json` file (or use `config.json.example` as template):

```json
{
  "scheduler": {
    "onecycle": {
      "max_lr": 0.1,
      "pct_start": 0.3,
      "anneal_strategy": "cos",
      "div_factor": 25.0,
      "final_div_factor": 10000.0,
      "three_phase": false
    }
  }
}
```

#### OneCycleLR Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_lr` | 0.1 | Maximum learning rate (peak of the cycle) |
| `pct_start` | 0.3 | Percentage of cycle for warmup (30% = warmup phase) |
| `anneal_strategy` | "cos" | Annealing strategy: "cos" (cosine) or "linear" |
| `div_factor` | 25.0 | Initial LR = max_lr / div_factor (e.g., 0.1/25 = 0.004) |
| `final_div_factor` | 10000.0 | Final LR = initial_lr / final_div_factor |
| `three_phase` | false | Use three phases (warmup, anneal, final) vs two phases |

#### Recommended OneCycle Configurations

**Aggressive Training** (fast convergence):
```json
{
  "scheduler": {
    "onecycle": {
      "max_lr": 0.2,
      "pct_start": 0.2,
      "div_factor": 10.0
    }
  }
}
```

**Conservative Training** (stable, safer):
```json
{
  "scheduler": {
    "onecycle": {
      "max_lr": 0.05,
      "pct_start": 0.4,
      "div_factor": 50.0
    }
  }
}
```

**Default CIFAR-100** (balanced):
```json
{
  "scheduler": {
    "onecycle": {
      "max_lr": 0.1,
      "pct_start": 0.3,
      "div_factor": 25.0
    }
  }
}
```

### Configuration File Location

The configuration file can be specified in two ways:

1. **Default location**: Place `config.json` in the same directory as `train.py`
2. **Custom path**: Use `--scheduler-config` to specify a different path

```bash
# Use default config.json in current directory
python train.py --scheduler onecycle

# Use custom config file path
python train.py --scheduler onecycle --scheduler-config /path/to/my_config.json
```

## LR Finder (Automatic Learning Rate Discovery)

The LR Finder automatically determines optimal learning rates for your scheduler by running a range test before training. This feature is based on Leslie Smith's paper ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186).

### What is LR Finder?

LR Finder runs a short training session (typically 2-5 epochs) with exponentially increasing learning rates from very small (1e-6) to very large (1.0) values. It tracks the loss at each learning rate and helps you find:
- **For OneCycleLR**: Optimal `max_lr` and `base_lr`
- **For CosineAnnealing**: Optimal `initial_lr`

### How It Works

1. **Range Test**: Trains model with LRs ranging from `start_lr` to `end_lr`
2. **Clean Data**: Temporarily disables MixUp, label smoothing, and aggressive augmentations
3. **Loss Tracking**: Records smoothed loss at each learning rate
4. **Analysis**: Applies selected method to find optimal LR(s)
5. **Scheduler Update**: Automatically updates scheduler with found LRs
6. **Training Start**: Proceeds with normal training using optimal LRs

### Configuration

Add `lr_finder` section to your `config.json`:

```json
{
  "lr_finder": {
    "num_epochs": 3,
    "start_lr": 1e-6,
    "end_lr": 1.0,
    "selection_method": "steepest_gradient"
  },
  "scheduler": {
    "onecycle": {
      "max_lr": 0.1,
      "pct_start": 0.3,
      "div_factor": 25.0
    }
  }
}
```

#### LR Finder Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 3 | Number of epochs for range test (2-5 recommended) |
| `start_lr` | 1e-6 | Starting learning rate (very small) |
| `end_lr` | 1.0 | Ending learning rate (very large) |
| `selection_method` | "steepest_gradient" | Method for selecting optimal LR |

### Selection Methods

#### 1. Steepest Gradient (Recommended)

Selects the learning rate where loss is decreasing fastest (maximum negative gradient).

**When to use**: Default choice for most cases. Finds aggressive max_lr that enables fast learning.

**Pros**:
- Often finds the "sweet spot" for rapid convergence
- Good balance between speed and stability
- Works well for OneCycleLR

**Cons**:
- May be too aggressive for some models/datasets
- Requires smooth loss curve

**Configuration**:
```json
{
  "lr_finder": {
    "selection_method": "steepest_gradient"
  }
}
```

#### 2. Before Divergence (Conservative)

Selects the learning rate just before the loss starts increasing significantly.

**When to use**: When training stability is critical, or if steepest gradient gives unstable results.

**Pros**:
- Lower risk of divergence
- More stable training
- Good for sensitive models

**Cons**:
- May be overly conservative
- Slower convergence than steepest gradient

**Configuration**:
```json
{
  "lr_finder": {
    "selection_method": "before_divergence"
  }
}
```

#### 3. Valley (Experimental)

Selects the learning rate at the minimum loss point.

**When to use**: Exploration and experimentation. Not recommended for production.

**Pros**:
- Represents best performance during range test
- Easy to understand conceptually

**Cons**:
- May not generalize well to full training
- Can be misleading
- Often too conservative

**Configuration**:
```json
{
  "lr_finder": {
    "selection_method": "valley"
  }
}
```

#### 4. Manual Inspection

Displays the plot and allows you to manually inspect the curve.

**When to use**: For experts who want visual inspection and full control.

**Pros**:
- Maximum flexibility
- You can see exactly what's happening
- Good for learning and understanding

**Cons**:
- Requires manual intervention
- Breaks automation
- Slower workflow

**Configuration**:
```json
{
  "lr_finder": {
    "selection_method": "manual"
  }
}
```

### Usage Examples

#### Basic Usage with OneCycleLR

```bash
# Run LR Finder then train with OneCycleLR
python train.py --lr-finder --scheduler onecycle

# Example output:
# LR FINDER ANALYSIS
# ==================================================================
# Method: Steepest Gradient
# → Suggested LR: 0.085432
#
# ONECYCLE LR RECOMMENDATIONS
# ==================================================================
# Max LR:  0.085432 (peak learning rate)
# Base LR: 0.008543 (starting learning rate)
# Ratio:   10.0x (max_lr / base_lr)
# ==================================================================
#
# UPDATING ONECYCLE SCHEDULER WITH LR FINDER RESULTS
# Previous max_lr: 0.100000
# New max_lr:      0.085432
# New base_lr:     0.008543
```

#### Basic Usage with CosineAnnealing

```bash
# Run LR Finder then train with CosineAnnealing
python train.py --lr-finder --scheduler cosine

# Example output:
# COSINE ANNEALING LR RECOMMENDATIONS
# ==================================================================
# Initial LR: 0.092341
# This will be used with existing T_0 and eta_min parameters
# ==================================================================
```

#### With Custom Config File

```bash
# Use custom config with LR Finder settings
python train.py --lr-finder --scheduler onecycle --scheduler-config ./my_config.json
```

#### With ResNet50

```bash
# LR Finder with ResNet50
python train.py --model resnet50 --lr-finder --scheduler onecycle
```

### Python API Usage

```python
from train import CIFARTrainer

# Configure LR Finder
lr_finder_config = {
    'num_epochs': 3,
    'start_lr': 1e-6,
    'end_lr': 1.0,
    'selection_method': 'steepest_gradient'
}

# Train with LR Finder
trainer = CIFARTrainer(
    model_module_name='model',
    epochs=100,
    batch_size=256,
    scheduler_type='onecycle',
    use_lr_finder=True,              # Enable LR Finder
    lr_finder_config=lr_finder_config
)

best_acc = trainer.run()
```

### Output Files

When LR Finder runs, it creates:
- **`lr_finder_plot.png`**: Visualization showing:
  - Loss vs Learning Rate curve (log scale)
  - Marked optimal LR points (max_lr, base_lr, or initial_lr)
  - Clear legend and annotations

Saved in your checkpoint directory: `./checkpoint_N/lr_finder_plot.png`

### How LR Finder Affects Training

During LR Finder epochs (before actual training):
- ✅ **Model training happens** with varying LRs
- ❌ **No checkpoints saved** during LR Finder
- ❌ **MixUp disabled** for clean signal
- ❌ **Label smoothing disabled** (set to 0.0)
- ✅ **Basic augmentations kept** (flip, rotate, etc.)
- ✅ **Model state restored** after range test

After LR Finder completes:
- Optimal LR(s) determined and displayed
- Scheduler automatically updated with new LRs
- Normal training begins with all augmentations enabled
- Training proceeds as usual

### Best Practices

1. **Start with steepest_gradient**: Works well for most cases
2. **Try before_divergence if training is unstable**: More conservative
3. **Use 3 epochs for LR Finder**: Good balance of speed and accuracy
4. **Inspect the plot**: Always check `lr_finder_plot.png` to verify results
5. **Adjust if needed**: If suggested LR seems too high/low, try different method
6. **Dataset-specific tuning**: Different datasets may prefer different methods

### When to Use LR Finder

**Use LR Finder when**:
- ✅ Starting with a new dataset or model architecture
- ✅ Unsure about optimal learning rate
- ✅ Want to save time on manual LR tuning
- ✅ Using OneCycleLR (especially important for max_lr)
- ✅ Experimenting with different schedulers

**Skip LR Finder when**:
- ❌ You already know good LR values for your setup
- ❌ Reproducing published results with known hyperparameters
- ❌ Very short training runs (< 20 epochs)
- ❌ Doing quick debugging/testing

### Troubleshooting LR Finder

#### Issue: LR Finder suggests very high/low LR

**Solution**:
- Check `lr_finder_plot.png` to inspect the curve
- Try a different `selection_method`
- Adjust `start_lr` or `end_lr` range
- Manually override if needed

#### Issue: Loss curve is very noisy

**Solution**:
- Increase `num_epochs` (try 4-5 instead of 3)
- Check if data augmentation is too aggressive
- Ensure batch size is reasonable (not too small)

#### Issue: LR Finder diverges early

**Solution**:
- Reduce `end_lr` (try 0.5 instead of 1.0)
- Check model initialization
- Verify data preprocessing is correct

### Example: Complete Workflow

```bash
# Step 1: Prepare config with LR Finder settings
cat > config.json <<EOF
{
  "lr_finder": {
    "num_epochs": 3,
    "start_lr": 1e-6,
    "end_lr": 1.0,
    "selection_method": "steepest_gradient"
  },
  "scheduler": {
    "onecycle": {
      "max_lr": 0.1,
      "pct_start": 0.3,
      "div_factor": 25.0,
      "final_div_factor": 10000.0,
      "three_phase": false
    }
  }
}
EOF

# Step 2: Run training with LR Finder
python train.py --lr-finder --scheduler onecycle --epochs 100

# Step 3: Check results
# - LR Finder runs for 3 epochs
# - Optimal LRs displayed in console
# - Check ./checkpoint_1/lr_finder_plot.png
# - Training proceeds with optimal LRs
# - Final model saved in ./checkpoint_1/
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

#### Training with Different Schedulers

```bash
# Default: CosineAnnealingWarmRestarts
python train.py --model model --epochs 100

# Explicitly specify CosineAnnealing
python train.py --scheduler cosine --epochs 100

# Use OneCycleLR with default config.json
python train.py --scheduler onecycle --epochs 100

# Use OneCycleLR with custom config file
python train.py --scheduler onecycle --scheduler-config ./my_onecycle_config.json --epochs 100

# ResNet50 with OneCycleLR
python train.py --model resnet50 --scheduler onecycle --epochs 100
```

#### All Command-Line Options

```bash
python train.py \
  --model MODEL_MODULE_NAME \       # Model: 'model' (WideResNet) or 'resnet50' (default: 'model')
  --epochs NUM_EPOCHS \             # Number of training epochs (default: 100)
  --batch-size BATCH_SIZE \         # Batch size (default: 256)
  --device DEVICE \                 # Device: 'cuda', 'mps', 'cpu', or None for auto-detect (default: None)
  --scheduler SCHEDULER_TYPE \      # Scheduler: 'cosine' or 'onecycle' (default: 'cosine')
  --scheduler-config CONFIG_PATH \  # Path to scheduler config JSON (for onecycle, default: ./config.json)
  --lr-finder \                     # Run LR Finder to auto-determine optimal learning rates (optional)
  --hf-token HF_TOKEN \             # HuggingFace API token (optional)
  --hf-repo HF_REPO_ID              # HuggingFace repo ID like 'username/repo' (optional)
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

# Train WideResNet-28-10 with CosineAnnealing (default)
trainer = CIFARTrainer(
    model_module_name='model',       # WideResNet-28-10
    epochs=100,                      # Number of epochs
    batch_size=256,                  # Batch size
    scheduler_type='cosine',         # CosineAnnealingWarmRestarts (default)
    use_mixup=True,                  # Enable MixUp augmentation
    mixup_alpha=0.2,                 # MixUp alpha parameter
    label_smoothing=0.1,             # Label smoothing factor
    use_amp=True,                    # Use automatic mixed precision
    gradient_clip=1.0,               # Gradient clipping max norm
    warmup_epochs=5,                 # Number of warmup epochs (for cosine)
    checkpoint_epochs=[10, 25, 50, 75],  # Epochs to save checkpoints
    hf_token='your_token',           # HuggingFace token (optional)
    hf_repo_id='username/repo'       # HuggingFace repo ID (optional)
)
best_accuracy = trainer.run()

# Train with OneCycleLR scheduler
trainer_onecycle = CIFARTrainer(
    model_module_name='model',
    epochs=100,
    batch_size=256,
    scheduler_type='onecycle',       # Use OneCycleLR
    scheduler_config_path='./config.json',  # Path to OneCycle config
    use_mixup=True,
    label_smoothing=0.1,
    use_amp=True,
    gradient_clip=1.0,
    checkpoint_epochs=[10, 25, 50, 75],
    hf_token='your_token',
    hf_repo_id='username/repo'
)
best_accuracy = trainer_onecycle.run()

# Or train ResNet50 with OneCycleLR
trainer_resnet = CIFARTrainer(
    model_module_name='resnet50',    # ResNet50
    epochs=100,
    batch_size=256,
    scheduler_type='onecycle',       # OneCycleLR
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
| `scheduler_type` | str | `'cosine'` | Learning rate scheduler type: 'cosine' or 'onecycle' |
| `scheduler_config_path` | str | `None` | Path to scheduler config JSON (for OneCycleLR) |
| `use_lr_finder` | bool | `False` | Enable LR Finder to automatically determine optimal LRs |
| `lr_finder_config` | dict | `None` | LR Finder configuration (num_epochs, start_lr, end_lr, selection_method) |
| `use_mixup` | bool | `True` | Enable MixUp data augmentation |
| `mixup_alpha` | float | `0.2` | MixUp alpha parameter (beta distribution) |
| `label_smoothing` | float | `0.1` | Label smoothing factor (0.0 = no smoothing) |
| `use_amp` | bool | `True` | Use automatic mixed precision training |
| `gradient_clip` | float | `1.0` | Maximum gradient norm for clipping |
| `warmup_epochs` | int | `5` | Number of warmup epochs (only for cosine scheduler) |
| `checkpoint_epochs` | list | `[10, 20, 25, 30, 40, 50, 60, 75, 90]` | Epochs to save checkpoints |
| `hf_token` | str | `None` | HuggingFace API token for model upload |
| `hf_repo_id` | str | `None` | HuggingFace repository ID (e.g., 'username/repo-name') |
| `device` | str | `None` | Device to use: 'cuda', 'mps', 'cpu', or None for auto-detect |

### Default Training Configuration

```python
# Optimizer
optimizer = SGD(
    lr=0.01,              # Initial learning rate
    momentum=0.9,
    weight_decay=1e-3     # L2 regularization
)

# Learning Rate Schedule (Scheduler: CosineAnnealingWarmRestarts - default)
# 1. Warmup: 5 epochs (0.01 → 0.1)
# 2. Cosine Annealing with Warm Restarts
#    - T_0 = 25 (first cycle length)
#    - T_mult = 1 (cycle length stays same)
#    - eta_min = 1e-4 (minimum LR)

# Learning Rate Schedule (Scheduler: OneCycleLR - optional)
# 1. Single cycle over all epochs
# 2. Warmup phase: 30% of training (configurable via pct_start)
# 3. Annealing phase: 70% of training
# 4. max_lr: 0.1 (configurable via config.json)
# 5. Initial LR: max_lr / div_factor
# 6. Final LR: initial_lr / final_div_factor

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
├── config.json.example   # Example OneCycleLR scheduler configuration
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

**`config.json.example`**:
- Example configuration file for OneCycleLR scheduler
- Includes parameter descriptions and usage notes
- Recommended configurations for different training scenarios
- Copy to `config.json` and customize as needed

**`train.py`**:
- `CIFARTrainer` class with all training logic
- MixUp augmentation implementation
- WarmupScheduler for learning rate warmup (CosineAnnealing only)
- Scheduler configuration loading and management
- Checkpoint saving and loading
- HuggingFace Hub integration
- Visualization and plotting functions
- Command-line interface with scheduler selection

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

## Scheduler Selection Guidelines

### When to Use CosineAnnealingWarmRestarts

**Advantages**:
- Very stable and predictable
- Periodic learning rate resets help escape local minima
- Well-tested on CIFAR-100
- Good for research and reproducibility

**Use when**:
- You want stable, predictable training
- Reproducing research results
- Don't want to tune many hyperparameters
- Training for many epochs (100+)

### When to Use OneCycleLR

**Advantages**:
- Often achieves faster convergence
- Can reach higher accuracy with fewer epochs
- Super-convergence possible with optimal settings
- Great for competitions and fast prototyping

**Use when**:
- You need fast results
- Willing to tune hyperparameters (max_lr, pct_start)
- Training for fewer epochs (30-60)
- Have time for experimentation

**Tips for OneCycleLR**:
1. Start with default config and adjust based on results
2. If training is unstable, reduce `max_lr` or increase `div_factor`
3. If convergence is slow, try increasing `max_lr` carefully
4. Adjust `pct_start` based on dataset size (larger = more warmup)

## Troubleshooting

### Issue: OneCycleLR config file not found

**Solution**: Create `config.json` in the same directory as `train.py`:
```bash
# Copy the example file
cp config.json.example config.json

# Or specify custom path
python train.py --scheduler onecycle --scheduler-config /path/to/config.json
```

### Issue: Training unstable with OneCycleLR

**Solution**: Reduce max_lr or increase div_factor in config.json:
```json
{
  "scheduler": {
    "onecycle": {
      "max_lr": 0.05,        // Reduced from 0.1
      "div_factor": 50.0     // Increased from 25.0
    }
  }
}
```

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
- **OneCycleLR**: [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120) by Smith, 2018
- **CIFAR-100**: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) by Krizhevsky

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Contribute via pull requests

---

**Happy Training!** 🚀
