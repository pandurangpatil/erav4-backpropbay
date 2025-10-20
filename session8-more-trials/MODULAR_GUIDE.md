# Modular Codebase Guide

This guide explains the new modular structure of the CIFAR training codebase.

## Directory Structure

```
session8-more-trials/
├── data_loaders/          # Dataset management
│   ├── __init__.py       # Dataset factory: get_dataset()
│   ├── base.py           # BaseDataset abstract class
│   ├── cifar100.py       # CIFAR100 implementation
│   └── transforms/       # Data augmentation
│       ├── __init__.py
│       └── augmentations.py
├── models/               # Model architectures
│   ├── __init__.py      # Model registry: get_model()
│   ├── wideresnet.py    # WideResNet implementation
│   └── resnet50.py      # ResNet50 implementation
├── training/            # Training components
│   ├── __init__.py
│   ├── optimizer.py     # Optimizer factory
│   ├── scheduler.py     # Scheduler factory
│   ├── lr_finder.py     # LR Finder implementation
│   └── trainer.py       # Main Trainer class
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── device.py        # GPU detection
│   ├── checkpoints.py   # Checkpoint management
│   ├── metrics.py       # Metrics tracking & plotting
│   └── huggingface.py   # HuggingFace integration
├── notebooks/           # Jupyter notebooks
│   ├── train.ipynb
│   └── train-local.ipynb
├── train.py            # Main entry point (simplified)
├── config.json         # Configuration file
└── requirements.txt    # Dependencies
```

## Quick Start

### Basic Training

```bash
# Train with default settings (WideResNet-28-10, CIFAR-100, 100 epochs)
python train.py

# Train ResNet50 for 50 epochs with OneCycle scheduler
python train.py --model resnet50 --epochs 50 --scheduler onecycle

# Train with LR Finder (automatically finds optimal learning rate)
python train.py --lr-finder --scheduler onecycle

# Train with specific augmentation strength
python train.py --augmentation weak

# Train without MixUp
python train.py --no-mixup
```

### Advanced Usage

```bash
# Full configuration
python train.py \
  --model wideresnet28-10 \
  --dataset cifar100 \
  --epochs 100 \
  --batch-size 256 \
  --optimizer sgd \
  --scheduler onecycle \
  --augmentation strong \
  --mixup-alpha 0.2 \
  --label-smoothing 0.1 \
  --gradient-clip 1.0 \
  --lr-finder \
  --hf-token YOUR_TOKEN \
  --hf-repo username/model-name
```

## Using the Modular Components

### 1. Dataset Module

```python
from data_loaders import get_dataset, get_dataset_info

# Get dataset info without loading
info = get_dataset_info('cifar100')
print(f"Classes: {info['num_classes']}")

# Load dataset with different augmentation strengths
train_dataset = get_dataset('cifar100', train=True, augmentation='strong')
test_dataset = get_dataset('cifar100', train=False, augmentation='none')

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
```

### 2. Model Module

```python
from models import get_model, list_models

# List available models
print(list_models())  # ['wideresnet28-10', 'resnet50', ...]

# Create model
model = get_model('wideresnet28-10', num_classes=100)
model = get_model('resnet50', num_classes=10)
```

### 3. Training Module

```python
from training import get_optimizer, get_scheduler, Trainer

# Create optimizer
optimizer = get_optimizer('sgd', model, lr=0.01, momentum=0.9)

# Create scheduler
scheduler = get_scheduler('onecycle', optimizer, train_loader, epochs=100)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    use_mixup=True,
    use_amp=True
)

# Run training
best_acc = trainer.run(epochs=100)
trainer.save_results()
```

### 4. LR Finder

```python
from training import LRFinder

# Create LR Finder
lr_finder = LRFinder(model, optimizer, criterion, device, data_loader)

# Run range test
lrs, losses = lr_finder.range_test(start_lr=1e-6, end_lr=1.0, num_epochs=3)

# Get suggestions for OneCycle scheduler
suggested = lr_finder.suggest_scheduler_lrs('onecycle', method='steepest_gradient')
print(f"Max LR: {suggested['max_lr']}")
print(f"Base LR: {suggested['base_lr']}")
```

### 5. Utilities

```python
from utils import get_device, CheckpointManager, MetricsTracker
from utils import HuggingFaceUploader, create_model_card

# Device detection
device = get_device()  # Auto-detect
device = get_device('cuda')  # Force CUDA

# Checkpoint management
checkpoint_mgr = CheckpointManager()
checkpoint_mgr.save_checkpoint(model, optimizer, epoch, metrics, config, is_best=True)

# Metrics tracking
metrics = MetricsTracker()
metrics.update(train_loss, train_acc, test_loss, test_acc, lr)
metrics.save('metrics.json')
metrics.print_console_plots()

# HuggingFace upload
uploader = HuggingFaceUploader('username/repo', token='YOUR_TOKEN')
uploader.upload_checkpoint_files('./checkpoint_1')
```

## Adding New Components

### Adding a New Dataset

1. Create `data_loaders/cifar10.py`:

```python
from .base import BaseDataset
from .transforms import get_cifar_transforms
from torchvision import datasets

class CIFAR10Dataset(BaseDataset):
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)
    NUM_CLASSES = 10
    IMAGE_SIZE = (32, 32)

    def __init__(self, train=True, data_dir='../data', augmentation='strong', download=True):
        self.train = train
        self.transform = get_cifar_transforms(self.MEAN, self.STD, train, augmentation)
        self.dataset = datasets.CIFAR10(root=data_dir, train=train, download=download, transform=self.transform)

    # Implement required methods...
```

2. Register in `data_loaders/__init__.py`:

```python
from .cifar10 import CIFAR10Dataset

DATASETS = {
    'cifar100': CIFAR100Dataset,
    'cifar10': CIFAR10Dataset,  # Add this
}
```

3. Use it:

```bash
python train.py --dataset cifar10
```

### Adding a New Model

1. Create `models/resnet18.py` with your model class

2. Register in `models/__init__.py`:

```python
from .resnet18 import ResNet18

MODELS = {
    'wideresnet28-10': lambda num_classes=100: WideResNet(...),
    'resnet50': lambda num_classes=100: ResNet50(...),
    'resnet18': lambda num_classes=100: ResNet18(num_classes=num_classes),  # Add this
}
```

3. Use it:

```bash
python train.py --model resnet18
```

## Configuration File

The `config.json` file contains settings for LR Finder and schedulers:

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
    },
    "cosine": {
      "T_0": 25,
      "eta_min": 1e-4
    }
  }
}
```

## Migration from Old Code

### Old way (monolithic):

```python
from model import Net, get_optimizer, get_scheduler, train_transforms
model = Net()
```

### New way (modular):

```python
from models import get_model
from training import get_optimizer, get_scheduler
from data_loaders import get_dataset

model = get_model('wideresnet28-10')
optimizer = get_optimizer('sgd', model)
scheduler = get_scheduler('onecycle', optimizer, train_loader)
train_dataset = get_dataset('cifar100', train=True)
```

## Benefits

✅ **Modularity**: Each component has a single responsibility
✅ **Extensibility**: Add new datasets/models by creating new files
✅ **Reusability**: Import and use components independently
✅ **Testability**: Each module can be tested in isolation
✅ **Maintainability**: Clear structure, easier to navigate
✅ **Config-driven**: Easy to experiment with different settings

## Command Line Reference

```bash
# Model options
--model wideresnet28-10|resnet50

# Dataset options
--dataset cifar100

# Training options
--epochs 100
--batch-size 256
--device cuda|mps|cpu

# Optimizer & Scheduler
--optimizer sgd|adam|adamw
--scheduler cosine|onecycle

# Augmentation & Regularization
--augmentation none|weak|strong
--no-mixup
--mixup-alpha 0.2
--label-smoothing 0.1

# Advanced features
--no-amp                # Disable mixed precision
--gradient-clip 1.0
--lr-finder             # Run LR Finder before training

# HuggingFace
--hf-token TOKEN
--hf-repo username/repo
```

## Examples

### Example 1: Quick training run

```bash
python train.py --model wideresnet28-10 --epochs 10
```

### Example 2: Find optimal LR and train

```bash
python train.py --lr-finder --scheduler onecycle --epochs 50
```

### Example 3: Custom configuration

```bash
python train.py \
  --model resnet50 \
  --epochs 100 \
  --batch-size 128 \
  --augmentation weak \
  --no-mixup \
  --scheduler cosine
```

### Example 4: Upload to HuggingFace

```bash
python train.py \
  --model wideresnet28-10 \
  --epochs 100 \
  --lr-finder \
  --hf-token YOUR_TOKEN \
  --hf-repo username/cifar100-wideresnet
```

## Testing Individual Components

```python
# Test dataset
from data_loaders import get_dataset
ds = get_dataset('cifar100')
print(len(ds))  # 50000

# Test model
from models import get_model
model = get_model('wideresnet28-10')
print(sum(p.numel() for p in model.parameters()))  # Parameter count

# Test device detection
from utils import get_device
device = get_device()
print(device)  # cuda/mps/cpu
```

## Troubleshooting

**Import errors**: Make sure you're in the `session8-more-trials` directory

**CUDA errors**: Check device with `python -c "import torch; print(torch.cuda.is_available())"`

**Dataset not found**: Run with `download=True` or manually download to `../data/`

**Config not loading**: Check that `config.json` exists in the current directory

## Further Reading

- Dataset implementation: See `data_loaders/cifar100.py`
- Model architectures: See `models/`
- Training loop: See `training/trainer.py`
- LR Finder details: See `training/lr_finder.py`
