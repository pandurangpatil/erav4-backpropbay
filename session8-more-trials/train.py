import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm
import importlib
from torchsummary import summary
from torchinfo import summary as torchinfo_summary
import matplotlib.pyplot as plt
import plotille
import numpy as np
import os
import json
from datetime import datetime
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# HuggingFace Hub imports (optional)
try:
    from huggingface_hub import HfApi, create_repo, upload_file
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")


# ------------------------------
# GPU Detection and Configuration
# ------------------------------
def detect_gpu_device(preferred_device=None):
    """
    Detect and configure the best available device for training.

    Args:
        preferred_device: Optional device string ('cuda', 'cuda:0', 'mps', 'cpu')

    Returns:
        torch.device: The configured device
    """
    print("\n" + "="*70)
    print("GPU DETECTION AND CONFIGURATION")
    print("="*70)

    # If user specified a device, try to use it
    if preferred_device:
        try:
            device = torch.device(preferred_device)
            print(f"Using user-specified device: {device}")
            return device
        except Exception as e:
            print(f"Warning: Could not use specified device '{preferred_device}': {e}")
            print("Falling back to automatic detection...")

    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()

        print(f"✓ CUDA is available")
        print(f"✓ Number of GPUs: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Memory: {gpu_memory:.2f} GB")

        current_device = torch.cuda.current_device()
        print(f"✓ Using GPU: {torch.cuda.get_device_name(current_device)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")

    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Apple MPS (Metal Performance Shaders) is available")
        print(f"✓ Using device: {device}")

    # Fallback to CPU
    else:
        device = torch.device('cpu')
        print(f"⚠ No GPU detected. Using CPU")
        print(f"  Training will be significantly slower on CPU")
        print(f"  Consider using a machine with CUDA-capable GPU")

    print(f"✓ PyTorch Version: {torch.__version__}")
    print("="*70 + "\n")

    return device


# ------------------------------
# Helper Functions
# ------------------------------
def load_scheduler_config(config_path='./config.json'):
    """
    Load scheduler configuration from JSON file.

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        dict: OneCycleLR configuration parameters, or None if file not found
    """
    if not os.path.exists(config_path):
        print(f"⚠ Config file not found: {config_path}")
        print("  Using default OneCycleLR parameters")
        return None

    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Extract OneCycle scheduler config
        if 'scheduler' in config_data and 'onecycle' in config_data['scheduler']:
            onecycle_config = config_data['scheduler']['onecycle']
            print(f"✓ Loaded OneCycleLR config from: {config_path}")
            print(f"  Parameters: {onecycle_config}")
            return onecycle_config
        else:
            print(f"⚠ No 'scheduler.onecycle' section found in {config_path}")
            print("  Using default OneCycleLR parameters")
            return None
    except json.JSONDecodeError as e:
        print(f"⚠ Error parsing JSON config file: {e}")
        print("  Using default OneCycleLR parameters")
        return None
    except Exception as e:
        print(f"⚠ Error loading config file: {e}")
        print("  Using default OneCycleLR parameters")
        return None


def mixup_data(x, y, alpha=0.2, device=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)

    # Auto-detect device if not provided
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1

    def is_warmup(self):
        return self.current_step < self.warmup_steps


# ------------------------------
# HuggingFace Upload Functions
# ------------------------------
def upload_to_huggingface(file_path, path_in_repo, repo_id, hf_token, commit_message="Upload file"):
    """Upload a file to HuggingFace Hub"""
    if not HF_AVAILABLE or not hf_token:
        print(f"⚠ Skipping upload of {path_in_repo} (HuggingFace not available or no token)")
        return

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=commit_message
        )
        print(f"✓ Uploaded: {path_in_repo}")
    except Exception as e:
        print(f"✗ Upload failed for {path_in_repo}: {e}")


def create_model_card(repo_id, best_acc, total_epochs, train_accuracies, test_accuracies,
                      train_losses, test_losses, config):
    """Create README.md model card"""
    model_card = f"""---
tags:
- image-classification
- cifar100
- wideresnet
- pytorch
datasets:
- cifar100
metrics:
- accuracy
---

# CIFAR-100 WideResNet-28-10

## Model Description

WideResNet-28-10 trained on CIFAR-100 dataset with advanced augmentation techniques.

### Model Architecture
- **Architecture**: WideResNet-28-10
- **Parameters**: ~36.5M
- **Depth**: 28 layers
- **Width Factor**: 10
- **Dropout**: 0.3

### Training Configuration
- **Batch Size**: {config.get('batch_size', 256)}
- **Optimizer**: SGD (momentum=0.9, weight_decay=1e-3)
- **Learning Rate**: Cosine annealing with warmup (0.01→0.1, min=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=25)
- **Augmentation**: HorizontalFlip, ShiftScaleRotate, Cutout, ColorJitter
- **MixUp**: Alpha={config.get('mixup_alpha', 0.2)}
- **Label Smoothing**: {config.get('label_smoothing', 0.1)}
- **Mixed Precision**: {config.get('mixed_precision', True)}
- **Gradient Clipping**: {config.get('gradient_clipping', 1.0)}

### Performance
- **Best Test Accuracy**: {best_acc:.2f}%
- **Total Epochs Trained**: {total_epochs}
- **Final Train Accuracy**: {train_accuracies[-1]:.2f}%
- **Final Test Accuracy**: {test_accuracies[-1]:.2f}%

### Training History
- **Best Epoch**: {test_accuracies.index(max(test_accuracies)) + 1}
- **Train Loss**: {train_losses[0]:.4f} → {train_losses[-1]:.4f}
- **Test Loss**: {test_losses[0]:.4f} → {test_losses[-1]:.4f}

### Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
checkpoint_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="best_model.pth"
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load model (define WideResNet class first)
from model import WideResNet
model = WideResNet(depth=28, widen_factor=10, num_classes=100)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Training Details
- **Dataset**: CIFAR-100 (50,000 train, 10,000 test)
- **Classes**: 100
- **Image Size**: 32×32
- **Normalization**: mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761)

### Files
- `best_model.pth` - Best performing model checkpoint
- `training_curves.png` - Training/test accuracy and loss curves
- `metrics.json` - Complete training history
- `config.json` - Hyperparameter configuration

### License
MIT

### Citation
```bibtex
@misc{{wideresnet-cifar100,
  author = {{Your Name}},
  title = {{CIFAR-100 WideResNet-28-10}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""
    return model_card


# ------------------------------
# CIFAR Trainer Class
# ------------------------------
class CIFARTrainer:
    def _get_next_checkpoint_folder(self):
        """
        Find the next available checkpoint folder in sequence.
        Scans for existing checkpoint_N folders and returns the next number.

        Returns:
            str: Path to the next checkpoint folder (e.g., './checkpoint_3')
        """
        import re

        # Get all directories in current path
        existing_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]

        # Filter for checkpoint folders matching pattern checkpoint_N
        checkpoint_pattern = re.compile(r'^checkpoint_(\d+)$')
        checkpoint_numbers = []

        for dirname in existing_dirs:
            match = checkpoint_pattern.match(dirname)
            if match:
                checkpoint_numbers.append(int(match.group(1)))

        # Find next available number
        if checkpoint_numbers:
            next_num = max(checkpoint_numbers) + 1
        else:
            next_num = 1

        next_folder = f'./checkpoint_{next_num}'
        print(f"📁 Creating checkpoint folder: {next_folder}")
        return next_folder

    def __init__(self, model_module_name='model', epochs=100, batch_size=256,
                 use_mixup=True, mixup_alpha=0.2, label_smoothing=0.1,
                 use_amp=True, gradient_clip=1.0, warmup_epochs=5,
                 checkpoint_epochs=None, hf_token=None, hf_repo_id=None,
                 device=None, scheduler_type='cosine', scheduler_config_path=None):
        """
        Initialize the CIFAR-100 trainer with advanced features.

        Args:
            model_module_name: Name of the module containing the model
            epochs: Number of training epochs
            batch_size: Batch size for training and testing
            use_mixup: Enable MixUp augmentation
            mixup_alpha: MixUp alpha parameter
            label_smoothing: Label smoothing parameter
            use_amp: Use automatic mixed precision training
            gradient_clip: Gradient clipping max norm
            warmup_epochs: Number of warmup epochs
            checkpoint_epochs: List of epochs to save checkpoints
            hf_token: HuggingFace API token
            hf_repo_id: HuggingFace repository ID
            device: Device to use for training ('cuda', 'mps', 'cpu', or None for auto-detect)
            scheduler_type: Learning rate scheduler type ('cosine' or 'onecycle')
            scheduler_config_path: Path to scheduler config JSON file (for OneCycleLR)
        """
        self.model_module = importlib.import_module(model_module_name)
        self.epochs = epochs
        self.batch_size = batch_size

        # Detect and configure GPU/device
        self.device = detect_gpu_device(preferred_device=device)

        # Training configuration
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        self.checkpoint_epochs = checkpoint_epochs or [10, 20, 25, 30, 40, 50, 60, 75, 90]

        # Scheduler configuration
        self.scheduler_type = scheduler_type.lower()
        self.scheduler_config_path = scheduler_config_path

        # HuggingFace configuration
        self.hf_token = hf_token
        self.hf_repo_id = hf_repo_id

        # Initialize model
        self.model = self.model_module.Net().to(self.device)

        # Get transforms from model module
        self.train_transforms = self.model_module.train_transforms
        self.test_transforms = self.model_module.test_transforms

        # Setup datasets and data loaders
        self._setup_data()

        # Get optimizer from model module
        self.optimizer = self.model_module.get_optimizer(self.model)

        # Load scheduler configuration if using OneCycleLR
        onecycle_config = None
        if self.scheduler_type == 'onecycle':
            if self.scheduler_config_path:
                onecycle_config = load_scheduler_config(self.scheduler_config_path)
            else:
                # Try default location
                onecycle_config = load_scheduler_config('./config.json')

        # Get scheduler from model module with configuration
        self.scheduler = self.model_module.get_scheduler(
            self.optimizer,
            self.train_loader,
            scheduler_type=self.scheduler_type,
            epochs=self.epochs,
            onecycle_config=onecycle_config
        )

        # Warmup scheduler (only used for CosineAnnealing, OneCycleLR has built-in warmup)
        self.use_custom_warmup = (self.scheduler_type == 'cosine')
        if self.use_custom_warmup:
            self.warmup_scheduler = WarmupScheduler(
                self.optimizer, self.warmup_epochs, 0.01, 0.1, len(self.train_loader)
            )
        else:
            self.warmup_scheduler = None

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Initialize metric tracking
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.learning_rates = []

        # Checkpoint tracking
        self.best_test_acc = 0.0
        self.best_epoch = 0
        self.checkpoint_dir = self._get_next_checkpoint_folder()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _setup_data(self):
        """Setup CIFAR-100 datasets and data loaders."""
        self.train_dataset = datasets.CIFAR100(
            '../data',
            train=True,
            download=True,
            transform=self.train_transforms
        )
        self.test_dataset = datasets.CIFAR100(
            '../data',
            train=False,
            transform=self.test_transforms
        )

        # Only use pin_memory on CUDA devices (not supported on MPS)
        use_pin_memory = self.device.type == 'cuda'

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin_memory
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_pin_memory
        )

    def train(self, epoch):
        """
        Train the model for one epoch with MixUp and label smoothing.

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (average_loss, accuracy) for the epoch
        """
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Mixed precision training
            if self.use_amp:
                with autocast(device_type=self.device.type):
                    if self.use_mixup:
                        inputs, targets_a, targets_b, lam = mixup_data(
                            data, target, alpha=self.mixup_alpha, device=self.device
                        )
                        outputs = self.model(inputs)
                        loss = lam * F.cross_entropy(outputs, targets_a, label_smoothing=self.label_smoothing) + \
                               (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=self.label_smoothing)
                    else:
                        outputs = self.model(data)
                        loss = F.cross_entropy(outputs, target, label_smoothing=self.label_smoothing)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without AMP
                if self.use_mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(
                        data, target, alpha=self.mixup_alpha, device=self.device
                    )
                    outputs = self.model(inputs)
                    loss = lam * F.cross_entropy(outputs, targets_a, label_smoothing=self.label_smoothing) + \
                           (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=self.label_smoothing)
                else:
                    outputs = self.model(data)
                    loss = F.cross_entropy(outputs, target, label_smoothing=self.label_smoothing)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.optimizer.step()

            # Update learning rate
            if self.scheduler_type == 'onecycle':
                # OneCycleLR steps per batch
                self.scheduler.step()
            else:
                # CosineAnnealing with custom warmup steps per epoch (handled later)
                if self.use_custom_warmup and self.warmup_scheduler.is_warmup():
                    self.warmup_scheduler.step()
                elif not self.use_custom_warmup or not self.warmup_scheduler.is_warmup():
                    # For cosine, step after epoch (not here)
                    pass

            # Accuracy tracking
            _, pred = outputs.max(1)
            if self.use_mixup:
                correct += lam * pred.eq(targets_a).sum().item() + (1 - lam) * pred.eq(targets_b).sum().item()
            else:
                correct += pred.eq(target).sum().item()
            processed += len(data)
            epoch_loss += loss.item()

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_description(
                f"Epoch {epoch} Loss={loss.item():.4f} "
                f"Acc={100*correct/processed:.2f}% LR={current_lr:.6f}"
            )

        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100. * correct / processed
        return avg_loss, accuracy

    def print_model_summary(self):
        """Print the model architecture summary."""
        print("\n" + "="*70)
        print("Model Architecture Summary")
        print("="*70)
        print(f"Device: {self.device}")

        # Display detailed GPU information if available
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(self.device)}")
            gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            print(f"GPU Memory: {gpu_memory_total:.2f} GB")

            # Show current memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f"Memory Allocated: {memory_allocated:.2f} GB")
                print(f"Memory Reserved: {memory_reserved:.2f} GB")

        elif self.device.type == 'mps':
            print("Device Type: Apple Metal Performance Shaders (MPS)")

        print("\nModel Summary:")
        # Use torchinfo for MPS devices (better compatibility)
        # Use torchsummary for CUDA/CPU (legacy support)
        if self.device.type == 'mps':
            torchinfo_summary(self.model, input_size=(1, 3, 32, 32), device=self.device)
        else:
            summary(self.model, input_size=(3, 32, 32), device=str(self.device.type))
        print("="*70 + "\n")

    def test(self):
        """
        Test the model and return loss and accuracy.

        Returns:
            tuple: (test_loss, accuracy) for the test set
        """
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        acc = 100. * correct / len(self.test_loader.dataset)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(self.test_loader.dataset)} ({acc:.2f}%)\n"
        )
        return test_loss, acc

    def save_checkpoint(self, epoch, train_acc, test_acc, train_loss, test_loss, is_best=False):
        """Save checkpoint locally"""
        checkpoint_name = 'best_model.pth' if is_best else f'checkpoint_epoch{epoch}.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': 'WideResNet-28-10',
                'batch_size': self.batch_size,
                'scheduler_type': self.scheduler_type,
                'mixup_alpha': self.mixup_alpha,
                'label_smoothing': self.label_smoothing,
                'weight_decay': 1e-3,
                'dropout': 0.3,
                'mixed_precision': self.use_amp,
                'gradient_clipping': self.gradient_clip
            }
        }

        torch.save(checkpoint, checkpoint_path)
        if is_best:
            print(f"💾 Saved best model: {checkpoint_path}")
        else:
            print(f"💾 Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def upload_final_results(self):
        """Save all artifacts locally and optionally upload to HuggingFace Hub"""
        print("\n" + "="*70)
        print("Saving training results...")
        print("="*70)

        # Save metrics locally (always)
        metrics = {
            'epochs': list(range(1, len(self.train_losses) + 1)),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'learning_rates': self.learning_rates,
            'best_test_accuracy': self.best_test_acc,
            'best_epoch': self.best_epoch
        }
        metrics_path = os.path.join(self.checkpoint_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Saved metrics: {metrics_path}")

        # Save config locally (always)
        config_full = {
            'model': 'WideResNet-28-10',
            'depth': 28,
            'widen_factor': 10,
            'dropout': 0.3,
            'num_classes': 100,
            'batch_size': self.batch_size,
            'epochs': len(self.train_losses),
            'optimizer': 'SGD',
            'momentum': 0.9,
            'weight_decay': 1e-3,
            'initial_lr': 0.01,
            'max_lr': 0.1,
            'min_lr': 1e-4,
            'scheduler': 'CosineAnnealingWarmRestarts',
            'T_0': 25,
            'warmup_epochs': self.warmup_epochs,
            'mixup_alpha': self.mixup_alpha,
            'label_smoothing': self.label_smoothing,
            'gradient_clipping': self.gradient_clip,
            'mixed_precision': self.use_amp,
            'best_test_accuracy': self.best_test_acc,
            'final_train_accuracy': self.train_accuracies[-1],
            'final_test_accuracy': self.test_accuracies[-1]
        }
        config_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_full, f, indent=2)
        print(f"✓ Saved config: {config_path}")

        # Create and save model card locally (always)
        config = {
            'batch_size': self.batch_size,
            'mixup_alpha': self.mixup_alpha,
            'label_smoothing': self.label_smoothing,
            'mixed_precision': self.use_amp,
            'gradient_clipping': self.gradient_clip
        }
        model_card = create_model_card(
            self.hf_repo_id or "local-model", self.best_test_acc, len(self.train_losses),
            self.train_accuracies, self.test_accuracies, self.train_losses, self.test_losses, config
        )
        readme_path = os.path.join(self.checkpoint_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(model_card)
        print(f"✓ Saved model card: {readme_path}")

        # Verify training curves exists
        curves_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
        if os.path.exists(curves_path):
            print(f"✓ Saved training curves: {curves_path}")

        # Verify best model exists
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"✓ Saved best model: {best_model_path}")

        print(f"\n✓ All files saved locally to: {self.checkpoint_dir}")

        # Upload to HuggingFace if credentials provided
        if not self.hf_token or not self.hf_repo_id:
            print("\n⚠ HuggingFace upload skipped (no token or repo ID provided)")
            print("  All artifacts have been saved locally in ./checkpoints/")
            print("="*70 + "\n")
            return

        if not HF_AVAILABLE:
            print("\n⚠ HuggingFace upload skipped (huggingface_hub not installed)")
            print("  All artifacts have been saved locally in ./checkpoints/")
            print("="*70 + "\n")
            return

        print("\n" + "="*70)
        print("Uploading to HuggingFace Hub...")
        print("="*70)

        try:
            # Create repository
            api = HfApi()
            create_repo(repo_id=self.hf_repo_id, repo_type="model", exist_ok=True, token=self.hf_token)
            print(f"✓ Repository ready: https://huggingface.co/{self.hf_repo_id}")
        except Exception as e:
            print(f"Warning: Could not create repository: {e}")

        # Upload best model
        if os.path.exists(best_model_path):
            upload_to_huggingface(
                best_model_path, 'best_model.pth', self.hf_repo_id, self.hf_token,
                commit_message=f"Best model: {self.best_test_acc:.2f}% accuracy"
            )

        # Upload metrics
        upload_to_huggingface(
            metrics_path, 'metrics.json', self.hf_repo_id, self.hf_token,
            commit_message="Upload training metrics"
        )

        # Upload training curves
        if os.path.exists(curves_path):
            upload_to_huggingface(
                curves_path, 'training_curves.png', self.hf_repo_id, self.hf_token,
                commit_message="Upload training curves"
            )

        # Upload model card
        upload_to_huggingface(
            readme_path, 'README.md', self.hf_repo_id, self.hf_token,
            commit_message="Update model card"
        )

        # Upload config
        upload_to_huggingface(
            config_path, 'config.json', self.hf_repo_id, self.hf_token,
            commit_message="Upload training configuration"
        )

        print(f"✓ All files uploaded to: https://huggingface.co/{self.hf_repo_id}")
        print("="*70 + "\n")

    def plot_metrics(self):
        """Plot training and testing metrics (loss and accuracy)."""
        epochs = list(range(1, len(self.train_losses) + 1))

        # Print console-based plots
        print("\n" + "="*80)
        print("TRAINING AND TESTING LOSS")
        print("="*80)

        # Create loss plot
        fig = plotille.Figure()
        fig.width = 70
        fig.height = 20
        fig.color_mode = 'byte'
        fig.set_x_limits(min_=1, max_=len(epochs))
        fig.set_y_limits(min_=0, max_=max(max(self.train_losses), max(self.test_losses)) * 1.1)

        # Plot training and testing loss
        fig.plot(epochs, self.train_losses, lc=25, label='Training Loss')  # Blue
        fig.plot(epochs, self.test_losses, lc=196, label='Testing Loss')   # Red

        print(fig.show(legend=True))

        print("\n" + "="*80)
        print("TRAINING AND TESTING ACCURACY")
        print("="*80)

        # Create accuracy plot
        fig = plotille.Figure()
        fig.width = 70
        fig.height = 20
        fig.color_mode = 'byte'
        fig.set_x_limits(min_=1, max_=len(epochs))
        fig.set_y_limits(min_=min(min(self.train_accuracies), min(self.test_accuracies)) * 0.95,
                         max_=100)

        # Plot training and testing accuracy
        fig.plot(epochs, self.train_accuracies, lc=25, label='Training Accuracy')  # Blue
        fig.plot(epochs, self.test_accuracies, lc=196, label='Testing Accuracy')   # Red

        print(fig.show(legend=True))
        print("="*80 + "\n")

        # Save matplotlib plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        axs[0, 0].plot(self.train_losses, label='Train Loss')
        axs[0, 0].plot(self.test_losses, label='Test Loss')
        axs[0, 0].set_title("Loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axs[0, 1].plot(self.test_accuracies, label='Test Accuracy')
        axs[0, 1].axhline(y=74, color='r', linestyle='--', label='Target (74%)')
        axs[0, 1].set_title("Accuracy")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Accuracy (%)")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(self.learning_rates, color='green')
        axs[1, 0].set_title("Learning Rate Schedule")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Learning Rate")
        axs[1, 0].grid(True)

        # Gap between train and test accuracy
        accuracy_gap = [train - test for train, test in zip(self.train_accuracies, self.test_accuracies)]
        axs[1, 1].plot(accuracy_gap, color='orange')
        axs[1, 1].set_title("Train-Test Accuracy Gap")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Gap (%)")
        axs[1, 1].grid(True)

        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(self.checkpoint_dir, 'training_curves.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Training curves saved as '{plot_filename}'\n")

        plt.close()

    def run(self):
        """Run the complete training process for all epochs."""
        print(f"Training {self.model_module.__name__} for {self.epochs} epochs")
        print("="*70)
        print("Configuration:")
        print(f"  - Model: WideResNet-28-10 (36.5M parameters)")
        print(f"  - Dataset: CIFAR-100")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Scheduler: {self.scheduler_type.upper()}")
        if self.scheduler_type == 'onecycle':
            print(f"    • OneCycleLR config: {self.scheduler_config_path or 'defaults'}")
        else:
            print(f"    • CosineAnnealingWarmRestarts (T_0=25, eta_min=1e-4)")
            print(f"    • Warmup Epochs: {self.warmup_epochs}")
        print(f"  - MixUp: {self.use_mixup} (alpha={self.mixup_alpha})")
        print(f"  - Label Smoothing: {self.label_smoothing}")
        print(f"  - Mixed Precision: {self.use_amp}")
        print(f"  - Gradient Clipping: {self.gradient_clip}")
        print(f"  - Checkpoint Epochs: {self.checkpoint_epochs}")
        if self.hf_repo_id and self.hf_token:
            print(f"  - HuggingFace Repo: {self.hf_repo_id}")
        else:
            print("  - HuggingFace Upload: DISABLED (no token or repo provided)")
            print("  ⚠ Warning: Final model will NOT be uploaded to HuggingFace")
            print("  ℹ Models will be saved locally to ./checkpoints/")
        print("="*70)

        # Print model summary before training
        self.print_model_summary()

        patience = 15
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train(epoch)
            test_loss, test_acc = self.test()

            # Step scheduler after epoch for CosineAnnealing (OneCycle steps per batch)
            if self.scheduler_type == 'cosine':
                if self.use_custom_warmup and not self.warmup_scheduler.is_warmup():
                    self.scheduler.step()
                elif not self.use_custom_warmup:
                    self.scheduler.step()

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            # Save best model
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_epoch = epoch
                patience_counter = 0
                print(f"*** New best model! Test Accuracy: {self.best_test_acc:.2f}% ***")
                self.save_checkpoint(epoch, train_acc, test_acc, train_loss, test_loss, is_best=True)
            else:
                patience_counter += 1

            # Save checkpoint at breakpoints
            if epoch in self.checkpoint_epochs:
                print(f"📍 Breakpoint checkpoint at epoch {epoch}")
                self.save_checkpoint(epoch, train_acc, test_acc, train_loss, test_loss)

            # Early stopping check
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}. No improvement for {patience} epochs.")
                break

            # Check if target reached
            if test_acc >= 74.0:
                print(f"\n{'=' * 70}")
                print(f"Target accuracy of 74% reached at epoch {epoch}!")
                print(f"Final test accuracy: {test_acc:.2f}%")
                print(f"{'=' * 70}")
                break

            print(f"Best Test Accuracy so far: {self.best_test_acc:.2f}% | Patience: {patience_counter}/{patience}\n")

        print(f"\nTraining completed. Best test accuracy: {self.best_test_acc:.2f}% (epoch {self.best_epoch})")

        # Plot metrics
        self.plot_metrics()

        # Upload to HuggingFace
        self.upload_final_results()

        return self.best_test_acc


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CIFAR-100 model')
    parser.add_argument('--model', type=str, default='model',
                       help='Model module name to use for training (default: model)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training: cuda, cuda:0, mps, cpu (default: auto-detect)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'onecycle'],
                       help='Learning rate scheduler type: cosine or onecycle (default: cosine)')
    parser.add_argument('--scheduler-config', type=str, default=None,
                       help='Path to scheduler config JSON file (for OneCycleLR, default: ./config.json)')
    parser.add_argument('--hf-token', type=str, default=None,
                       help='HuggingFace API token for model upload')
    parser.add_argument('--hf-repo', type=str, default=None,
                       help='HuggingFace repository ID (e.g., username/repo-name)')
    args = parser.parse_args()

    # Create trainer with selected model and run training
    trainer = CIFARTrainer(
        model_module_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        scheduler_type=args.scheduler,
        scheduler_config_path=args.scheduler_config,
        hf_token=args.hf_token,
        hf_repo_id=args.hf_repo
    )
    trainer.run()
