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
# LR Finder for Automatic LR Selection
# ------------------------------
class LRFinder:
    """
    Learning Rate Finder using the range test method.
    Based on Leslie Smith's paper: "Cyclical Learning Rates for Training Neural Networks"
    """
    def __init__(self, model, optimizer, criterion, device, data_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.data_loader = data_loader

        # Storage for results
        self.lrs = []
        self.losses = []

        # State management
        self.best_loss = float('inf')
        self.initial_model_state = None
        self.initial_optimizer_state = None

    def range_test(self, start_lr=1e-6, end_lr=1.0, num_epochs=3, smoothing=0.05):
        """
        Run the LR range test.

        Args:
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_epochs: Number of epochs to run
            smoothing: Exponential smoothing factor for loss

        Returns:
            tuple: (learning_rates, losses)
        """
        print("\n" + "="*70)
        print("LEARNING RATE FINDER - RANGE TEST")
        print("="*70)
        print(f"Testing learning rates from {start_lr:.2e} to {end_lr:.2e}")
        print(f"Running for {num_epochs} epochs")
        print(f"Note: Using clean data (no MixUp, no Label Smoothing)")
        print("="*70 + "\n")

        # Save initial state
        self.initial_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.initial_optimizer_state = self.optimizer.state_dict()

        # Calculate total iterations
        total_iters = len(self.data_loader) * num_epochs
        lr_lambda = lambda x: np.exp(x * np.log(end_lr / start_lr) / total_iters)

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr

        # Run training
        self.model.train()
        iteration = 0
        smoothed_loss = 0.0

        for epoch in range(num_epochs):
            pbar = tqdm(self.data_loader, desc=f"LR Finder Epoch {epoch+1}/{num_epochs}")

            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass (no mixup, no label smoothing)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target)

                # Track smoothed loss
                if iteration == 0:
                    smoothed_loss = loss.item()
                else:
                    smoothed_loss = smoothing * loss.item() + (1 - smoothing) * smoothed_loss

                # Check for divergence
                if smoothed_loss > 4 * self.best_loss or torch.isnan(loss):
                    print(f"\n⚠ Loss diverging (current: {smoothed_loss:.4f}, best: {self.best_loss:.4f})")
                    print("Stopping LR range test early")
                    break

                # Update best loss
                if smoothed_loss < self.best_loss:
                    self.best_loss = smoothed_loss

                # Store LR and loss
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lrs.append(current_lr)
                self.losses.append(smoothed_loss)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Update learning rate
                iteration += 1
                new_lr = start_lr * lr_lambda(iteration)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

                pbar.set_postfix({'lr': f'{current_lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})

            # Early stopping if loss diverged
            if smoothed_loss > 4 * self.best_loss or torch.isnan(loss):
                break

        # Restore initial state
        print("\nRestoring initial model and optimizer state...")
        self.model.load_state_dict(self.initial_model_state)
        self.optimizer.load_state_dict(self.initial_optimizer_state)

        print(f"✓ LR range test completed: {len(self.lrs)} data points collected\n")
        return self.lrs, self.losses

    def find_steepest_gradient(self, skip_start=10, skip_end=5):
        """
        Find LR with steepest negative gradient (fastest learning).

        Args:
            skip_start: Skip first N points (noisy at very low LR)
            skip_end: Skip last N points (diverging region)

        Returns:
            float: Suggested learning rate
        """
        if len(self.losses) < skip_start + skip_end + 10:
            print("⚠ Not enough data points for gradient calculation")
            return None

        # Calculate gradients
        losses = np.array(self.losses[skip_start:-skip_end if skip_end > 0 else None])
        lrs = np.array(self.lrs[skip_start:-skip_end if skip_end > 0 else None])

        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)

        suggested_lr = lrs[min_gradient_idx]
        print(f"  Method: Steepest Gradient")
        print(f"  → Suggested LR: {suggested_lr:.6f}")
        print(f"  → Loss at this point: {losses[min_gradient_idx]:.4f}")

        return suggested_lr

    def find_before_divergence(self, threshold=0.1, skip_start=10):
        """
        Find LR just before loss starts increasing significantly.

        Args:
            threshold: Percentage increase to consider as divergence
            skip_start: Skip first N points

        Returns:
            float: Suggested learning rate
        """
        if len(self.losses) < skip_start + 10:
            print("⚠ Not enough data points")
            return None

        losses = np.array(self.losses[skip_start:])
        lrs = np.array(self.lrs[skip_start:])

        # Find minimum loss
        min_loss_idx = np.argmin(losses)
        min_loss = losses[min_loss_idx]

        # Find where loss increases by threshold percentage
        for i in range(min_loss_idx, len(losses)):
            if losses[i] > min_loss * (1 + threshold):
                # Go back a bit for safety
                safe_idx = max(0, i - 5)
                suggested_lr = lrs[safe_idx]
                print(f"  Method: Before Divergence")
                print(f"  → Suggested LR: {suggested_lr:.6f}")
                print(f"  → Loss at this point: {losses[safe_idx]:.4f}")
                return suggested_lr

        # If no divergence found, use LR at minimum loss
        suggested_lr = lrs[min_loss_idx]
        print(f"  Method: Before Divergence (no divergence detected)")
        print(f"  → Suggested LR: {suggested_lr:.6f}")
        print(f"  → Loss at this point: {min_loss:.4f}")

        return suggested_lr

    def find_valley(self, skip_start=10, skip_end=5):
        """
        Find LR at the valley (minimum loss point).

        Args:
            skip_start: Skip first N points
            skip_end: Skip last N points

        Returns:
            float: Suggested learning rate
        """
        if len(self.losses) < skip_start + skip_end + 10:
            print("⚠ Not enough data points")
            return None

        losses = np.array(self.losses[skip_start:-skip_end if skip_end > 0 else None])
        lrs = np.array(self.lrs[skip_start:-skip_end if skip_end > 0 else None])

        min_idx = np.argmin(losses)
        suggested_lr = lrs[min_idx]

        print(f"  Method: Valley (Minimum Loss)")
        print(f"  → Suggested LR: {suggested_lr:.6f}")
        print(f"  → Loss at this point: {losses[min_idx]:.4f}")

        return suggested_lr

    def suggest_lr(self, method='steepest_gradient'):
        """
        Suggest learning rate based on the selected method.

        Args:
            method: One of 'steepest_gradient', 'before_divergence', 'valley', 'manual'

        Returns:
            float or None: Suggested learning rate (None for manual)
        """
        print("\n" + "="*70)
        print("LR FINDER ANALYSIS")
        print("="*70)

        if method == 'steepest_gradient':
            return self.find_steepest_gradient()
        elif method == 'before_divergence':
            return self.find_before_divergence()
        elif method == 'valley':
            return self.find_valley()
        elif method == 'manual':
            print(f"  Method: Manual Selection")
            print(f"  → Please inspect the plot and select LR manually")
            print(f"  → LR range tested: {self.lrs[0]:.2e} to {self.lrs[-1]:.2e}")
            return None
        else:
            print(f"⚠ Unknown method: {method}, defaulting to steepest_gradient")
            return self.find_steepest_gradient()

    def plot(self, save_path=None, suggested_lr=None, max_lr=None, base_lr=None):
        """
        Plot the LR finder results.

        Args:
            save_path: Path to save the plot
            suggested_lr: Single LR to mark (for CosineAnnealing)
            max_lr: Max LR to mark (for OneCycle)
            base_lr: Base LR to mark (for OneCycle)
        """
        if len(self.lrs) == 0:
            print("⚠ No data to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.lrs, self.losses, linewidth=2, label='Loss')
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Learning Rate Finder - Range Test', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Mark suggested LR points
        if suggested_lr is not None:
            plt.axvline(x=suggested_lr, color='red', linestyle='--', linewidth=2, label=f'Suggested LR: {suggested_lr:.6f}')

        if max_lr is not None:
            plt.axvline(x=max_lr, color='green', linestyle='--', linewidth=2, label=f'Max LR: {max_lr:.6f}')

        if base_lr is not None:
            plt.axvline(x=base_lr, color='blue', linestyle='--', linewidth=2, label=f'Base LR: {base_lr:.6f}')

        plt.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ LR Finder plot saved to: {save_path}")

        plt.close()

    def suggest_scheduler_lrs(self, scheduler_type, method='steepest_gradient'):
        """
        Suggest learning rates specific to the scheduler type.

        For OneCycle: Suggests both max_lr and base_lr
        For CosineAnnealing: Suggests initial_lr

        Args:
            scheduler_type: 'onecycle' or 'cosine'
            method: LR selection method

        Returns:
            dict: Dictionary with suggested LR values
        """
        if scheduler_type == 'onecycle':
            return self._suggest_onecycle_lrs(method)
        else:  # cosine
            return self._suggest_cosine_lr(method)

    def _suggest_onecycle_lrs(self, method):
        """
        Suggest max_lr and base_lr for OneCycleLR scheduler.

        Strategy:
        1. Find max_lr using the selected method
        2. Find base_lr at an earlier point in the curve (1/10 to 1/5 of max_lr)
        """
        # Find max_lr using the selected method
        max_lr = self.suggest_lr(method)

        if max_lr is None:
            print("⚠ Could not determine max_lr")
            return {'max_lr': None, 'base_lr': None}

        # Find base_lr: look for a safe starting point
        # Strategy: Find LR where loss is still decreasing but at ~20% of max_lr
        base_lr = max_lr / 10.0  # Default to 1/10

        # Try to find a more optimal base_lr by analyzing early portion of curve
        if len(self.lrs) > 20:
            # Find index of max_lr
            max_lr_idx = min(range(len(self.lrs)), key=lambda i: abs(self.lrs[i] - max_lr))

            # Look at first 30% of the range before max_lr
            early_portion = int(max_lr_idx * 0.3)
            if early_portion > 10:
                # Find where loss starts decreasing significantly
                early_losses = np.array(self.losses[:early_portion])
                early_lrs = np.array(self.lrs[:early_portion])

                # Find steepest descent in early portion
                if len(early_losses) > 5:
                    gradients = np.gradient(early_losses)
                    # Find where significant learning starts (gradient becomes more negative)
                    threshold = np.percentile(gradients, 20)  # 20th percentile of gradients
                    significant_learning_idx = np.where(gradients < threshold)[0]

                    if len(significant_learning_idx) > 0:
                        base_lr_idx = significant_learning_idx[0]
                        base_lr = early_lrs[base_lr_idx]

        print(f"\n{'='*70}")
        print(f"ONECYCLE LR RECOMMENDATIONS")
        print(f"{'='*70}")
        print(f"  Max LR:  {max_lr:.6f} (peak learning rate)")
        print(f"  Base LR: {base_lr:.6f} (starting learning rate)")
        print(f"  Ratio:   {max_lr/base_lr:.1f}x (max_lr / base_lr)")
        print(f"{'='*70}\n")

        return {'max_lr': max_lr, 'base_lr': base_lr}

    def _suggest_cosine_lr(self, method):
        """
        Suggest initial_lr for CosineAnnealingWarmRestarts scheduler.
        """
        initial_lr = self.suggest_lr(method)

        if initial_lr is None:
            print("⚠ Could not determine initial_lr")
            return {'initial_lr': None}

        print(f"\n{'='*70}")
        print(f"COSINE ANNEALING LR RECOMMENDATIONS")
        print(f"{'='*70}")
        print(f"  Initial LR: {initial_lr:.6f}")
        print(f"  This will be used with existing T_0 and eta_min parameters")
        print(f"{'='*70}\n")

        return {'initial_lr': initial_lr}


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
                 device=None, scheduler_type='cosine', scheduler_config_path=None,
                 use_lr_finder=False, lr_finder_config=None):
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
            use_lr_finder: Enable LR Finder to automatically determine optimal learning rates
            lr_finder_config: LR Finder configuration dict (num_epochs, start_lr, end_lr, selection_method)
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

        # LR Finder configuration
        self.use_lr_finder = use_lr_finder
        self.lr_finder_config = lr_finder_config or {
            'num_epochs': 3,
            'start_lr': 1e-6,
            'end_lr': 1.0,
            'selection_method': 'steepest_gradient'
        }

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

    def run_lr_finder(self):
        """
        Run the LR Finder to automatically determine optimal learning rates.

        Returns:
            dict: Dictionary with suggested LR values based on scheduler type
        """
        print("\n" + "="*70)
        print("STARTING LR FINDER")
        print("="*70)

        # Create a clean data loader (no mixup, no label smoothing)
        # We'll temporarily use the test transforms to get clean data
        clean_dataset = datasets.CIFAR100(
            '../data',
            train=True,
            download=False,
            transform=self.test_transforms  # Use test transforms for clean data
        )

        # Only use pin_memory on CUDA devices
        use_pin_memory = self.device.type == 'cuda'

        clean_loader = torch.utils.data.DataLoader(
            clean_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        # Create LR Finder with clean criterion (no label smoothing)
        criterion = torch.nn.CrossEntropyLoss()
        lr_finder = LRFinder(
            model=self.model,
            optimizer=self.optimizer,
            criterion=criterion,
            device=self.device,
            data_loader=clean_loader
        )

        # Run range test
        num_epochs = self.lr_finder_config.get('num_epochs', 3)
        start_lr = self.lr_finder_config.get('start_lr', 1e-6)
        end_lr = self.lr_finder_config.get('end_lr', 1.0)
        selection_method = self.lr_finder_config.get('selection_method', 'steepest_gradient')

        lr_finder.range_test(
            start_lr=start_lr,
            end_lr=end_lr,
            num_epochs=num_epochs
        )

        # Get scheduler-specific LR suggestions
        suggested_lrs = lr_finder.suggest_scheduler_lrs(
            scheduler_type=self.scheduler_type,
            method=selection_method
        )

        # Plot and save
        plot_path = os.path.join(self.checkpoint_dir, 'lr_finder_plot.png')

        if self.scheduler_type == 'onecycle':
            lr_finder.plot(
                save_path=plot_path,
                max_lr=suggested_lrs.get('max_lr'),
                base_lr=suggested_lrs.get('base_lr')
            )
        else:  # cosine
            lr_finder.plot(
                save_path=plot_path,
                suggested_lr=suggested_lrs.get('initial_lr')
            )

        print(f"\n{'='*70}")
        print(f"LR FINDER COMPLETED")
        print(f"{'='*70}\n")

        return suggested_lrs

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

        # Run LR Finder if enabled
        if self.use_lr_finder:
            suggested_lrs = self.run_lr_finder()

            # Update scheduler with found LRs
            if suggested_lrs:
                if self.scheduler_type == 'onecycle':
                    max_lr = suggested_lrs.get('max_lr')
                    base_lr = suggested_lrs.get('base_lr')

                    if max_lr is not None and base_lr is not None:
                        print(f"\n{'='*70}")
                        print(f"UPDATING ONECYCLE SCHEDULER WITH LR FINDER RESULTS")
                        print(f"{'='*70}")
                        print(f"  Previous max_lr: {self.scheduler.max_lrs[0]:.6f}")
                        print(f"  New max_lr:      {max_lr:.6f}")
                        print(f"  New base_lr:     {base_lr:.6f}")
                        print(f"  (base_lr will be set via div_factor = max_lr / base_lr)")
                        print(f"{'='*70}\n")

                        # Calculate div_factor from base_lr
                        div_factor = max_lr / base_lr

                        # Get current onecycle config
                        onecycle_config = {
                            'max_lr': max_lr,
                            'div_factor': div_factor,
                            'pct_start': self.scheduler.pct_start,
                            'anneal_strategy': 'cos',
                            'final_div_factor': self.scheduler.final_div_factor,
                            'three_phase': self.scheduler.three_phase
                        }

                        # Recreate scheduler with new LR
                        self.scheduler = self.model_module.get_scheduler(
                            self.optimizer,
                            self.train_loader,
                            scheduler_type='onecycle',
                            epochs=self.epochs,
                            onecycle_config=onecycle_config
                        )

                else:  # cosine
                    initial_lr = suggested_lrs.get('initial_lr')

                    if initial_lr is not None:
                        print(f"\n{'='*70}")
                        print(f"UPDATING COSINE ANNEALING SCHEDULER WITH LR FINDER RESULTS")
                        print(f"{'='*70}")
                        print(f"  Previous initial LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                        print(f"  New initial LR:      {initial_lr:.6f}")
                        print(f"{'='*70}\n")

                        # Update optimizer LR
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = initial_lr

                        # Update warmup scheduler if using custom warmup
                        if self.use_custom_warmup:
                            self.warmup_scheduler = WarmupScheduler(
                                self.optimizer, self.warmup_epochs, initial_lr * 0.1, initial_lr, len(self.train_loader)
                            )

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
    parser.add_argument('--lr-finder', action='store_true',
                       help='Run LR Finder before training to automatically determine optimal learning rates')
    parser.add_argument('--hf-token', type=str, default=None,
                       help='HuggingFace API token for model upload')
    parser.add_argument('--hf-repo', type=str, default=None,
                       help='HuggingFace repository ID (e.g., username/repo-name)')
    args = parser.parse_args()

    # Load LR Finder config from config.json if it exists
    lr_finder_config = None
    if args.lr_finder:
        config_path = args.scheduler_config or './config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    if 'lr_finder' in config_data:
                        lr_finder_config = config_data['lr_finder']
                        print(f"✓ Loaded LR Finder config from: {config_path}")
            except Exception as e:
                print(f"⚠ Could not load LR Finder config: {e}")
                print("  Using default LR Finder parameters")

    # Create trainer with selected model and run training
    trainer = CIFARTrainer(
        model_module_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        scheduler_type=args.scheduler,
        scheduler_config_path=args.scheduler_config,
        use_lr_finder=args.lr_finder,
        lr_finder_config=lr_finder_config,
        hf_token=args.hf_token,
        hf_repo_id=args.hf_repo
    )
    trainer.run()
