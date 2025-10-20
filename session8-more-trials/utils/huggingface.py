"""
HuggingFace Hub integration for model upload and model card creation.
"""
import os

# HuggingFace Hub imports (optional)
try:
    from huggingface_hub import HfApi, create_repo, upload_file
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HuggingFaceUploader:
    """Manages uploads to HuggingFace Hub."""

    def __init__(self, repo_id, token):
        """
        Initialize HuggingFace uploader.

        Args:
            repo_id: HuggingFace repository ID (e.g., 'username/model-name')
            token: HuggingFace API token
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )

        self.repo_id = repo_id
        self.token = token
        self.api = HfApi()

        # Create repository if it doesn't exist
        try:
            create_repo(repo_id=self.repo_id, repo_type="model", exist_ok=True, token=self.token)
            print(f"✓ Repository ready: https://huggingface.co/{self.repo_id}")
        except Exception as e:
            print(f"Warning: Could not create repository: {e}")

    def upload_file(self, file_path, path_in_repo, commit_message="Upload file"):
        """
        Upload a file to HuggingFace Hub.

        Args:
            file_path: Local path to file
            path_in_repo: Path in the repository
            commit_message: Commit message for upload
        """
        if not os.path.exists(file_path):
            print(f"⚠ File not found, skipping: {file_path}")
            return

        try:
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token,
                commit_message=commit_message
            )
            print(f"✓ Uploaded: {path_in_repo}")
        except Exception as e:
            print(f"✗ Upload failed for {path_in_repo}: {e}")

    def upload_checkpoint_files(self, checkpoint_dir):
        """
        Upload all important files from checkpoint directory.

        Args:
            checkpoint_dir: Path to checkpoint directory
        """
        print("\n" + "="*70)
        print("Uploading to HuggingFace Hub...")
        print("="*70)

        files_to_upload = {
            'best_model.pth': ('best_model.pth', 'Upload best model checkpoint'),
            'training_curves.png': ('training_curves.png', 'Upload training curves'),
            'lr_finder_plot.png': ('lr_finder_plot.png', 'Upload LR finder plot'),
            'metrics.json': ('metrics.json', 'Upload training metrics'),
            'config.json': ('config.json', 'Upload configuration'),
            'README.md': ('README.md', 'Update model card'),
        }

        for filename, (repo_path, commit_msg) in files_to_upload.items():
            file_path = os.path.join(checkpoint_dir, filename)
            self.upload_file(file_path, repo_path, commit_msg)

        print(f"✓ All files uploaded to: https://huggingface.co/{self.repo_id}")
        print("="*70 + "\n")


def create_model_card(repo_id, model_name, best_acc, total_epochs, metrics_tracker, config):
    """
    Create a comprehensive model card (README.md) for HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        model_name: Model architecture name
        best_acc: Best test accuracy achieved
        total_epochs: Total epochs trained
        metrics_tracker: MetricsTracker instance with training history
        config: Training configuration dict

    Returns:
        str: Model card content in markdown format
    """
    train_accuracies = metrics_tracker.train_accuracies
    test_accuracies = metrics_tracker.test_accuracies
    train_losses = metrics_tracker.train_losses
    test_losses = metrics_tracker.test_losses

    model_card = f"""---
tags:
- image-classification
- cifar100
- {model_name.lower()}
- pytorch
datasets:
- cifar100
metrics:
- accuracy
---

# CIFAR-100 {model_name}

## Model Description

{model_name} trained on CIFAR-100 dataset with advanced augmentation techniques.

### Model Architecture
- **Architecture**: {model_name}
- **Dataset**: CIFAR-100
- **Classes**: 100

### Training Configuration
- **Batch Size**: {config.get('batch_size', 256)}
- **Optimizer**: {config.get('optimizer', 'SGD')} (momentum=0.9, weight_decay=1e-3)
- **Scheduler**: {config.get('scheduler', 'OneCycleLR')}
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
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Load model (you'll need to have the model definition)
# from models import get_model
# model = get_model('{model_name.lower()}', num_classes=100)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
```

### Training Details
- **Dataset**: CIFAR-100 (50,000 train, 10,000 test)
- **Classes**: 100
- **Image Size**: 32×32
- **Normalization**: mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761)

### Files
- `best_model.pth` - Best performing model checkpoint
- `training_curves.png` - Training/test accuracy and loss curves
- `lr_finder_plot.png` - Learning rate finder results
- `metrics.json` - Complete training history
- `config.json` - Hyperparameter configuration

### License
MIT

### Citation
```bibtex
@misc{{{model_name.lower()}-cifar100,
  title = {{CIFAR-100 {model_name}}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""
    return model_card
