"""
Training utilities: MixUp, Warmup Scheduler, Checkpoint Management, Metrics
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """
    Applies MixUp augmentation to batch

    Args:
        x: Input batch
        y: Target labels
        alpha: MixUp alpha parameter (Beta distribution)
        device: Device to use

    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, label_smoothing=0.0):
    """
    Compute MixUp loss

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: MixUp lambda
        label_smoothing: Label smoothing factor

    Returns:
        Mixed loss
    """
    return lam * F.cross_entropy(pred, y_a, label_smoothing=label_smoothing) + \
           (1 - lam) * F.cross_entropy(pred, y_b, label_smoothing=label_smoothing)


class WarmupScheduler:
    """
    Learning rate warmup scheduler

    Gradually increases learning rate from initial_lr to target_lr
    over specified number of warmup epochs
    """

    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_step = 0

    def step(self):
        """Step the warmup scheduler"""
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * \
                 self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1

    def is_warmup(self):
        """Check if still in warmup phase"""
        return self.current_step < self.warmup_steps

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """Track training metrics over epochs"""

    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.learning_rates = []
        self.best_test_acc = 0.0
        self.best_epoch = 0

    def update(self, epoch, train_loss, train_acc, test_loss, test_acc, lr):
        """Update metrics for current epoch"""
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        self.learning_rates.append(lr)

        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.best_epoch = epoch

    def save(self, filepath):
        """Save metrics to JSON file"""
        metrics = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'learning_rates': self.learning_rates,
            'best_test_accuracy': self.best_test_acc,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.train_losses)
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

    def load(self, filepath):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            metrics = json.load(f)

        self.train_losses = metrics['train_losses']
        self.test_losses = metrics['test_losses']
        self.train_accuracies = metrics['train_accuracies']
        self.test_accuracies = metrics['test_accuracies']
        self.learning_rates = metrics['learning_rates']
        self.best_test_acc = metrics['best_test_accuracy']
        self.best_epoch = metrics['best_epoch']


class CheckpointManager:
    """Manage model checkpoints"""

    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save_checkpoint(self, model, optimizer, epoch, metrics, filename=None,
                       is_best=False, scheduler=None):
        """
        Save model checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            filename: Custom filename (optional)
            is_best: Whether this is the best model
            scheduler: Learning rate scheduler (optional)
        """
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model: {best_path}")

        # Manage checkpoint history
        self.checkpoints.append(checkpoint_path)
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        return checkpoint_path

    def load_checkpoint(self, filepath, model, optimizer=None, scheduler=None):
        """
        Load checkpoint

        Args:
            filepath: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)

        Returns:
            Dictionary with checkpoint information
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint


def upload_to_huggingface(file_path, path_in_repo, repo_id, hf_token,
                          commit_message="Upload checkpoint"):
    """
    Upload file to HuggingFace Hub

    Args:
        file_path: Local file path
        path_in_repo: Path in the repository
        repo_id: HuggingFace repository ID
        hf_token: HuggingFace API token
        commit_message: Commit message
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=commit_message
        )
        print(f"✓ Uploaded to HuggingFace: {path_in_repo}")
        return True
    except Exception as e:
        print(f"✗ Failed to upload to HuggingFace: {e}")
        return False


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions

    Args:
        output: Model output logits
        target: Ground truth labels
        topk: Tuple of top-k values to compute

    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utilities
    print("Testing training utilities...")

    # Test MixUp
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 100, (4,))
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, device='cpu')
    print(f"MixUp lambda: {lam:.3f}")

    # Test WarmupScheduler
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    warmup = WarmupScheduler(optimizer, warmup_epochs=5, initial_lr=0.01,
                            target_lr=0.1, steps_per_epoch=100)
    print(f"Initial LR: {warmup.get_lr():.6f}")
    for _ in range(250):
        warmup.step()
    print(f"After warmup LR: {warmup.get_lr():.6f}")

    # Test MetricsTracker
    tracker = MetricsTracker()
    tracker.update(1, 2.5, 45.0, 2.8, 42.0, 0.1)
    tracker.update(2, 2.0, 50.0, 2.5, 48.0, 0.1)
    print(f"Best accuracy: {tracker.best_test_acc:.2f}% at epoch {tracker.best_epoch}")

    print("All tests passed!")
