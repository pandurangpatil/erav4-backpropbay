"""
Main Trainer class for model training with advanced features.
"""
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
from torchvision import datasets

from utils import CheckpointManager, MetricsTracker, plot_training_curves, get_device
from utils import HuggingFaceUploader, create_model_card, plot_lr_finder
from .lr_finder import LRFinder


class Trainer:
    """
    Main trainer class handling the complete training pipeline.
    """

    def __init__(self, model, train_loader, test_loader, optimizer, scheduler,
                 device=None, checkpoint_manager=None, metrics_tracker=None,
                 scheduler_type='onecycle', use_mixup=True, mixup_alpha=0.2,
                 label_smoothing=0.1, use_amp=True, gradient_clip=1.0,
                 hf_uploader=None, model_name='model', config=None):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            test_loader: Test DataLoader
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            device: Device to train on (auto-detect if None)
            checkpoint_manager: CheckpointManager instance
            metrics_tracker: MetricsTracker instance
            scheduler_type: Type of scheduler ('onecycle' or 'cosine')
            use_mixup: Enable MixUp augmentation
            mixup_alpha: MixUp alpha parameter
            label_smoothing: Label smoothing parameter
            use_amp: Use automatic mixed precision
            gradient_clip: Gradient clipping max norm
            hf_uploader: HuggingFaceUploader instance (optional)
            model_name: Model name for logging
            config: Training configuration dict
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.model.to(self.device)

        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.metrics_tracker = metrics_tracker or MetricsTracker()

        self.scheduler_type = scheduler_type.lower()
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.hf_uploader = hf_uploader
        self.model_name = model_name
        self.config = config or {}

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

    def mixup_data(self, x, y, alpha=0.2):
        """Apply MixUp data augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (avg_loss, accuracy)
        """
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
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
                        inputs, targets_a, targets_b, lam = self.mixup_data(
                            data, target, alpha=self.mixup_alpha
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
                    inputs, targets_a, targets_b, lam = self.mixup_data(
                        data, target, alpha=self.mixup_alpha
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

            # Update learning rate for OneCycleLR (per batch)
            if self.scheduler_type == 'onecycle':
                self.scheduler.step()

            # Accuracy tracking
            _, pred = outputs.max(1)
            if self.use_mixup:
                correct += lam * pred.eq(targets_a).sum().item() + (1 - lam) * pred.eq(targets_b).sum().item()
            else:
                correct += pred.eq(target).sum().item()
            processed += len(data)
            epoch_loss += loss.item()

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/processed:.2f}%',
                'lr': f'{current_lr:.6f}'
            })

        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100. * correct / processed
        return avg_loss, accuracy

    def test(self):
        """
        Test the model.

        Returns:
            tuple: (test_loss, accuracy)
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

    def run(self, epochs, patience=15, checkpoint_epochs=None, target_accuracy=None):
        """
        Run the complete training loop.

        Args:
            epochs: Number of epochs to train
            patience: Early stopping patience
            checkpoint_epochs: List of epochs to save checkpoints
            target_accuracy: Target accuracy to stop training

        Returns:
            float: Best test accuracy achieved
        """
        checkpoint_epochs = checkpoint_epochs or [10, 20, 25, 30, 40, 50, 60, 75, 90]
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            test_loss, test_acc = self.test()

            # Step scheduler after epoch for CosineAnnealing (OneCycle steps per batch)
            if self.scheduler_type == 'cosine':
                self.scheduler.step()

            # Check if best model BEFORE updating tracker
            current_lr = self.optimizer.param_groups[0]['lr']
            is_best = test_acc > self.metrics_tracker.best_test_acc

            # Track metrics (this will update best_test_acc)
            self.metrics_tracker.update(train_loss, train_acc, test_loss, test_acc, current_lr)

            # Save best model
            if is_best:
                patience_counter = 0
                print(f"*** New best model! Test Accuracy: {test_acc:.2f}% ***")

                metrics = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, metrics, self.config, is_best=True
                )
            else:
                patience_counter += 1

            # Save checkpoint at breakpoints
            if epoch in checkpoint_epochs:
                print(f"📍 Breakpoint checkpoint at epoch {epoch}")
                metrics = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, metrics, self.config
                )

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}. No improvement for {patience} epochs.")
                break

            # Check if target reached
            if target_accuracy and test_acc >= target_accuracy:
                print(f"\n{'=' * 70}")
                print(f"Target accuracy of {target_accuracy}% reached at epoch {epoch}!")
                print(f"Final test accuracy: {test_acc:.2f}%")
                print(f"{'=' * 70}")
                break

            print(f"Best so far: {self.metrics_tracker.best_test_acc:.2f}% | Patience: {patience_counter}/{patience}\n")

        print(f"\nTraining completed. Best test accuracy: {self.metrics_tracker.best_test_acc:.2f}% "
              f"(epoch {self.metrics_tracker.best_epoch})")

        return self.metrics_tracker.best_test_acc

    def save_results(self):
        """Save all training results (metrics, plots, model card)."""
        checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir()

        # Save metrics
        metrics_path = f"{checkpoint_dir}/metrics.json"
        self.metrics_tracker.save(metrics_path)

        # Plot and save training curves
        self.metrics_tracker.print_console_plots()
        curves_path = f"{checkpoint_dir}/training_curves.png"
        plot_training_curves(self.metrics_tracker, curves_path)

        # Save config
        import json
        config_path = f"{checkpoint_dir}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✓ Saved config: {config_path}")

        # Create model card
        if self.hf_uploader:
            model_card = create_model_card(
                self.hf_uploader.repo_id,
                self.model_name,
                self.metrics_tracker.best_test_acc,
                len(self.metrics_tracker.train_losses),
                self.metrics_tracker,
                self.config
            )
            readme_path = f"{checkpoint_dir}/README.md"
            with open(readme_path, 'w') as f:
                f.write(model_card)
            print(f"✓ Saved model card: {readme_path}")

    def upload_to_huggingface(self):
        """Upload results to HuggingFace Hub if configured."""
        if not self.hf_uploader:
            print("\n⚠ HuggingFace upload skipped (no uploader configured)")
            return

        checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir()
        self.hf_uploader.upload_checkpoint_files(checkpoint_dir)

    def run_lr_finder(self, config):
        """
        Run LR Finder to find optimal learning rates.

        Args:
            config: LR Finder configuration dict

        Returns:
            dict: Suggested LR values
        """
        print("\n" + "="*70)
        print("STARTING LR FINDER")
        print("="*70)

        # Create clean data loader (no mixup, no label smoothing)
        from data_loaders.transforms import TestTransformWrapper
        from data_loaders import get_dataset_info

        dataset_info = get_dataset_info('cifar100')
        test_transform = TestTransformWrapper(dataset_info['mean'], dataset_info['std'])

        clean_dataset = datasets.CIFAR100(
            '../data',
            train=True,
            download=False,
            transform=test_transform
        )

        # Only use pin_memory on CUDA devices
        use_pin_memory = self.device.type == 'cuda'

        clean_loader = torch.utils.data.DataLoader(
            clean_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        # Create LR Finder
        criterion = torch.nn.CrossEntropyLoss()
        lr_finder = LRFinder(
            model=self.model,
            optimizer=self.optimizer,
            criterion=criterion,
            device=self.device,
            data_loader=clean_loader
        )

        # Run range test
        num_epochs = config.get('num_epochs', 3)
        start_lr = config.get('start_lr', 1e-6)
        end_lr = config.get('end_lr', 1.0)
        selection_method = config.get('selection_method', 'steepest_gradient')

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

        # Save plot
        checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir()
        plot_path = f"{checkpoint_dir}/lr_finder_plot.png"

        if self.scheduler_type == 'onecycle':
            plot_lr_finder(
                lr_finder.lrs,
                lr_finder.losses,
                plot_path,
                max_lr=suggested_lrs.get('max_lr'),
                base_lr=suggested_lrs.get('base_lr')
            )
        else:
            plot_lr_finder(
                lr_finder.lrs,
                lr_finder.losses,
                plot_path,
                suggested_lr=suggested_lrs.get('initial_lr')
            )

        print(f"\n{'='*70}")
        print(f"LR FINDER COMPLETED")
        print(f"{'='*70}\n")

        return suggested_lrs
