"""
SageMaker Training Script for CIFAR-100 WideResNet

This script is designed to work with AWS SageMaker training jobs.
It can also be run locally for testing.
"""

import argparse
import os
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Import custom modules
from model import create_model
from data import get_dataloaders
from utils import (
    mixup_data, mixup_criterion, WarmupScheduler, MetricsTracker,
    CheckpointManager, set_seed, accuracy, upload_to_huggingface
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='CIFAR-100 Training on SageMaker')

    # Model parameters
    parser.add_argument('--depth', type=int, default=28,
                       help='Model depth (default: 28)')
    parser.add_argument('--widen-factor', type=int, default=10,
                       help='Width factor (default: 10)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--min-lr', type=float, default=1e-4,
                       help='Minimum learning rate (default: 1e-4)')
    parser.add_argument('--initial-lr', type=float, default=0.01,
                       help='Initial learning rate for warmup (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay (default: 1e-3)')

    # Augmentation and regularization
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='MixUp alpha (default: 0.2)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1)')
    parser.add_argument('--use-mixup', type=int, default=1,
                       help='Use MixUp augmentation (default: 1)')

    # Scheduler parameters
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs (default: 5)')
    parser.add_argument('--cosine-t0', type=int, default=25,
                       help='Cosine annealing T_0 (default: 25)')

    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping (default: 1.0)')

    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--target-accuracy', type=float, default=74.0,
                       help='Target accuracy to stop training (default: 74.0)')

    # HuggingFace integration
    parser.add_argument('--hf-repo-id', type=str, default='',
                       help='HuggingFace repository ID for model upload')
    parser.add_argument('--hf-token', type=str, default='',
                       help='HuggingFace API token')

    # SageMaker parameters
    parser.add_argument('--model-dir', type=str,
                       default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--output-dir', type=str,
                       default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--train-dir', type=str,
                       default=os.environ.get('SM_CHANNEL_TRAINING', './data'))

    return parser.parse_args()


def train_epoch(model, device, train_loader, optimizer, scheduler, warmup_scheduler,
               scaler, epoch, args):
    """Train for one epoch"""
    model.train()
    losses = []
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            if args.use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(
                    data, target, alpha=args.mixup_alpha, device=device
                )
                outputs = model(inputs)
                loss = mixup_criterion(
                    F.cross_entropy, outputs, targets_a, targets_b, lam,
                    label_smoothing=args.label_smoothing
                )
            else:
                outputs = model(data)
                loss = F.cross_entropy(outputs, target,
                                     label_smoothing=args.label_smoothing)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        if args.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                          max_norm=args.gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        if warmup_scheduler.is_warmup():
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Calculate accuracy
        _, pred = outputs.max(1)
        if args.use_mixup:
            correct += lam * pred.eq(targets_a).sum().item() + \
                      (1 - lam) * pred.eq(targets_b).sum().item()
        else:
            correct += pred.eq(target).sum().item()
        total += len(data)
        losses.append(loss.item())

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{current_lr:.6f}'
        })

    avg_loss = sum(losses) / len(losses)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test_epoch(model, device, test_loader):
    """Evaluate on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)

    test_loss /= total
    accuracy = 100. * correct / total
    return test_loss, accuracy


def main():
    """Main training function"""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("Training Configuration:")
    print("=" * 70)
    print(f"Model: WideResNet-{args.depth}-{args.widen_factor}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"MixUp Alpha: {args.mixup_alpha}")
    print(f"Label Smoothing: {args.label_smoothing}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"Data Directory: {args.train_dir}")
    print(f"Model Directory: {args.model_dir}")
    print("=" * 70)

    # Create model
    model = create_model(
        depth=args.depth,
        widen_factor=args.widen_factor,
        dropout=args.dropout,
        num_classes=100
    )
    model = model.to(device)
    print(f"Model created with {model.num_parameters():,} parameters")

    # Get data loaders
    train_loader, test_loader, num_classes = get_dataloaders(
        data_dir=args.train_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_albumentations=True
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Setup optimizer and schedulers
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.initial_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.cosine_t0,
        T_mult=1,
        eta_min=args.min_lr
    )

    warmup_scheduler = WarmupScheduler(
        optimizer,
        args.warmup_epochs,
        args.initial_lr,
        args.lr,
        len(train_loader)
    )

    # Gradient scaler for mixed precision
    scaler = GradScaler()

    # Setup tracking
    metrics_tracker = MetricsTracker()
    checkpoint_manager = CheckpointManager(args.model_dir, max_checkpoints=3)

    # Early stopping
    best_test_acc = 0.0
    patience_counter = 0

    # Training loop
    print("\nStarting training...\n")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, scheduler,
            warmup_scheduler, scaler, epoch, args
        )

        # Test
        test_loss, test_acc = test_epoch(model, device, test_loader)

        # Update metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics_tracker.update(epoch, train_loss, train_acc,
                             test_loss, test_acc, current_lr)

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save checkpoint if best model
        is_best = test_acc > best_test_acc
        if is_best:
            best_test_acc = test_acc
            patience_counter = 0
            print(f"  *** New best model! Test Accuracy: {best_test_acc:.2f}% ***")

            checkpoint_manager.save_checkpoint(
                model, optimizer, epoch,
                {'train_acc': train_acc, 'test_acc': test_acc,
                 'train_loss': train_loss, 'test_loss': test_loss},
                is_best=True,
                scheduler=scheduler
            )

            # Upload to HuggingFace if configured
            if args.hf_repo_id and args.hf_token:
                best_path = Path(args.model_dir) / 'best_model.pth'
                upload_to_huggingface(
                    str(best_path), 'best_model.pth',
                    args.hf_repo_id, args.hf_token,
                    f"Epoch {epoch}: Test Acc {test_acc:.2f}%"
                )
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_manager.save_checkpoint(
                model, optimizer, epoch,
                {'train_acc': train_acc, 'test_acc': test_acc,
                 'train_loss': train_loss, 'test_loss': test_loss},
                filename=f'checkpoint_epoch_{epoch}.pth',
                scheduler=scheduler
            )

        # Save metrics
        metrics_path = Path(args.output_dir) / 'metrics.json'
        metrics_tracker.save(str(metrics_path))

        print(f"  Best Test Acc: {best_test_acc:.2f}% | "
              f"Patience: {patience_counter}/{args.patience}\n")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # Check if target reached
        if test_acc >= args.target_accuracy:
            print(f"Target accuracy of {args.target_accuracy}% reached!")
            break

    # Save final model
    print(f"\nTraining completed. Best test accuracy: {best_test_acc:.2f}%")

    # Save final configuration
    config = {
        'model': f'WideResNet-{args.depth}-{args.widen_factor}',
        'depth': args.depth,
        'widen_factor': args.widen_factor,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'epochs': epoch,
        'best_test_accuracy': best_test_acc,
        'parameters': model.num_parameters(),
        'hyperparameters': vars(args)
    }

    config_path = Path(args.model_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {config_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Best model saved to {args.model_dir}/best_model.pth")


if __name__ == '__main__':
    main()
