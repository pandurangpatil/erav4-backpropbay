"""
Main training script - simplified entry point using modular components.

Example usage:
    python train.py --model wideresnet28-10 --dataset cifar100 --epochs 50 --lr-finder
    python train.py --model resnet50 --scheduler onecycle --batch-size 256
"""
import argparse
import json
import os
import torch.utils.data

from data_loaders import get_dataset, get_dataset_info
from models import get_model
from training import get_optimizer, get_scheduler, Trainer
from utils import get_device, CheckpointManager, MetricsTracker, HuggingFaceUploader


def load_config_file(config_path='./config.json'):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"⚠ Config file not found: {config_path}")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded config from: {config_path}")
        return config
    except Exception as e:
        print(f"⚠ Error loading config: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Train image classification model')

    # Model and dataset
    parser.add_argument('--model', type=str, default='wideresnet28-10',
                       help='Model architecture (default: wideresnet28-10)')
    parser.add_argument('--dataset', type=str, default='cifar100',
                       help='Dataset name (default: cifar100)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda, cuda:0, mps, cpu (default: auto-detect)')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='sgd',
                       help='Optimizer: sgd, adam, adamw (default: sgd)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'onecycle'],
                       help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--config', type=str, default='./config.json',
                       help='Path to config file (default: ./config.json)')

    # Augmentation and regularization
    parser.add_argument('--augmentation', type=str, default='strong',
                       choices=['none', 'weak', 'strong'],
                       help='Augmentation strength (default: strong)')
    parser.add_argument('--no-mixup', action='store_true',
                       help='Disable MixUp augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='MixUp alpha parameter (default: 0.2)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1)')

    # Training features
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping max norm (default: 1.0)')

    # LR Finder
    parser.add_argument('--lr-finder', action='store_true',
                       help='Run LR Finder before training')

    # HuggingFace
    parser.add_argument('--hf-token', type=str, default=None,
                       help='HuggingFace API token')
    parser.add_argument('--hf-repo', type=str, default=None,
                       help='HuggingFace repository ID (e.g., username/repo-name)')

    args = parser.parse_args()

    # Load config file
    config_data = load_config_file(args.config)

    # Print configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Augmentation: {args.augmentation}")
    print(f"MixUp: {not args.no_mixup} (alpha={args.mixup_alpha})")
    print(f"Label Smoothing: {args.label_smoothing}")
    print(f"Mixed Precision: {not args.no_amp}")
    print(f"Gradient Clipping: {args.gradient_clip}")
    print(f"LR Finder: {args.lr_finder}")
    if args.hf_repo:
        print(f"HuggingFace Repo: {args.hf_repo}")
    print("="*70 + "\n")

    # Get device
    device = get_device(args.device)

    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    print(f"\n📊 Dataset: {dataset_info['name']}")
    print(f"   Classes: {dataset_info['num_classes']}")
    print(f"   Train samples: {dataset_info['train_samples']}")
    print(f"   Test samples: {dataset_info['test_samples']}\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = get_dataset(args.dataset, train=True, augmentation=args.augmentation)
    test_dataset = get_dataset(args.dataset, train=False, augmentation='none')

    # Create data loaders
    use_pin_memory = device.type == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}\n")

    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_classes=dataset_info['num_classes'])
    print(f"✓ Model created\n")

    # Create optimizer
    optimizer = get_optimizer(args.optimizer, model)
    print(f"✓ Optimizer: {args.optimizer}\n")

    # Get scheduler config from file
    scheduler_config = None
    if args.scheduler == 'onecycle' and 'scheduler' in config_data:
        scheduler_config = config_data['scheduler'].get('onecycle')
    elif args.scheduler == 'cosine' and 'scheduler' in config_data:
        scheduler_config = config_data['scheduler'].get('cosine')

    # Create scheduler
    scheduler = get_scheduler(
        args.scheduler,
        optimizer,
        train_loader,
        epochs=args.epochs,
        config=scheduler_config
    )
    print(f"✓ Scheduler: {args.scheduler}\n")

    # Create checkpoint manager and metrics tracker
    checkpoint_manager = CheckpointManager()
    metrics_tracker = MetricsTracker()

    # Setup HuggingFace uploader if credentials provided
    hf_uploader = None
    if args.hf_token and args.hf_repo:
        try:
            hf_uploader = HuggingFaceUploader(args.hf_repo, args.hf_token)
        except Exception as e:
            print(f"⚠ Could not initialize HuggingFace uploader: {e}")

    # Training configuration dict
    training_config = {
        'model': args.model,
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'augmentation': args.augmentation,
        'mixup_alpha': args.mixup_alpha if not args.no_mixup else 0.0,
        'label_smoothing': args.label_smoothing,
        'mixed_precision': not args.no_amp,
        'gradient_clipping': args.gradient_clip
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_manager=checkpoint_manager,
        metrics_tracker=metrics_tracker,
        scheduler_type=args.scheduler,
        use_mixup=not args.no_mixup,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        use_amp=not args.no_amp,
        gradient_clip=args.gradient_clip,
        hf_uploader=hf_uploader,
        model_name=args.model,
        config=training_config
    )

    # Run LR Finder if requested
    if args.lr_finder:
        lr_finder_config = config_data.get('lr_finder', {
            'num_epochs': 3,
            'start_lr': 1e-6,
            'end_lr': 1.0,
            'selection_method': 'steepest_gradient'
        })

        suggested_lrs = trainer.run_lr_finder(lr_finder_config)

        # Update scheduler with found LRs
        if suggested_lrs and args.scheduler == 'onecycle':
            max_lr = suggested_lrs.get('max_lr')
            base_lr = suggested_lrs.get('base_lr')

            if max_lr and base_lr:
                print(f"\n{'='*70}")
                print(f"UPDATING SCHEDULER WITH LR FINDER RESULTS")
                print(f"{'='*70}")
                print(f"  Max LR: {max_lr:.6f}")
                print(f"  Base LR (via div_factor): {base_lr:.6f}")
                print(f"{'='*70}\n")

                # Update scheduler config
                div_factor = max_lr / base_lr
                updated_config = scheduler_config.copy() if scheduler_config else {}
                updated_config['max_lr'] = max_lr
                updated_config['div_factor'] = div_factor

                # Recreate scheduler
                trainer.scheduler = get_scheduler(
                    args.scheduler,
                    optimizer,
                    train_loader,
                    epochs=args.epochs,
                    config=updated_config
                )

                # Update training config
                training_config['lr_finder'] = {
                    'max_lr': max_lr,
                    'base_lr': base_lr,
                    'div_factor': div_factor
                }
                trainer.config = training_config

        elif suggested_lrs and args.scheduler == 'cosine':
            initial_lr = suggested_lrs.get('initial_lr')
            if initial_lr:
                print(f"\n{'='*70}")
                print(f"UPDATING SCHEDULER WITH LR FINDER RESULTS")
                print(f"{'='*70}")
                print(f"  Initial LR: {initial_lr:.6f}")
                print(f"{'='*70}\n")

                # Update optimizer LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr

                training_config['lr_finder'] = {'initial_lr': initial_lr}
                trainer.config = training_config

    # Run training
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    best_accuracy = trainer.run(
        epochs=args.epochs,
        patience=15,
        target_accuracy=74.0 if args.dataset == 'cifar100' else None
    )

    # Save results
    trainer.save_results()

    # Upload to HuggingFace if configured
    if hf_uploader:
        trainer.upload_to_huggingface()

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoint saved to: {checkpoint_manager.get_checkpoint_dir()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
