#!/usr/bin/env python3
"""
Local testing script for CIFAR-100 training
Run this before deploying to SageMaker to verify everything works
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(description='Local test for CIFAR-100 training')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of epochs for local test (default: 2)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for local test (default: 32)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("=" * 70)
    print("LOCAL TESTING MODE")
    print("=" * 70)
    print("Running quick training test with:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data dir: {args.data_dir}")
    print("=" * 70)
    print()

    # Override sys.argv for train.py
    sys.argv = [
        'train.py',
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--train-dir', args.data_dir,
        '--model-dir', './model',
        '--output-dir', './output',
        '--num-workers', '2',
        '--patience', '999',  # Disable early stopping for testing
    ]

    try:
        train_main()
        print()
        print("=" * 70)
        print("✓ LOCAL TEST PASSED")
        print("=" * 70)
        print("Your code is ready for SageMaker deployment!")
        print()
        print("Next steps:")
        print("  1. Review config.yaml for hyperparameters")
        print("  2. Run: ./deploy.sh --spot --monitor")
        print()
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ LOCAL TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("Please fix the errors before deploying to SageMaker")
        sys.exit(1)
