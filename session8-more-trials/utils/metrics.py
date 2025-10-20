"""
Metrics tracking and visualization utilities.
"""
import json
import os
import matplotlib.pyplot as plt
import plotille
import numpy as np


class MetricsTracker:
    """Track and store training metrics across epochs."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.learning_rates = []
        self.best_test_acc = 0.0
        self.best_epoch = 0

    def update(self, train_loss, train_acc, test_loss, test_acc, lr):
        """
        Update metrics for current epoch.

        Args:
            train_loss: Training loss
            train_acc: Training accuracy
            test_loss: Test loss
            test_acc: Test accuracy
            lr: Current learning rate
        """
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        self.learning_rates.append(lr)

        # Update best metrics
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.best_epoch = len(self.test_accuracies)

    def save(self, filepath):
        """
        Save metrics to JSON file.

        Args:
            filepath: Path to save metrics JSON
        """
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

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Saved metrics: {filepath}")

    def load(self, filepath):
        """
        Load metrics from JSON file.

        Args:
            filepath: Path to metrics JSON file
        """
        with open(filepath, 'r') as f:
            metrics = json.load(f)

        self.train_losses = metrics['train_losses']
        self.train_accuracies = metrics['train_accuracies']
        self.test_losses = metrics['test_losses']
        self.test_accuracies = metrics['test_accuracies']
        self.learning_rates = metrics['learning_rates']
        self.best_test_acc = metrics['best_test_accuracy']
        self.best_epoch = metrics['best_epoch']

        print(f"✓ Loaded metrics: {filepath}")

    def print_console_plots(self):
        """Print ASCII plots to console using plotille."""
        epochs = list(range(1, len(self.train_losses) + 1))

        # Loss plot
        print("\n" + "="*80)
        print("TRAINING AND TESTING LOSS")
        print("="*80)

        fig = plotille.Figure()
        fig.width = 70
        fig.height = 20
        fig.color_mode = 'byte'
        fig.set_x_limits(min_=1, max_=len(epochs))
        fig.set_y_limits(min_=0, max_=max(max(self.train_losses), max(self.test_losses)) * 1.1)

        fig.plot(epochs, self.train_losses, lc=25, label='Training Loss')
        fig.plot(epochs, self.test_losses, lc=196, label='Testing Loss')

        print(fig.show(legend=True))

        # Accuracy plot
        print("\n" + "="*80)
        print("TRAINING AND TESTING ACCURACY")
        print("="*80)

        fig = plotille.Figure()
        fig.width = 70
        fig.height = 20
        fig.color_mode = 'byte'
        fig.set_x_limits(min_=1, max_=len(epochs))
        fig.set_y_limits(min_=min(min(self.train_accuracies), min(self.test_accuracies)) * 0.95,
                         max_=100)

        fig.plot(epochs, self.train_accuracies, lc=25, label='Training Accuracy')
        fig.plot(epochs, self.test_accuracies, lc=196, label='Testing Accuracy')

        print(fig.show(legend=True))
        print("="*80 + "\n")


def plot_training_curves(metrics_tracker, save_path, target_accuracy=74.0):
    """
    Plot and save training curves.

    Args:
        metrics_tracker: MetricsTracker instance
        save_path: Path to save the plot image
        target_accuracy: Optional target accuracy to mark on plot
    """
    epochs = list(range(1, len(metrics_tracker.train_losses) + 1))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Loss
    axs[0, 0].plot(epochs, metrics_tracker.train_losses, label='Train Loss')
    axs[0, 0].plot(epochs, metrics_tracker.test_losses, label='Test Loss')
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: Accuracy
    axs[0, 1].plot(epochs, metrics_tracker.train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(epochs, metrics_tracker.test_accuracies, label='Test Accuracy')
    if target_accuracy:
        axs[0, 1].axhline(y=target_accuracy, color='r', linestyle='--',
                          label=f'Target ({target_accuracy}%)')
    axs[0, 1].set_title("Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy (%)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Learning Rate
    axs[1, 0].plot(epochs, metrics_tracker.learning_rates, color='green')
    axs[1, 0].set_title("Learning Rate Schedule")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Learning Rate")
    axs[1, 0].grid(True)

    # Plot 4: Train-Test Gap
    accuracy_gap = [train - test for train, test in
                    zip(metrics_tracker.train_accuracies, metrics_tracker.test_accuracies)]
    axs[1, 1].plot(epochs, accuracy_gap, color='orange')
    axs[1, 1].set_title("Train-Test Accuracy Gap")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Gap (%)")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved: {save_path}")
    plt.close()


def plot_lr_finder(lrs, losses, save_path, suggested_lr=None, max_lr=None, base_lr=None):
    """
    Plot LR Finder results.

    Args:
        lrs: List of learning rates tested
        losses: List of corresponding losses
        save_path: Path to save the plot
        suggested_lr: Single LR to mark (for CosineAnnealing)
        max_lr: Max LR to mark (for OneCycle)
        base_lr: Base LR to mark (for OneCycle)
    """
    if len(lrs) == 0:
        print("⚠ No data to plot")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, linewidth=2, label='Loss')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Learning Rate Finder - Range Test', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Mark suggested LR points
    if suggested_lr is not None:
        plt.axvline(x=suggested_lr, color='red', linestyle='--', linewidth=2,
                   label=f'Suggested LR: {suggested_lr:.6f}')

    if max_lr is not None:
        plt.axvline(x=max_lr, color='green', linestyle='--', linewidth=2,
                   label=f'Max LR: {max_lr:.6f}')

    if base_lr is not None:
        plt.axvline(x=base_lr, color='blue', linestyle='--', linewidth=2,
                   label=f'Base LR: {base_lr:.6f}')

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ LR Finder plot saved: {save_path}")
    plt.close()
