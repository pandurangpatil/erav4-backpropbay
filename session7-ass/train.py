import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm
import importlib
from torchsummary import summary
import matplotlib.pyplot as plt
import plotille


class CIFARTrainer:
    def __init__(self, model_module_name='model', epochs=20, batch_size=128):
        """
        Initialize the CIFAR trainer with a model module.

        Args:
            model_module_name: Name of the module containing the model (e.g., 'model')
            epochs: Number of training epochs
            batch_size: Batch size for training and testing
        """
        self.model_module = importlib.import_module(model_module_name)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = self.model_module.Net().to(self.device)

        # Get transforms from model module
        self.train_transforms = self.model_module.train_transforms
        self.test_transforms = self.model_module.test_transforms

        # Setup datasets and data loaders
        self._setup_data()

        # Get optimizer and scheduler from model module
        self.optimizer = self.model_module.get_optimizer(self.model)
        self.scheduler = self.model_module.get_scheduler(self.optimizer, self.train_loader)

        # Initialize metric tracking
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def _setup_data(self):
        """Setup CIFAR-10 datasets and data loaders."""
        self.train_dataset = datasets.CIFAR10(
            '../data',
            train=True,
            download=True,
            transform=self.train_transforms
        )
        self.test_dataset = datasets.CIFAR10(
            '../data',
            train=False,
            transform=self.test_transforms
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def train(self, epoch):
        """
        Train the model for one epoch.

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
            y_pred = self.model(data)
            loss = F.nll_loss(y_pred, target)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss += loss.item()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(
                f"Epoch {epoch} Loss={loss.item():.4f} "
                f"Acc={100*correct/processed:.2f}"
            )

        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100. * correct / processed
        return avg_loss, accuracy

    def print_model_summary(self):
        """Print the model architecture summary."""
        print("\n" + "="*50)
        print("Model Architecture Summary")
        print("="*50)
        print(f"Device: {self.device}")
        print("\nModel Summary:")
        summary(self.model, input_size=(3, 32, 32))
        print("="*50 + "\n")

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
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        acc = 100. * correct / len(self.test_loader.dataset)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(self.test_loader.dataset)} ({acc:.2f}%)\n"
        )
        return test_loss, acc

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

        # Also save matplotlib plots for reference
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")

        axs[1, 0].plot(self.train_accuracies)
        axs[1, 0].set_title("Training Accuracy")

        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")

        axs[1, 1].plot(self.test_accuracies)
        axs[1, 1].set_title("Test Accuracy")

        plt.tight_layout()

        # Save the plot
        plot_filename = f'training_metrics_{self.model_module.__name__}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Metrics plot also saved as '{plot_filename}'\n")

        plt.close()

    def run(self):
        """Run the complete training process for all epochs."""
        print(f"Training {self.model_module.__name__} for {self.epochs} epochs")

        # Print model summary before training
        self.print_model_summary()

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train(epoch)
            test_loss, test_acc = self.test()

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

        print(f"Training completed. Final test accuracy: {test_acc:.2f}%")

        # Plot metrics
        self.plot_metrics()

        return test_acc


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CIFAR model')
    parser.add_argument('--model', type=str, default='model',
                       help='Model module name to use for training (default: model)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    args = parser.parse_args()

    # Create trainer with selected model and run training
    trainer = CIFARTrainer(model_module_name=args.model,
                          epochs=args.epochs,
                          batch_size=args.batch_size)
    trainer.run()