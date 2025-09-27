import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm
import importlib
from torchsummary import summary


class MNISTTrainer:
    def __init__(self, model_module_name='model3', epochs=20, batch_size=128):
        """
        Initialize the MNIST trainer with a model module.

        Args:
            model_module_name: Name of the module containing the model (e.g., 'model3')
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

    def _setup_data(self):
        """Setup MNIST datasets and data loaders."""
        self.train_dataset = datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=self.train_transforms
        )
        self.test_dataset = datasets.MNIST(
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
        """
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(data)
            loss = F.nll_loss(y_pred, target)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(
                f"Epoch {epoch} Loss={loss.item():.4f} "
                f"Acc={100*correct/processed:.2f}"
            )

    def print_model_summary(self):
        """Print the model architecture summary."""
        print("\n" + "="*50)
        print("Model Architecture Summary")
        print("="*50)
        print(f"Device: {self.device}")
        print("\nModel Summary:")
        summary(self.model, input_size=(1, 28, 28))
        print("="*50 + "\n")

    def test(self):
        """
        Test the model and return accuracy.

        Returns:
            float: Test accuracy percentage
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
        return acc

    def run(self):
        """Run the complete training process for all epochs."""
        print(f"Training {self.model_module.__name__} for {self.epochs} epochs")

        # Print model summary before training
        self.print_model_summary()

        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            acc = self.test()

        print(f"Training completed. Final accuracy: {acc:.2f}%")
        return acc


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MNIST model')
    parser.add_argument('--model', type=str, default='model3',
                       choices=['model1', 'model2', 'model3'],
                       help='Model to use for training (default: model3)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    args = parser.parse_args()

    # Create trainer with selected model and run training
    trainer = MNISTTrainer(model_module_name=args.model,
                          epochs=args.epochs,
                          batch_size=args.batch_size)
    trainer.run()