import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR

# ------------------------------
# Data Augmentation
# ------------------------------
train_transforms = transforms.Compose([
    transforms.RandomRotation((-7.0, 7.0), fill=(1,)),   # small rotations
    transforms.RandomAffine(0, translate=(0.1, 0.1)),    # slight translations
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ------------------------------
# Model Definition
# ------------------------------
dropout_value = 0.05

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # 26x26x8

        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) # 24x24x16

        self.convblock3 = nn.Conv2d(16, 8, 1, bias=False)  # 24x24x8
        self.pool1 = nn.MaxPool2d(2, 2)                    # 12x12x8

        self.convblock4 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=0, bias=False),    # 10x10x16
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0, bias=False),   # 8x8x16
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),   # 8x8x16
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # → 1x1x16
        self.fc = nn.Conv2d(16, 10, 1, bias=False)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

# ------------------------------
# Optimizer & Scheduler Configuration
# ------------------------------
def get_optimizer(model):
    """Get the optimizer for the model."""
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

def get_scheduler(optimizer, train_loader):
    """Get the learning rate scheduler."""
    return OneCycleLR(
        optimizer,
        max_lr=0.05,
        steps_per_epoch=len(train_loader),
        epochs=20
    )

