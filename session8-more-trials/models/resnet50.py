"""
ResNet50 architecture for CIFAR-100.

Adapted from the original ResNet paper (https://arxiv.org/abs/1512.03385)
Modified for CIFAR (32x32 images) by using smaller initial conv and no max pooling.
"""
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet50.

    Architecture: 1x1 conv -> 3x3 conv -> 1x1 conv with residual connection
    Expansion factor is 4 (output channels = 4 * base channels)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv for main processing
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv to expand dimensions
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 conv
        out = self.conv3(out)
        out = self.bn3(out)

        # Downsample residual if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual connection
        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """
    ResNet50 implementation for CIFAR-100.

    Architecture adapted for small images (32x32):
    - Initial conv layer (3x3, stride 1) instead of (7x7, stride 2)
    - No max pooling layer
    - Layer1: 3 Bottleneck blocks, 64 channels
    - Layer2: 4 Bottleneck blocks, 128 channels, stride 2
    - Layer3: 6 Bottleneck blocks, 256 channels, stride 2
    - Layer4: 3 Bottleneck blocks, 512 channels, stride 2
    - Adaptive avg pool + FC layer

    Total: 3 + 4 + 6 + 3 = 16 blocks * 3 layers + 2 = 50 layers

    Args:
        num_classes: Number of output classes (default: 100)
    """

    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()

        self.in_channels = 64

        # Initial convolution - adapted for CIFAR (32x32 images)
        # Using 3x3 kernel with stride 1 instead of 7x7 with stride 2
        # No max pooling (would be too aggressive for 32x32 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet50 layers: [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Create a ResNet layer with multiple blocks.

        Args:
            block: Bottleneck block class
            out_channels: Number of output channels for the blocks
            num_blocks: Number of blocks in this layer
            stride: Stride for the first block (subsequent blocks use stride=1)
        """
        downsample = None

        # If stride != 1 or dimensions change, we need downsampling for residual
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block (may have stride > 1 and downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * block.expansion

        # Remaining blocks (stride = 1, no downsample needed)
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using He initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier
        x = self.fc(x)

        return x
