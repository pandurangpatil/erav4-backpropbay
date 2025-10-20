"""
Model registry and factory for managing different architectures.
"""
from .wideresnet import WideResNet
from .resnet50 import ResNet50

# Model registry
MODELS = {
    'wideresnet28-10': lambda num_classes=100: WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropRate=0.3),
    'wideresnet': lambda num_classes=100: WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropRate=0.3),
    'resnet50': lambda num_classes=100: ResNet50(num_classes=num_classes),
    # Future models can be added here
    # 'resnet18': lambda num_classes=100: ResNet18(num_classes=num_classes),
    # 'efficientnet': lambda num_classes=100: EfficientNet(num_classes=num_classes),
}


def get_model(name, num_classes=100, **kwargs):
    """
    Factory function to get model by name.

    Args:
        name: Model name ('wideresnet28-10', 'resnet50', etc.)
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        torch.nn.Module: Initialized model

    Example:
        >>> model = get_model('wideresnet28-10', num_classes=100)
        >>> model = get_model('resnet50', num_classes=10)
    """
    name = name.lower()
    if name not in MODELS:
        available = ', '.join(MODELS.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    model_fn = MODELS[name]
    return model_fn(num_classes=num_classes, **kwargs)


def list_models():
    """
    List all available models.

    Returns:
        list: List of model names
    """
    return list(MODELS.keys())


__all__ = ['get_model', 'list_models', 'WideResNet', 'ResNet50']
