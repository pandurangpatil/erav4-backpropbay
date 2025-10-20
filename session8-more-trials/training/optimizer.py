"""
Optimizer configuration and factory.
"""
import torch.optim as optim


def get_optimizer(name, model, lr=0.01, momentum=0.9, weight_decay=1e-3, **kwargs):
    """
    Get optimizer by name.

    Args:
        name: Optimizer name ('sgd', 'adam', 'adamw')
        model: PyTorch model
        lr: Learning rate
        momentum: Momentum (for SGD)
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific parameters

    Returns:
        torch.optim.Optimizer: Optimizer instance

    Example:
        >>> optimizer = get_optimizer('sgd', model, lr=0.01)
        >>> optimizer = get_optimizer('adam', model, lr=0.001)
    """
    name = name.lower()

    if name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    elif name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif name == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}. Supported: sgd, adam, adamw")
