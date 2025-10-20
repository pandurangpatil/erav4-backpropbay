"""
Learning rate scheduler configuration and factory.
"""
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR


def get_scheduler(name, optimizer, train_loader, epochs=100, config=None):
    """
    Get learning rate scheduler by name.

    Args:
        name: Scheduler name ('onecycle', 'cosine')
        optimizer: PyTorch optimizer
        train_loader: Training data loader
        epochs: Total number of epochs
        config: Scheduler-specific configuration dict

    Returns:
        torch.optim.lr_scheduler: Scheduler instance

    Example:
        >>> scheduler = get_scheduler('onecycle', optimizer, train_loader, epochs=100)
        >>> scheduler = get_scheduler('cosine', optimizer, train_loader)
    """
    name = name.lower()

    if name == 'onecycle':
        return _get_onecycle_scheduler(optimizer, train_loader, epochs, config)
    elif name == 'cosine':
        return _get_cosine_scheduler(optimizer, config)
    else:
        raise ValueError(f"Unknown scheduler: {name}. Supported: onecycle, cosine")


def _get_onecycle_scheduler(optimizer, train_loader, epochs, config=None):
    """
    Get OneCycleLR scheduler with configuration.

    Args:
        optimizer: PyTorch optimizer
        train_loader: Training data loader
        epochs: Total number of epochs
        config: OneCycleLR configuration dict

    Returns:
        OneCycleLR scheduler
    """
    # Default configuration
    default_config = {
        'max_lr': 0.1,
        'pct_start': 0.3,
        'anneal_strategy': 'cos',
        'div_factor': 25.0,
        'final_div_factor': 10000.0,
        'three_phase': False
    }

    # Merge with provided config
    if config:
        default_config.update(config)

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    return OneCycleLR(
        optimizer,
        max_lr=default_config['max_lr'],
        total_steps=total_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=default_config['pct_start'],
        anneal_strategy=default_config['anneal_strategy'],
        div_factor=default_config['div_factor'],
        final_div_factor=default_config['final_div_factor'],
        three_phase=default_config['three_phase']
    )


def _get_cosine_scheduler(optimizer, config=None):
    """
    Get CosineAnnealingWarmRestarts scheduler with configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Cosine annealing configuration dict

    Returns:
        CosineAnnealingWarmRestarts scheduler
    """
    # Default configuration
    default_config = {
        'T_0': 25,
        'T_mult': 1,
        'eta_min': 1e-4
    }

    # Merge with provided config
    if config:
        default_config.update(config)

    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=default_config['T_0'],
        T_mult=default_config['T_mult'],
        eta_min=default_config['eta_min']
    )
