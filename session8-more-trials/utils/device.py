"""
Device detection and configuration for GPU/MPS/CPU.
"""
import torch


def get_device(preferred_device=None):
    """
    Detect and configure the best available device for training.

    Args:
        preferred_device: Optional device string ('cuda', 'cuda:0', 'mps', 'cpu')

    Returns:
        torch.device: The configured device

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda')  # Force CUDA
    """
    print("\n" + "="*70)
    print("GPU DETECTION AND CONFIGURATION")
    print("="*70)

    # If user specified a device, try to use it
    if preferred_device:
        try:
            device = torch.device(preferred_device)
            print(f"Using user-specified device: {device}")
            return device
        except Exception as e:
            print(f"Warning: Could not use specified device '{preferred_device}': {e}")
            print("Falling back to automatic detection...")

    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()

        print(f"✓ CUDA is available")
        print(f"✓ Number of GPUs: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Memory: {gpu_memory:.2f} GB")

        current_device = torch.cuda.current_device()
        print(f"✓ Using GPU: {torch.cuda.get_device_name(current_device)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")

    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Apple MPS (Metal Performance Shaders) is available")
        print(f"✓ Using device: {device}")

    # Fallback to CPU
    else:
        device = torch.device('cpu')
        print(f"⚠ No GPU detected. Using CPU")
        print(f"  Training will be significantly slower on CPU")
        print(f"  Consider using a machine with CUDA-capable GPU")

    print(f"✓ PyTorch Version: {torch.__version__}")
    print("="*70 + "\n")

    return device
