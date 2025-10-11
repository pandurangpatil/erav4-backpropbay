"""
SageMaker Inference Handler for CIFAR-100 WideResNet

This module handles model serving for SageMaker endpoints
"""

import os
import json
import io
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

from model import create_model


# CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# CIFAR-100 normalization constants
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)


def model_fn(model_dir):
    """
    Load the model for inference

    Args:
        model_dir: Directory where model artifacts are stored

    Returns:
        Loaded PyTorch model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        depth = config.get('depth', 28)
        widen_factor = config.get('widen_factor', 10)
        dropout = config.get('dropout', 0.3)
    else:
        # Default configuration
        depth = 28
        widen_factor = 10
        dropout = 0.3

    # Create model
    model = create_model(depth=depth, widen_factor=widen_factor,
                        dropout=dropout, num_classes=100)

    # Load weights
    checkpoint_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_dir, 'model.pth')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    return model


def input_fn(request_body, content_type='application/json'):
    """
    Deserialize and prepare the prediction input

    Args:
        request_body: The request payload
        content_type: The content type of the request

    Returns:
        Preprocessed input tensor
    """
    if content_type == 'application/json':
        # Input is JSON with base64 encoded image or image array
        data = json.loads(request_body)

        if 'image' in data:
            # Base64 encoded image
            import base64
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif 'array' in data:
            # NumPy array
            arr = np.array(data['array'])
            image = Image.fromarray(arr.astype('uint8')).convert('RGB')
        else:
            raise ValueError("JSON must contain 'image' (base64) or 'array' field")

    elif content_type in ['image/jpeg', 'image/png', 'image/jpg']:
        # Direct image upload
        image = Image.open(io.BytesIO(request_body)).convert('RGB')

    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-100 size
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    ])

    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def predict_fn(input_data, model):
    """
    Make prediction

    Args:
        input_data: Preprocessed input tensor
        model: Loaded model

    Returns:
        Model predictions
    """
    device = next(model.parameters()).device
    input_data = input_data.to(device)

    with torch.no_grad():
        output = model(input_data)
        probabilities = F.softmax(output, dim=1)

    return probabilities


def output_fn(prediction, accept='application/json'):
    """
    Serialize and prepare the prediction output

    Args:
        prediction: Model output probabilities
        accept: Accept header from request

    Returns:
        Serialized prediction
    """
    if accept == 'application/json':
        # Get top-5 predictions
        probs, indices = torch.topk(prediction[0], k=5)

        results = {
            'predictions': [
                {
                    'class_id': int(idx),
                    'class_name': CIFAR100_CLASSES[idx],
                    'probability': float(prob)
                }
                for idx, prob in zip(indices, probs)
            ],
            'top_prediction': {
                'class_id': int(indices[0]),
                'class_name': CIFAR100_CLASSES[indices[0]],
                'probability': float(probs[0])
            }
        }

        return json.dumps(results)

    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# Optional: Handler for SageMaker PyTorch serving container
def handle(data, context):
    """
    Custom handler for SageMaker PyTorch serving

    Args:
        data: Input data
        context: Context object with model and other info

    Returns:
        Prediction output
    """
    if not hasattr(handle, 'model'):
        handle.model = model_fn(context.system_properties.get('model_dir'))

    if data is None:
        return None

    # Get content type from context
    content_type = context.request_headers.get('Content-Type', 'application/json')

    # Process input
    input_data = input_fn(data.read(), content_type)

    # Make prediction
    prediction = predict_fn(input_data, handle.model)

    # Format output
    accept = context.request_headers.get('Accept', 'application/json')
    return output_fn(prediction, accept)


if __name__ == "__main__":
    # Test inference pipeline
    print("Testing inference pipeline...")

    # Create a dummy model
    model = create_model(depth=28, widen_factor=10, dropout=0.3)
    model.eval()

    # Create a test image
    test_image = Image.new('RGB', (32, 32), color='red')

    # Test preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    ])
    tensor = transform(test_image).unsqueeze(0)

    # Test prediction
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)

    # Test output formatting
    probs, indices = torch.topk(probabilities[0], k=5)
    print(f"Top prediction: {CIFAR100_CLASSES[indices[0]]} ({probs[0]:.4f})")

    print("Inference pipeline test passed!")
