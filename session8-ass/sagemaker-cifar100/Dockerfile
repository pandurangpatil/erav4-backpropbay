# Dockerfile for CIFAR-100 WideResNet Training on AWS SageMaker
# Based on PyTorch official image with SageMaker compatibility

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Copy training code
COPY src/ /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# SageMaker expects the training script to be executable
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# Set Python path
ENV PYTHONPATH=/opt/ml/code:$PYTHONPATH

# Create necessary directories for SageMaker
RUN mkdir -p /opt/ml/model && \
    mkdir -p /opt/ml/input/data/training && \
    mkdir -p /opt/ml/output

# Set permissions
RUN chmod -R 755 /opt/ml/code

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('PyTorch version:', torch.__version__)" || exit 1

# Entry point for SageMaker training
ENTRYPOINT ["python", "train.py"]
