# CIFAR-100 WideResNet Training on AWS SageMaker

Complete AWS SageMaker training module for CIFAR-100 image classification using WideResNet-28-10.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Cost Estimation](#cost-estimation)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## 🎯 Overview

This project provides a production-ready implementation of WideResNet-28-10 for CIFAR-100 classification, optimized for AWS SageMaker training. It includes:

- **Model**: WideResNet-28-10 (36.5M parameters)
- **Dataset**: CIFAR-100 (50,000 train, 10,000 test images)
- **Target Accuracy**: 74%+
- **Training Time**: ~6-8 hours on ml.p3.2xlarge

## ✨ Features

- 🚀 **SageMaker-native** training with S3 integration
- 🎨 **Advanced augmentation** (MixUp, Cutout, Albumentations)
- 📊 **Mixed precision** training (faster, less memory)
- 🔄 **Cosine annealing** with warmup scheduler
- 💾 **Automatic checkpointing** to S3
- 🤗 **HuggingFace Hub** integration (optional)
- 💰 **Spot instance** support for cost savings
- 📈 **CloudWatch metrics** logging
- 🔧 **Easy deployment** via shell script

## 📁 Project Structure

```
sagemaker-cifar100/
├── src/
│   ├── train.py              # Main SageMaker training script
│   ├── model.py              # WideResNet architecture
│   ├── data.py               # Data loading & augmentation
│   ├── utils.py              # Training utilities
│   └── inference.py          # SageMaker inference handler
├── requirements.txt          # Python dependencies
├── config.yaml               # Hyperparameter configuration
├── deploy.sh                 # AWS CLI deployment script
├── Dockerfile                # Custom container (optional)
└── README.md                 # This file
```

## 🔧 Prerequisites

### 1. AWS Account Setup

- AWS account with appropriate permissions
- AWS CLI installed and configured
- IAM user with SageMaker access

### 2. Local Requirements

```bash
# Required
aws-cli >= 2.0.0
python >= 3.8

# Optional (for local testing)
docker >= 20.0.0
```

### 3. AWS Permissions

Your IAM user/role needs:
- `AmazonSageMakerFullAccess` (or custom policy)
- `AmazonS3FullAccess` (for S3 buckets)
- `IAMFullAccess` (for role creation)

## 🚀 Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Clone/navigate to the project
cd sagemaker-cifar100

# Make deploy script executable (if not already)
chmod +x deploy.sh

# Deploy with default settings
./deploy.sh

# Deploy with monitoring
./deploy.sh --monitor

# Deploy with spot instances (70% cost savings)
./deploy.sh --spot --monitor
```

### Option 2: Manual Deployment

See [Detailed Setup](#detailed-setup) below.

## 📚 Detailed Setup

### Step 1: Configure AWS CLI

```bash
# Configure AWS credentials
aws configure

# Verify credentials
aws sts get-caller-identity
```

### Step 2: Create S3 Bucket

```bash
# Set variables
export BUCKET_NAME="sagemaker-cifar100-${USER}"
export REGION="us-east-1"

# Create bucket
aws s3 mb s3://${BUCKET_NAME} --region ${REGION}
```

### Step 3: Create IAM Role

```bash
# Create trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name SageMaker-ExecutionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach policy
aws iam attach-role-policy \
  --role-name SageMaker-ExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

### Step 4: Upload Training Code

```bash
# Package source code
cd src
tar -czf ../sourcedir.tar.gz .
cd ..

# Upload to S3
aws s3 cp sourcedir.tar.gz s3://${BUCKET_NAME}/code/
```

### Step 5: Launch Training Job

```bash
# Get account ID and role ARN
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/SageMaker-ExecutionRole"

# Create training job
aws sagemaker create-training-job \
  --training-job-name cifar100-$(date +%Y%m%d-%H%M%S) \
  --role-arn ${ROLE_ARN} \
  --algorithm-specification \
    TrainingImage=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310,\
    TrainingInputMode=File \
  --input-data-config \
    '[{"ChannelName":"training","DataSource":{"S3DataSource":{"S3DataType":"S3Prefix","S3Uri":"s3://'${BUCKET_NAME}'/data","S3DataDistributionType":"FullyReplicated"}}}]' \
  --output-data-config S3OutputPath=s3://${BUCKET_NAME}/output \
  --resource-config \
    InstanceType=ml.p3.2xlarge,\
    InstanceCount=1,\
    VolumeSizeInGB=30 \
  --stopping-condition MaxRuntimeInSeconds=86400 \
  --hyper-parameters \
    epochs=100,\
    batch-size=256,\
    lr=0.1,\
    depth=28,\
    widen-factor=10
```

## ⚙️ Configuration

### Hyperparameters

Edit `config.yaml` or pass via command line:

```yaml
model:
  depth: 28                    # Network depth
  widen_factor: 10             # Width multiplier
  dropout: 0.3                 # Dropout rate

training:
  epochs: 100                  # Training epochs
  batch_size: 256              # Batch size
  lr: 0.1                      # Learning rate
```

### Instance Types

| Instance Type | GPU | VRAM | Cost/hour | Recommended For |
|--------------|-----|------|-----------|-----------------|
| ml.p3.2xlarge | 1x V100 | 16GB | $3.06 | **Recommended** |
| ml.p3.8xlarge | 4x V100 | 64GB | $12.24 | Multi-GPU |
| ml.g4dn.xlarge | 1x T4 | 16GB | $0.526 | Budget (slower) |
| ml.g5.xlarge | 1x A10G | 24GB | $1.006 | Good balance |

💡 **Tip**: Use spot instances for 70% cost savings:
```bash
./deploy.sh --spot --monitor
```

## 📊 Monitoring

### CloudWatch Logs

```bash
# View training logs
aws logs tail /aws/sagemaker/TrainingJobs --follow \
  --log-stream-name-prefix cifar100-
```

### SageMaker Console

Monitor training at:
```
https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs
```

### Describe Training Job

```bash
aws sagemaker describe-training-job \
  --training-job-name <job-name>
```

## 💰 Cost Estimation

### Training Cost (ml.p3.2xlarge)

| Duration | On-Demand | Spot (70% off) |
|----------|-----------|----------------|
| 6 hours  | $18.36    | $5.51          |
| 12 hours | $36.72    | $11.02         |
| 24 hours | $73.44    | $22.03         |

### Storage Cost

- S3 storage: ~$0.023/GB/month
- Training data: ~165MB (negligible)
- Model checkpoints: ~500MB ($0.01/month)

### Total Estimated Cost

**Single training run**: $5-20 (depending on duration and spot usage)

## 🔍 Troubleshooting

### Common Issues

#### 1. Role doesn't exist

```bash
# Create the role
./deploy.sh  # Will create automatically
```

#### 2. Insufficient capacity

Try different instance types or regions:
```bash
./deploy.sh --instance-type ml.g4dn.xlarge --region us-west-2
```

#### 3. Out of memory

Reduce batch size:
```bash
# Edit config.yaml or pass via hyperparameters
--batch-size 128
```

#### 4. Slow training

- Use larger instance (ml.p3.8xlarge)
- Enable spot instances
- Reduce num_workers if CPU-bound

### Debug Mode

Run training locally first:

```bash
cd src
python train.py \
  --epochs 2 \
  --batch-size 32 \
  --train-dir ./data
```

## 🚀 Advanced Usage

### Multi-GPU Training

```bash
./deploy.sh --instance-type ml.p3.8xlarge --instance-count 1
```

Update `train.py` to use `DistributedDataParallel`:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])
```

### Custom Docker Container

Build and use custom container:

```bash
# Build image
docker build -t cifar100-training .

# Push to ECR
aws ecr create-repository --repository-name cifar100-training
docker tag cifar100-training:latest ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/cifar100-training:latest
docker push ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/cifar100-training:latest

# Use in training job
# (Update deploy.sh to use custom image)
```

### HuggingFace Integration

Upload models to HuggingFace Hub:

```bash
# Set environment variables
export HF_TOKEN="your_token_here"
export HF_REPO_ID="username/cifar100-wideresnet"

# Pass to training
python src/train.py \
  --hf-token ${HF_TOKEN} \
  --hf-repo-id ${HF_REPO_ID}
```

### Hyperparameter Tuning

Use SageMaker Hyperparameter Tuning:

```bash
aws sagemaker create-hyper-parameter-tuning-job \
  --hyper-parameter-tuning-job-name cifar100-tuning-$(date +%s) \
  --hyper-parameter-tuning-job-config file://tuning-config.json \
  --training-job-definition file://training-job-definition.json
```

## 📈 Expected Results

### Training Metrics

- **Training Accuracy**: 75-85%
- **Test Accuracy**: 71-74%
- **Training Time**: 6-8 hours (ml.p3.2xlarge)
- **Best Epoch**: 75-100

### Model Performance

| Metric | Value |
|--------|-------|
| Parameters | 36.5M |
| Model Size | ~140MB |
| Inference Time | ~5ms/image (GPU) |
| Top-1 Accuracy | 71-74% |
| Top-5 Accuracy | 90-92% |

## 📝 Notes

- CIFAR-100 dataset is downloaded automatically during training
- Checkpoints are saved every 10 epochs
- Best model is saved automatically
- Early stopping with patience=15 epochs
- Gradient clipping prevents exploding gradients
- Mixed precision training is enabled by default

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License

## 🔗 References

- [Wide Residual Networks Paper](https://arxiv.org/abs/1605.07146)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 📧 Support

For issues specific to this implementation, please open a GitHub issue.

For AWS SageMaker support, visit [AWS Support](https://aws.amazon.com/support/).

---

**Created by**: Pandurang Patil
**Date**: 2025
**Version**: 1.0
