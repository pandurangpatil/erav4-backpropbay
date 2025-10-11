# AWS SageMaker Deployment Guide

## Complete Step-by-Step Deployment Instructions

This guide walks you through deploying the CIFAR-100 WideResNet model to AWS SageMaker from scratch.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] AWS Account with billing enabled
- [ ] AWS CLI installed (`aws --version`)
- [ ] AWS credentials configured (`aws configure`)
- [ ] Python 3.8+ installed
- [ ] Basic understanding of AWS IAM and S3

---

## Step 1: AWS Account Setup (15 minutes)

### 1.1 Install AWS CLI

**macOS:**
```bash
brew install awscli
```

**Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Windows:**
Download from: https://aws.amazon.com/cli/

### 1.2 Configure AWS Credentials

```bash
aws configure
```

Enter:
- **AWS Access Key ID**: (from IAM Console)
- **AWS Secret Access Key**: (from IAM Console)
- **Default region**: `us-east-1` (recommended)
- **Default output format**: `json`

### 1.3 Verify Setup

```bash
# Check credentials
aws sts get-caller-identity

# Expected output:
# {
#     "UserId": "...",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/..."
# }
```

---

## Step 2: Test Locally (Optional but Recommended)

Before deploying to AWS, test the code locally:

```bash
cd sagemaker-cifar100

# Install dependencies
pip install -r requirements.txt

# Run local test (2 epochs, quick validation)
python local_test.py --epochs 2 --batch-size 32

# Expected: Should complete without errors
```

---

## Step 3: Quick Deployment (Automated)

### 3.1 Using the Deploy Script (Easiest)

The deploy script handles everything automatically:

```bash
# Make script executable
chmod +x deploy.sh

# Deploy with default settings
./deploy.sh --monitor

# Or deploy with spot instances (70% cheaper)
./deploy.sh --spot --monitor
```

The script will:
1. ✓ Check dependencies
2. ✓ Verify AWS credentials
3. ✓ Create S3 bucket
4. ✓ Create IAM role
5. ✓ Upload code to S3
6. ✓ Launch training job
7. ✓ Monitor progress (if --monitor flag used)

### 3.2 Customization Options

```bash
# Different region
./deploy.sh --region us-west-2

# Larger instance
./deploy.sh --instance-type ml.p3.8xlarge

# Custom job name
./deploy.sh --name my-experiment-v1

# Combine options
./deploy.sh --spot --region us-west-2 --monitor
```

---

## Step 4: Manual Deployment (Advanced)

For more control, deploy manually:

### 4.1 Create S3 Bucket

```bash
# Set variables
export BUCKET_NAME="sagemaker-cifar100-$(date +%s)"
export REGION="us-east-1"

# Create bucket
aws s3 mb s3://${BUCKET_NAME} --region ${REGION}
```

### 4.2 Create IAM Role

```bash
# Create trust policy
cat > trust-policy.json <<'EOF'
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
  --role-name SageMaker-CIFAR100-Role \
  --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name SageMaker-CIFAR100-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Wait for role to propagate
sleep 10
```

### 4.3 Upload Training Code

```bash
# Package code
cd src
tar -czf ../sourcedir.tar.gz .
cd ..

# Upload to S3
aws s3 cp sourcedir.tar.gz s3://${BUCKET_NAME}/code/
```

### 4.4 Launch Training Job

```bash
# Get account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/SageMaker-CIFAR100-Role"
JOB_NAME="cifar100-$(date +%Y%m%d-%H%M%S)"

# Create training job
aws sagemaker create-training-job \
  --training-job-name ${JOB_NAME} \
  --role-arn ${ROLE_ARN} \
  --algorithm-specification \
    TrainingImage=763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.0.0-gpu-py310,\
    TrainingInputMode=File,\
    EnableSageMakerMetricsTimeSeries=true \
  --input-data-config '[
    {
      "ChannelName": "training",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://'${BUCKET_NAME}'/data",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    }
  ]' \
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
    widen-factor=10,\
    dropout=0.3,\
    mixup-alpha=0.2,\
    label-smoothing=0.1,\
    warmup-epochs=5

echo "Training job launched: ${JOB_NAME}"
```

---

## Step 5: Monitor Training

### 5.1 AWS Console (Visual)

1. Navigate to: https://console.aws.amazon.com/sagemaker/
2. Click **Training jobs** in left menu
3. Find your job (e.g., `cifar100-20250111-143000`)
4. View metrics, logs, and status

### 5.2 AWS CLI (Command Line)

```bash
# Check job status
aws sagemaker describe-training-job \
  --training-job-name ${JOB_NAME} \
  --query 'TrainingJobStatus' \
  --output text

# Get full job details
aws sagemaker describe-training-job \
  --training-job-name ${JOB_NAME}

# Stream CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs --follow \
  --log-stream-name-prefix ${JOB_NAME}
```

### 5.3 Monitor Script

```bash
# Watch job until completion
watch -n 30 'aws sagemaker describe-training-job \
  --training-job-name '${JOB_NAME}' \
  --query "TrainingJobStatus" \
  --output text'
```

---

## Step 6: Retrieve Results

### 6.1 Download Model Artifacts

```bash
# Get model location
MODEL_ARTIFACTS=$(aws sagemaker describe-training-job \
  --training-job-name ${JOB_NAME} \
  --query 'ModelArtifacts.S3ModelArtifacts' \
  --output text)

echo "Model artifacts: ${MODEL_ARTIFACTS}"

# Download model
aws s3 cp ${MODEL_ARTIFACTS} ./model.tar.gz

# Extract
tar -xzf model.tar.gz
```

### 6.2 View Training Metrics

```bash
# Download metrics
aws s3 cp s3://${BUCKET_NAME}/output/${JOB_NAME}/output/metrics.json ./

# View metrics
cat metrics.json | jq '.best_test_accuracy'
```

---

## Step 7: Cleanup (Important!)

To avoid ongoing charges:

### 7.1 Stop Training Job (if needed)

```bash
aws sagemaker stop-training-job --training-job-name ${JOB_NAME}
```

### 7.2 Delete S3 Bucket

```bash
# Delete all objects first
aws s3 rm s3://${BUCKET_NAME} --recursive

# Delete bucket
aws s3 rb s3://${BUCKET_NAME}
```

### 7.3 Delete IAM Role

```bash
# Detach policies
aws iam detach-role-policy \
  --role-name SageMaker-CIFAR100-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Delete role
aws iam delete-role --role-name SageMaker-CIFAR100-Role
```

---

## Cost Management

### Estimated Costs

**Training Cost (ml.p3.2xlarge)**
- On-demand: $3.06/hour
- Spot: ~$0.92/hour (70% discount)
- Expected duration: 6-8 hours
- **Total: $5-25 per training run**

**Storage Cost**
- S3: $0.023/GB/month
- Training data: ~165MB
- Model: ~140MB
- **Total: ~$0.01/month**

### Cost Optimization Tips

1. **Use Spot Instances**: Save 70%
   ```bash
   ./deploy.sh --spot
   ```

2. **Use Smaller Instances for Testing**:
   ```bash
   ./deploy.sh --instance-type ml.g4dn.xlarge
   ```

3. **Set Budget Alerts**:
   - Go to AWS Billing Console
   - Create budget alert for $50/month

4. **Clean Up Resources**:
   - Delete old training jobs
   - Remove unused S3 objects
   - Stop idle endpoints

---

## Troubleshooting

### Problem: "Access Denied" Error

**Solution**:
```bash
# Check IAM permissions
aws iam list-attached-role-policies --role-name SageMaker-CIFAR100-Role

# Ensure AmazonSageMakerFullAccess is attached
```

### Problem: "Insufficient Capacity"

**Solution**:
```bash
# Try different region or instance type
./deploy.sh --region us-west-2 --instance-type ml.g4dn.xlarge
```

### Problem: Training Job Fails

**Solution**:
```bash
# Check CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix ${JOB_NAME} \
  --follow

# Common issues:
# - Out of memory: Reduce batch size
# - Package missing: Add to requirements.txt
# - Data not found: Check S3 paths
```

### Problem: Slow Training

**Solution**:
- Use larger instance: `ml.p3.8xlarge`
- Reduce num_workers if CPU-bound
- Check if using GPU: Look for CUDA in logs

---

## Next Steps

After successful deployment:

1. **Experiment with Hyperparameters**
   - Edit `config.yaml`
   - Try different learning rates, batch sizes

2. **Deploy as Endpoint**
   - Use `inference.py` for real-time predictions
   - See AWS SageMaker Hosting documentation

3. **Set Up Continuous Training**
   - Use AWS Step Functions
   - Schedule periodic retraining

4. **Integrate with MLOps**
   - Use SageMaker Pipelines
   - Add model monitoring
   - Set up A/B testing

---

## Support

- **AWS Documentation**: https://docs.aws.amazon.com/sagemaker/
- **PyTorch on SageMaker**: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/
- **GitHub Issues**: (your repository)

---

**Happy Training! 🚀**
