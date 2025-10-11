# Quick Start Guide

Get your CIFAR-100 model training on AWS SageMaker in 5 minutes!

## Prerequisites

- AWS account configured (`aws configure`)
- Python 3.8+ installed
- This repository downloaded

## Option 1: Automated Deployment (Recommended)

```bash
cd sagemaker-cifar100

# Deploy with spot instances (saves 70% cost)
./deploy.sh --spot --monitor
```

That's it! The script will:
- ✅ Create S3 bucket
- ✅ Create IAM role
- ✅ Upload code
- ✅ Launch training
- ✅ Monitor progress

**Expected Time**: 6-8 hours
**Expected Cost**: $5-8 (with spot instances)
**Expected Accuracy**: 71-74%

## Option 2: Test Locally First

```bash
# Install dependencies
pip install -r requirements.txt

# Quick local test (2 epochs)
python local_test.py

# If successful, deploy to AWS
./deploy.sh --spot --monitor
```

## Monitor Your Training

### AWS Console
Visit: https://console.aws.amazon.com/sagemaker/home#/jobs

### Command Line
```bash
# Check status
aws sagemaker list-training-jobs --max-results 1

# View logs
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

## Get Results

After training completes:

```bash
# Model artifacts are automatically saved to S3
# Location printed at end of training

# Download model
aws sagemaker describe-training-job \
  --training-job-name <job-name> \
  --query 'ModelArtifacts.S3ModelArtifacts'
```

## Cleanup

```bash
# Stop training (if needed)
aws sagemaker stop-training-job --training-job-name <job-name>

# Delete S3 bucket (to avoid charges)
aws s3 rb s3://<bucket-name> --force
```

## Troubleshooting

**Problem**: "Access Denied"
**Solution**: Run `aws configure` and check credentials

**Problem**: "Insufficient Capacity"
**Solution**: Try different region: `./deploy.sh --region us-west-2`

**Problem**: "Out of Memory"
**Solution**: Reduce batch size in config.yaml

## Cost Breakdown

| Item | Cost |
|------|------|
| Training (8 hrs, spot) | $7.36 |
| S3 Storage (1 month) | $0.01 |
| **Total** | **~$7.50** |

## Next Steps

- ✅ Check `README.md` for detailed documentation
- ✅ See `DEPLOYMENT_GUIDE.md` for step-by-step instructions
- ✅ Modify `config.yaml` to experiment with hyperparameters
- ✅ Use `inference.py` to deploy as real-time endpoint

## Support

For issues: Open a GitHub issue or check AWS SageMaker documentation

---

**Happy Training! 🚀**
