# AWS SageMaker Deployment Checklist

## Pre-Deployment Checklist

### AWS Account Setup
- [ ] AWS account created and billing enabled
- [ ] AWS CLI installed (`aws --version`)
- [ ] AWS credentials configured (`aws configure`)
- [ ] Credentials verified (`aws sts get-caller-identity`)
- [ ] Default region set (e.g., `us-east-1`)

### Local Environment
- [ ] Python 3.8+ installed (`python --version`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Project files downloaded/cloned
- [ ] Deployment script is executable (`chmod +x deploy.sh`)

### Optional: Local Testing
- [ ] Local test runs successfully (`python local_test.py`)
- [ ] Training completes without errors
- [ ] Model saves correctly
- [ ] Metrics are generated

## Deployment Checklist

### Quick Deployment (Automated)
- [ ] Navigate to project directory (`cd sagemaker-cifar100`)
- [ ] Review configuration (`cat config.yaml`)
- [ ] Run deployment script (`./deploy.sh --spot --monitor`)
- [ ] Note the job name printed by script
- [ ] Bookmark AWS console URL provided

### Manual Deployment (Advanced)
- [ ] S3 bucket created
- [ ] IAM role created with SageMaker permissions
- [ ] Training code packaged and uploaded to S3
- [ ] Training job configuration prepared
- [ ] Training job launched successfully
- [ ] Job name recorded for monitoring

## Post-Deployment Checklist

### Monitoring
- [ ] Training job appears in SageMaker console
- [ ] Job status is "InProgress"
- [ ] CloudWatch logs are being generated
- [ ] GPU utilization is high (>80%)
- [ ] Training metrics are increasing

### During Training (Check Every Hour)
- [ ] Training accuracy improving
- [ ] Test accuracy improving
- [ ] No error messages in logs
- [ ] Loss decreasing steadily
- [ ] Learning rate following expected schedule

## Completion Checklist

### After Training Finishes
- [ ] Training job status is "Completed"
- [ ] Final accuracy meets expectations (>70%)
- [ ] Model artifacts saved to S3
- [ ] Metrics file generated (metrics.json)
- [ ] Best model checkpoint saved
- [ ] Configuration saved

### Retrieve Results
- [ ] Model artifacts downloaded from S3
- [ ] Metrics file downloaded
- [ ] Training logs reviewed
- [ ] Final accuracy documented
- [ ] Training time recorded
- [ ] Cost calculated

## Cleanup Checklist

### To Avoid Ongoing Charges
- [ ] Training job completed or stopped
- [ ] No active SageMaker endpoints running
- [ ] S3 bucket contents reviewed
- [ ] Unnecessary files deleted from S3
- [ ] Consider keeping: best model, final metrics
- [ ] Optional: Delete S3 bucket entirely
- [ ] Optional: Delete IAM role if not reusing

### Cost Verification
- [ ] Check AWS billing dashboard
- [ ] Verify expected charges
- [ ] Set up billing alerts for future
- [ ] Document actual cost vs estimate

## Troubleshooting Checklist

### If Training Fails
- [ ] Check CloudWatch logs for errors
- [ ] Verify S3 paths are correct
- [ ] Confirm IAM role has correct permissions
- [ ] Check if instance type available in region
- [ ] Review hyperparameters for invalid values
- [ ] Try with smaller instance for testing

### If Out of Memory
- [ ] Reduce batch size in config.yaml
- [ ] Use larger instance type
- [ ] Reduce number of workers
- [ ] Check for memory leaks in custom code

### If Training Too Slow
- [ ] Verify GPU is being used (check logs)
- [ ] Increase batch size if memory allows
- [ ] Use larger instance (e.g., ml.p3.8xlarge)
- [ ] Check if data loading is bottleneck

## Next Steps Checklist

### After Successful Training
- [ ] Document final accuracy and settings
- [ ] Save model for future use
- [ ] Consider hyperparameter tuning
- [ ] Plan model deployment strategy
- [ ] Set up continuous training if needed

### Model Deployment (Optional)
- [ ] Create SageMaker model from artifacts
- [ ] Create endpoint configuration
- [ ] Deploy as real-time endpoint
- [ ] Test endpoint with sample data
- [ ] Set up endpoint monitoring
- [ ] Configure auto-scaling if needed

### Experimentation (Optional)
- [ ] Try different learning rates
- [ ] Adjust MixUp alpha parameter
- [ ] Experiment with dropout rates
- [ ] Test different batch sizes
- [ ] Try different augmentation strategies

## Documentation Checklist

### Record Keeping
- [ ] Training job name recorded
- [ ] Final accuracy documented
- [ ] Training time logged
- [ ] Cost calculated and recorded
- [ ] Hyperparameters documented
- [ ] Any issues/solutions documented

### Knowledge Sharing
- [ ] Update team documentation
- [ ] Share learnings with team
- [ ] Document any custom modifications
- [ ] Update cost estimates based on actual
- [ ] Create training report if required

## Security Checklist

### Best Practices
- [ ] IAM role uses least-privilege permissions
- [ ] S3 bucket has appropriate access controls
- [ ] No credentials hardcoded in scripts
- [ ] HuggingFace token stored securely (if used)
- [ ] Training data appropriately encrypted
- [ ] Logs don't contain sensitive data

### Compliance (If Applicable)
- [ ] Data usage complies with policies
- [ ] Model training documented for audit
- [ ] Access logs maintained
- [ ] Data retention policy followed

## Quick Reference

### Essential Commands
```bash
# Check training status
aws sagemaker describe-training-job --training-job-name <job-name>

# View logs
aws logs tail /aws/sagemaker/TrainingJobs --follow

# List recent jobs
aws sagemaker list-training-jobs --max-results 5

# Stop training
aws sagemaker stop-training-job --training-job-name <job-name>

# Clean up S3
aws s3 rm s3://<bucket-name> --recursive
aws s3 rb s3://<bucket-name>
```

### Important URLs
- SageMaker Console: https://console.aws.amazon.com/sagemaker/
- Billing Dashboard: https://console.aws.amazon.com/billing/
- CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/

## Sign-Off

### Before Starting
- [ ] I have read the README.md
- [ ] I understand the expected costs
- [ ] I have a valid AWS account
- [ ] I am ready to deploy

### After Completion
- [ ] Training completed successfully
- [ ] Results documented
- [ ] Resources cleaned up
- [ ] Costs verified
- [ ] Project complete ✅

---

**Date**: _______________
**Job Name**: _______________
**Final Accuracy**: _______________
**Total Cost**: _______________
**Notes**:
_______________________________________________
_______________________________________________
_______________________________________________

**Print this checklist and use it during deployment!**
