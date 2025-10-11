#!/bin/bash

################################################################################
# AWS SageMaker Deployment Script for CIFAR-100 WideResNet Training
#
# This script automates the deployment of training jobs to AWS SageMaker
# Usage: ./deploy.sh [OPTIONS]
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration (modify these as needed)
PROJECT_NAME="cifar100-wideresnet"
REGION="${AWS_REGION:-us-east-1}"
ROLE_NAME="${SAGEMAKER_ROLE_NAME:-SageMaker-ExecutionRole}"
S3_BUCKET="${S3_BUCKET:-sagemaker-${PROJECT_NAME}-$(date +%s)}"
ECR_REPOSITORY="${ECR_REPOSITORY:-${PROJECT_NAME}-training}"

# Training configuration
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.p3.2xlarge}"
INSTANCE_COUNT="${INSTANCE_COUNT:-1}"
VOLUME_SIZE="${VOLUME_SIZE:-30}"
MAX_RUN="${MAX_RUN:-86400}"  # 24 hours
USE_SPOT="${USE_SPOT:-false}"

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
JOB_NAME="${PROJECT_NAME}-$(date +%Y%m%d-%H%M%S)"

################################################################################
# Helper Functions
################################################################################

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_info "Checking dependencies..."

    local missing_deps=()

    if ! command -v aws &> /dev/null; then
        missing_deps+=("aws-cli")
    fi

    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    if ! command -v jq &> /dev/null; then
        print_warning "jq not found (optional, for JSON parsing)"
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install missing dependencies and try again"
        exit 1
    fi

    print_success "All required dependencies found"
}

check_aws_credentials() {
    print_info "Checking AWS credentials..."

    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured"
        print_error "Run: aws configure"
        exit 1
    fi

    local account_id=$(aws sts get-caller-identity --query Account --output text)
    print_success "AWS credentials valid (Account: ${account_id})"
}

create_s3_bucket() {
    print_info "Setting up S3 bucket: ${S3_BUCKET}"

    if aws s3 ls "s3://${S3_BUCKET}" 2>&1 | grep -q 'NoSuchBucket'; then
        print_info "Creating S3 bucket..."

        if [ "${REGION}" = "us-east-1" ]; then
            aws s3 mb "s3://${S3_BUCKET}" --region "${REGION}"
        else
            aws s3 mb "s3://${S3_BUCKET}" --region "${REGION}" \
                --create-bucket-configuration LocationConstraint="${REGION}"
        fi

        print_success "S3 bucket created: ${S3_BUCKET}"
    else
        print_success "S3 bucket already exists: ${S3_BUCKET}"
    fi
}

create_iam_role() {
    print_info "Checking IAM role: ${ROLE_NAME}"

    if ! aws iam get-role --role-name "${ROLE_NAME}" &> /dev/null; then
        print_info "Creating IAM role..."

        # Create trust policy
        cat > /tmp/trust-policy.json <<EOF
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

        aws iam create-role \
            --role-name "${ROLE_NAME}" \
            --assume-role-policy-document file:///tmp/trust-policy.json \
            --description "Execution role for SageMaker training jobs"

        # Attach managed policy
        aws iam attach-role-policy \
            --role-name "${ROLE_NAME}" \
            --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

        print_success "IAM role created: ${ROLE_NAME}"
        print_warning "Waiting 10 seconds for IAM role to propagate..."
        sleep 10
    else
        print_success "IAM role already exists: ${ROLE_NAME}"
    fi
}

upload_code_to_s3() {
    print_info "Uploading source code to S3..."

    local s3_code_path="s3://${S3_BUCKET}/code/${JOB_NAME}"

    # Create a tarball of the source code
    cd "${SCRIPT_DIR}"
    tar -czf /tmp/sourcedir.tar.gz -C src .

    # Upload to S3
    aws s3 cp /tmp/sourcedir.tar.gz "${s3_code_path}/sourcedir.tar.gz"

    print_success "Code uploaded to: ${s3_code_path}"
    echo "${s3_code_path}"
}

create_training_job() {
    print_info "Creating SageMaker training job: ${JOB_NAME}"

    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local role_arn="arn:aws:iam::${account_id}:role/${ROLE_NAME}"
    local s3_code_path=$(upload_code_to_s3)
    local s3_output_path="s3://${S3_BUCKET}/output"

    # Build hyperparameters
    local hyperparameters=$(cat <<EOF
{
  "epochs": 100,
  "batch-size": 256,
  "lr": 0.1,
  "depth": 28,
  "widen-factor": 10,
  "dropout": 0.3,
  "mixup-alpha": 0.2,
  "label-smoothing": 0.1,
  "weight-decay": 0.001,
  "warmup-epochs": 5,
  "cosine-t0": 25,
  "patience": 15,
  "target-accuracy": 74.0
}
EOF
)

    # Create training job configuration
    cat > /tmp/training-job.json <<EOF
{
  "TrainingJobName": "${JOB_NAME}",
  "RoleArn": "${role_arn}",
  "AlgorithmSpecification": {
    "TrainingImage": "763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.0.0-gpu-py310",
    "TrainingInputMode": "File",
    "EnableSageMakerMetricsTimeSeries": true
  },
  "InputDataConfig": [
    {
      "ChannelName": "training",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://${S3_BUCKET}/data",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "${s3_output_path}"
  },
  "ResourceConfig": {
    "InstanceType": "${INSTANCE_TYPE}",
    "InstanceCount": ${INSTANCE_COUNT},
    "VolumeSizeInGB": ${VOLUME_SIZE}
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": ${MAX_RUN}
  },
  "HyperParameters": $(echo "${hyperparameters}" | jq -c 'with_entries(.value |= tostring)'),
  "EnableInterContainerTrafficEncryption": false,
  "EnableNetworkIsolation": false
}
EOF

    # Add spot instance configuration if enabled
    if [ "${USE_SPOT}" = "true" ]; then
        print_info "Enabling managed spot training..."
        jq '.EnableManagedSpotTraining = true | .CheckpointConfig = {"S3Uri": "s3://'"${S3_BUCKET}"'/checkpoints"}' \
            /tmp/training-job.json > /tmp/training-job-spot.json
        mv /tmp/training-job-spot.json /tmp/training-job.json
    fi

    # Create training job
    aws sagemaker create-training-job \
        --cli-input-json file:///tmp/training-job.json \
        --region "${REGION}"

    print_success "Training job created: ${JOB_NAME}"
    print_info "Monitor progress at: https://${REGION}.console.aws.amazon.com/sagemaker/home?region=${REGION}#/jobs/${JOB_NAME}"
}

monitor_training_job() {
    print_info "Monitoring training job: ${JOB_NAME}"
    print_info "Press Ctrl+C to stop monitoring (job will continue running)"

    local status=""
    local last_status=""

    while true; do
        status=$(aws sagemaker describe-training-job \
            --training-job-name "${JOB_NAME}" \
            --region "${REGION}" \
            --query 'TrainingJobStatus' \
            --output text)

        if [ "${status}" != "${last_status}" ]; then
            case "${status}" in
                InProgress)
                    print_info "Training job status: ${status}"
                    ;;
                Completed)
                    print_success "Training job completed successfully!"
                    break
                    ;;
                Failed|Stopped)
                    print_error "Training job ${status,,}"
                    print_info "Check CloudWatch logs for details"
                    exit 1
                    ;;
            esac
            last_status="${status}"
        fi

        sleep 30
    done

    # Print output location
    local model_artifacts=$(aws sagemaker describe-training-job \
        --training-job-name "${JOB_NAME}" \
        --region "${REGION}" \
        --query 'ModelArtifacts.S3ModelArtifacts' \
        --output text)

    print_success "Model artifacts saved to: ${model_artifacts}"
}

print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Deploy CIFAR-100 WideResNet training to AWS SageMaker

Options:
  -h, --help                Show this help message
  -r, --region REGION       AWS region (default: us-east-1)
  -i, --instance-type TYPE  Instance type (default: ml.p3.2xlarge)
  -c, --instance-count N    Number of instances (default: 1)
  -s, --spot                Use spot instances
  -m, --monitor             Monitor training job after creation
  -n, --name NAME           Custom job name prefix

Environment Variables:
  AWS_REGION               AWS region
  S3_BUCKET                S3 bucket name
  SAGEMAKER_ROLE_NAME      SageMaker execution role name
  INSTANCE_TYPE            EC2 instance type
  USE_SPOT                 Use spot instances (true/false)

Examples:
  # Deploy with defaults
  ./deploy.sh

  # Deploy with spot instances and monitoring
  ./deploy.sh --spot --monitor

  # Deploy with custom instance type
  ./deploy.sh --instance-type ml.p3.8xlarge --instance-count 1

  # Deploy to different region
  ./deploy.sh --region us-west-2

EOF
}

################################################################################
# Main Script
################################################################################

main() {
    local monitor=false

    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -i|--instance-type)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            -c|--instance-count)
                INSTANCE_COUNT="$2"
                shift 2
                ;;
            -s|--spot)
                USE_SPOT=true
                shift
                ;;
            -m|--monitor)
                monitor=true
                shift
                ;;
            -n|--name)
                PROJECT_NAME="$2"
                JOB_NAME="${PROJECT_NAME}-$(date +%Y%m%d-%H%M%S)"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║      AWS SageMaker CIFAR-100 Training Deployment              ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    print_info "Configuration:"
    echo "  Project Name:     ${PROJECT_NAME}"
    echo "  Job Name:         ${JOB_NAME}"
    echo "  Region:           ${REGION}"
    echo "  Instance Type:    ${INSTANCE_TYPE}"
    echo "  Instance Count:   ${INSTANCE_COUNT}"
    echo "  S3 Bucket:        ${S3_BUCKET}"
    echo "  Use Spot:         ${USE_SPOT}"
    echo ""

    # Execute deployment steps
    check_dependencies
    check_aws_credentials
    create_s3_bucket
    create_iam_role
    create_training_job

    if [ "${monitor}" = true ]; then
        echo ""
        monitor_training_job
    else
        print_info "To monitor the job, run:"
        print_info "  aws sagemaker describe-training-job --training-job-name ${JOB_NAME}"
        print_info "Or visit the AWS Console"
    fi

    echo ""
    print_success "Deployment completed successfully!"
    echo ""
}

# Run main function
main "$@"
