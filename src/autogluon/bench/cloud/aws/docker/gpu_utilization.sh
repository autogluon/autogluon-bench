#!/bin/bash

source /etc/environment

if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi -L &> /dev/null; then
    echo "No GPU or NVIDIA drivers found. Exiting."
    exit 0
fi

GPU_UTILIZATION=`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{ printf "%.2f\n", $1 }'`
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

/opt/conda/bin/aws cloudwatch put-metric-data --region $CDK_DEPLOY_REGION --namespace EC2 --dimensions InstanceId=$INSTANCE_ID --metric-name GPUUtilization --value $GPU_UTILIZATION
