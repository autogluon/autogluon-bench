#!/bin/bash

source /etc/environment

TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id)

while true; do
  # Check if nvidia-smi exists and GPU is present
  if command -v nvidia-smi &> /dev/null && lspci | grep -qi nvidia; then
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | while read line; do
      GPU_INDEX=$(echo $line | awk -F ', ' '{print $1}')
      GPU_UTILIZATION=$(echo $line | awk -F ', ' '{printf "%.2f", $2}')
      aws cloudwatch put-metric-data --region $CDK_DEPLOY_REGION --namespace EC2 --storage-resolution 1 --unit Percent --dimensions InstanceId=$INSTANCE_ID --metric-name GPUUtilization_$GPU_INDEX --value $GPU_UTILIZATION
    done
  fi

  CPU_UTILIZATION=$(top -bn1 | grep "Cpu(s)" | awk -F'id,' '{ split($1, vs, ","); v=vs[length(vs)]; sub("%", "", v); printf "%.2f", 100 - v }')
  aws cloudwatch put-metric-data --region $CDK_DEPLOY_REGION --namespace EC2 --storage-resolution 1 --unit Percent --dimensions InstanceId=$INSTANCE_ID --metric-name CPUUtilization --value $CPU_UTILIZATION
  
  sleep 5
done
