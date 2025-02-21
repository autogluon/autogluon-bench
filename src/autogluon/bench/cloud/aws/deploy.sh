#!/bin/bash

set -eo pipefail

STATIC_RESOURCE_STACK_NAME=$1
BATCH_STACK_NAME=$2
CDK_PATH=$3

ECR_REPO=763104351884.dkr.ecr.us-east-1.amazonaws.com
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO

echo "Running CDK deploy"
cdk deploy --app $CDK_PATH $STATIC_RESOURCE_STACK_NAME --require-approval never
cdk deploy --app $CDK_PATH $BATCH_STACK_NAME --require-approval never
