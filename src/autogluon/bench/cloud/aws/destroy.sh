#!/bin/bash

set -eo pipefail

STATIC_RESOURCE_STACK_NAME=$1
BATCH_STACK_NAME=$2
CDK_DEPLOY_REGION=$3
CDK_PATH=$4


# Get the stack outputs JSON object
OUTPUTS_JSON=$(aws cloudformation describe-stacks --stack-name "${BATCH_STACK_NAME}" --region "${CDK_DEPLOY_REGION}" --query "Stacks[0].Outputs" --output json)

REPOSITORY_NAME=$(echo "${OUTPUTS_JSON}" | jq -r '.[] | select(.OutputKey == "EcrRepositoryName") | .OutputValue')
IMAGE_URI=$(echo "${OUTPUTS_JSON}" | jq -r '.[] | select(.OutputKey == "ImageUri") | .OutputValue')
IMAGE_TAG_OR_DIGEST=$(echo "${IMAGE_URI}" | sed -n 's/.*://p')

echo "Destroying $BATCH_STACK_NAME"
cdk destroy --app $CDK_PATH $BATCH_STACK_NAME --force
echo "Destroying $STATIC_RESOURCE_STACK_NAME"
cdk destroy --app $CDK_PATH $STATIC_RESOURCE_STACK_NAME --force

echo "Removing image $IMAGE_TAG_OR_DIGEST from ECR repository $REPOSITORY_NAME."
aws ecr batch-delete-image --repository-name "${REPOSITORY_NAME}" --region $CDK_DEPLOY_REGION --image-ids "imageTag=${IMAGE_TAG_OR_DIGEST}"
