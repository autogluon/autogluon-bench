#!/bin/bash

STATIC_RESOURCE_STACK_NAME=$1
BATCH_STACK_NAME=$2


# Get the stack outputs JSON object
OUTPUTS_JSON=$(aws cloudformation describe-stacks --stack-name "${BATCH_STACK_NAME}" --query "Stacks[0].Outputs" --output json)

REPOSITORY_NAME=$(echo "${OUTPUTS_JSON}" | jq -r '.[] | select(.OutputKey == "EcrRepositoryName") | .OutputValue')
IMAGE_URI=$(echo "${OUTPUTS_JSON}" | jq -r '.[] | select(.OutputKey == "ImageUri") | .OutputValue')
IMAGE_TAG_OR_DIGEST=$(echo "${IMAGE_URI}" | sed -n 's/.*://p')

echo "Destroying $BATCH_STACK_NAME"
cdk destroy $BATCH_STACK_NAME
echo "Destroying $STATIC_RESOURCE_STACK_NAME"
cdk deploy $STATIC_RESOURCE_STACK_NAME

echo "Removing image $IMAGE_TAG_OR_DIGEST from ECR repository $REPOSITORY_NAME."
aws ecr batch-delete-image --repository-name "${REPOSITORY_NAME}" --image-ids "imageTag=${IMAGE_TAG_OR_DIGEST}"
