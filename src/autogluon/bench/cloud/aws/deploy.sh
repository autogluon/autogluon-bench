#!/bin/bash

set -eo pipefail

STACK_NAME_PREFIX=$1
STACK_NAME_TAG=$2
STATIC_RESOURCE_STACK_NAME=$3
BATCH_STACK_NAME=$4
CONTAINER_MEMORY=$5
CDK_DEPLOY_REGION=$6
CDK_PATH=$7

update_shm_size() {
    echo "Container shm_size is $CONTAINER_MEMORY"

    stack_template="./cdk.out/$BATCH_STACK_NAME.template.json"
    echo "Fixing shared size in CloudFormation template $stack_template"
    sed -i "s/\"LinuxParameters\": {}/\"LinuxParameters\": {\"SharedMemorySize\": $CONTAINER_MEMORY}/g" $stack_template

    echo "Updating container shm_size of stack $BATCH_STACK_NAME"
    aws cloudformation deploy \
    --template-file $stack_template \
    --stack-name $BATCH_STACK_NAME \
    --region $CDK_DEPLOY_REGION \
    --capabilities CAPABILITY_NAMED_IAM
}

update_batch_tags() {
    ARNS_TO_UPDATE=("ComputeEnvironmentARN" "JobDefinitionARN" "JobQueueARN")

    arns=`aws cloudformation describe-stacks \
    --stack-name $BATCH_STACK_NAME --region $CDK_DEPLOY_REGION \
    | jq '.Stacks | .[] | .Outputs \
    | reduce .[] as $i ({}; .[$i.OutputKey] = $i.OutputValue)'`
    
    stack_tag="$STACK_NAME_PREFIX=$STACK_NAME_TAG"
    
    for t in ${ARNS_TO_UPDATE[@]}; do
        resource=`echo $arns | jq --arg t $t ' .[$t]' | xargs`
        echo "Tagging $resource with $stack_tag"
        aws batch tag-resource --resource-arn $resource --region $CDK_DEPLOY_REGION --tags $stack_tag
    done
}

ECR_REPO=763104351884.dkr.ecr.us-east-1.amazonaws.com
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO

echo "Running CDK deploy"
cdk deploy --app $CDK_PATH $STATIC_RESOURCE_STACK_NAME --require-approval never
cdk deploy --app $CDK_PATH $BATCH_STACK_NAME --require-approval never

# Workaround for lack of support from CDK
update_shm_size
update_batch_tags
