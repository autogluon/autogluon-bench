#!/bin/bash

STACK_NAME_PREFIX=$1
STACK_NAME_TAG=$2
STATIC_RESOURCE_STACK_NAME=$3
BATCH_STACK_NAME=$4
CONTAINER_MEMORY=$5

update_shm_size() {
    echo "Container shm_size is $CONTAINER_MEMORY"

    stack_template="./cdk.out/$BATCH_STACK_NAME.template.json"
    echo "Fixing shared size in CloudFormation template $stack_template"
    sed -i "s/\"LinuxParameters\": {}/\"LinuxParameters\": {\"SharedMemorySize\": $CONTAINER_MEMORY}/g" $stack_template

    echo "Updating container shm_size of stack $BATCH_STACK_NAME"
    aws cloudformation deploy \
    --template-file $stack_template \
    --stack-name $BATCH_STACK_NAME \
    --capabilities CAPABILITY_NAMED_IAM
}

update_batch_tags() {
    ARNS_TO_UPDATE=("ComputeEnvironmentARN" "JobDefinitionARN" "JobQueueARN")

    arns=`aws cloudformation describe-stacks \
    --stack-name $BATCH_STACK_NAME \
    | jq '.Stacks | .[] | .Outputs \
    | reduce .[] as $i ({}; .[$i.OutputKey] = $i.OutputValue)'`
    
    stack_tag="$STACK_NAME_PREFIX=$STACK_NAME_TAG"
    
    for t in ${ARNS_TO_UPDATE[@]}; do
        resource=`echo $arns | jq --arg t $t ' .[$t]' | xargs`
        echo "Tagging $resource with $stack_tag"
        aws batch tag-resource --resource-arn $resource --tags $stack_tag
    done
}

echo "Running CDK deploy"
cdk deploy --all

# Workaround for lack of support from CDK
update_shm_size
update_batch_tags
