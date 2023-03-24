#!/usr/bin/env python3

import os

import aws_cdk as core
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as _lambda
from constructs import Construct


class BatchLambdaFunction(Construct):
    """
    Custom CDK construct class for a Lambda function with invocation
    rights for AWS Batch.
    """

    @property
    def function(self):
        return self._lambda_function

    def __init__(
        self,
        scope: Construct,
        id: str,
        prefix: str,
        tag: str,
        function_name: str,
        code_path: str,
        environment,
        timeout=600,
    ):
        super().__init__(scope, id)
        aws_account_region = os.environ["CDK_DEPLOY_REGION"]
        aws_account_id = os.environ["CDK_DEPLOY_ACCOUNT"]

        self._lambda_function_role = iam.Role(
            self,
            "lambda-function-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ],
            inline_policies={
                "SubmitBatchForLambda": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            resources=[
                                f"arn:aws:batch:{aws_account_region}:{aws_account_id}:job-definition/*",
                                f"arn:aws:batch:{aws_account_region}:{aws_account_id}:job-queue/*",
                            ],
                            conditions={
                                "StringEquals": {
                                    f"aws:ResourceTag/{prefix}": tag  # resources with tag, e.g. {"automm": "vision"}
                                }
                            },
                            actions=["batch:SubmitJob"],
                        )
                    ]
                )
            },
        )

        self._lambda_function = _lambda.Function(
            self,
            id,
            function_name=function_name,
            runtime=_lambda.Runtime.PYTHON_3_8,
            environment=environment,
            code=_lambda.Code.from_asset(
                code_path,
                bundling=core.BundlingOptions(
                    image=_lambda.Runtime.PYTHON_3_8.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "pip install --no-cache -r requirements.txt -t /asset-output && cp -au . /asset-output",
                    ],
                ),
            ),
            handler="lambda_function.handler",
            timeout=core.Duration.seconds(timeout),
            role=self._lambda_function_role,
        )
