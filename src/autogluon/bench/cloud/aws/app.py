#!/usr/bin/env python3

import logging
import os

from aws_cdk import App, Environment, Tags

from autogluon.bench.cloud.aws.batch_stack.stack import BatchJobStack, StaticResourceStack

logger = logging.getLogger(__name__)


def get_mandatory_env(name):
    """
    Reads the env variable, raises an exception if missing.
    """
    if name not in os.environ:
        raise Exception("Missing os enviroment variable '%s'" % name)
    return os.environ.get(name)


cdk_default_account = get_mandatory_env("CDK_DEPLOY_ACCOUNT")
cdk_default_region = get_mandatory_env("CDK_DEPLOY_REGION")
logger.info("Deploying the stack to %s %s", cdk_default_account, cdk_default_region)

app = App()
prefix = app.node.try_get_context("STACK_NAME_PREFIX")
tag = app.node.try_get_context("STACK_NAME_TAG")
static_resource_stack_name = app.node.try_get_context("STATIC_RESOURCE_STACK_NAME")
batch_stack_name = app.node.try_get_context("BATCH_STACK_NAME")

env = Environment(account=cdk_default_account, region=cdk_default_region)
static_resource_stack = StaticResourceStack(app, static_resource_stack_name, env=env)
batch_stack = BatchJobStack(app, batch_stack_name, env=env, static_stack=static_resource_stack)
batch_stack.add_dependency(static_resource_stack)

# Stack-level tag which expects to be propogated to resources
# Looks like Batch (compute environment, job definition, job queue)
# is not supported currently
# https://aws.amazon.com/premiumsupport/knowledge-center/cloudformation-propagate-stack-level-tag/
Tags.of(app).add(key=prefix, value=tag)
app.synth()
