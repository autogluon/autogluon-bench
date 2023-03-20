import os
from itertools import product

import boto3

aws_batch = boto3.client("batch")


def submit_batch_job(env: list, job_name: str, job_queue: str, job_definition: str):
    container_overrides = {"environment": env}
    response = aws_batch.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides=container_overrides,
    )
    print(response)


def handler(event, context):
    """
    Execution entrypoint for AWS Lambda.
    Triggers batch jobs with hyperparameter combinations.
    ENV variables are set by the AWS CDK infra code.
    
    For multimodal, an example of `event` looks like:
    event = {
        "benchmark_name": "test",
        "module": "multimodal",
        "data_paths": [
            "MNIST"
        ],
        "git_uri#branch": [
            "https://github.com/autogluon/autogluon.git#master"
        ],
        "s3_bucket": "autogluon-benchmark-metrics"
    }
    More hyperparameters to be added...

    For tabular, an example of `event` looks like:
    event = {
        "benchmark_name": "test",
        "module": "tabular",
        "frameworks": [
            "AutoGluon"
        ],
        "labels": [
            "stable",
            "latest"
        ],
        "amlb_benchmarks": [
            "test",
            "small",
            "validation"
        ],
        "amlb_constraints": [
            "test",
            "1h4c"
        ],
        "amlb_tasks": {
            "test": ["iris", "cholesterol"],
            "small": ["credit-g"],
        },
        "s3_bucket": "autogluon-benchmark-metrics"
    }
    """
    batch_job_queue = os.environ.get("BATCH_JOB_QUEUE")
    batch_job_definition = os.environ.get("BATCH_JOB_DEFINITION")
    benchmark_name = event.get("benchmark_name", "test")
    s3_bucket = event.get("s3_bucket", None)
    module = event.get("module", None)
    config_file = event.get("module", None)
    env = [
        {"name": "benchmark_name", "value": benchmark_name},
        {"name": "module", "value": module},
        {"name": "mode", "value": "local"},
    ]
    if s3_bucket is not None and len(s3_bucket) > 0:
        env.append({"name": "s3_bucket", "value": s3_bucket})
    if 'frameworks' in event:
        frameworks = event.get('frameworks', [])
        labels = event.get('labels', [])
        amlb_benchmarks = event.get('amlb_benchmarks', [])
        amlb_constraints = event.get('amlb_constraints', [])
        amlb_tasks = event.get('amlb_tasks', {})
        
        for combination in product(frameworks, labels, amlb_benchmarks, amlb_constraints):
            framework, label, amlb_benchmark, amlb_constraint = combination
            env.append({"name": "framework", "value": framework})
            env.append({"name": "label", "value": label})
            env.append({"name": "amlb_benchmark", "value": amlb_benchmark})
            env.append({"name": "amlb_constraint", "value": amlb_constraint})
                
            if amlb_benchmark in amlb_tasks:
                tasks = amlb_tasks[amlb_benchmark]
                for task in tasks:
                    env.append({"name": "amlb_task", "value": str(task)})
            job_name = f"%s-%s-%s-%s-%s" % (
                benchmark_name,
                module,
                label,
                amlb_benchmark,
                amlb_constraint,
            )
            submit_batch_job(
                env=env, 
                job_name=job_name, 
                job_queue=batch_job_queue, 
                job_definition=batch_job_definition
            )
            
    elif 'git_uri#branch' in event:
        git_uri_branches = event.get('git_uri#branch', [])
        data_paths = event.get('data_paths', [])
        
        for combination in product(data_paths, git_uri_branches):
            data_path, git_uri_branch = combination
            git_uri, git_branch = git_uri_branch.split("#")
            env.append({"name": "git_uri", "value": git_uri})
            env.append({"name": "git_branch", "value": git_branch})
            env.append({"name": "data_path", "value": data_path})
            
            job_name = f"%s-%s-%s-%s" % (
                benchmark_name,
                module,
                git_branch,
                data_path,
            )
            submit_batch_job(
                env=env, 
                job_name=job_name, 
                job_queue=batch_job_queue, 
                job_definition=batch_job_definition
            )
    else:
        raise NotImplementedError
    
    

    return "Lambda execution finished"

