import os
from itertools import product

import boto3

aws_batch = boto3.client("batch")


def get_container_job_environment(
    event,
    dataset="nike",
    max_epoch=-1,
    per_gpu_batch_size=-1,
    batch_size=-1,
    timm_chkpt=None,
):
    hpo = event.get("hpo", "false")
    epochs = event.get("epochs", "null")
    lr_range = event.get("lr_range", "null")
    optim_types = event.get("optim_types", "null")
    models = event.get("models", "null")
    timm_chkpts = (
        timm_chkpt if timm_chkpt is not None else event.get("timm_chkpts", "null")
    )
    num_trials = event.get("num_trials", 50)
    searcher = event.get("searcher", "bayes")
    scheduler = event.get("scheduler", "ASHA")
    experiment = event.get("experiment", "test")
    dynamo_db_name = os.environ.get("DYNAMODB_TABLE_NAME")
    dynamo_db_region = os.environ.get("DYNAMODB_TABLE_REGION")
    model_bucket = os.environ.get("MODEL_BUCKET")
    stack_name_prefix = os.environ.get("STACK_NAME_PREFIX")

    env = [
        {"name": "DATASET_NAME", "value": dataset},
        {"name": "MAX_EPOCHS", "value": str(max_epoch)},
        {"name": "PER_GPU_BATCH_SIZE", "value": str(per_gpu_batch_size)},
        {"name": "BATCH_SIZE", "value": str(batch_size)},
        {"name": "DB_NAME", "value": dynamo_db_name},
        {"name": "DB_REGION", "value": dynamo_db_region},
        {"name": "MODEL_BUCKET", "value": model_bucket},
        {"name": "LR_RANGE", "value": lr_range},
        {"name": "OPTIM_TYPES", "value": optim_types},
        {"name": "EPOCHS", "value": epochs},
        {"name": "MODELS", "value": models},
        {"name": "TIMM_CHKPTS", "value": timm_chkpts},
        {"name": "NUM_TRIALS", "value": str(num_trials)},
        {"name": "SEARCHER", "value": searcher},
        {"name": "SCHEDULER", "value": scheduler},
        {"name": "HPO", "value": hpo},
        {"name": "EXPERIMENT", "value": experiment},
    ]

    if hpo == "false":
        job_name = f"{stack_name_prefix}-%s-epoch-%s-bs-%s-gpu_bs-%s-model-%s" % (
            dataset,
            str(max_epoch),
            str(per_gpu_batch_size),
            str(batch_size),
            timm_chkpt[:10],
        )
    else:
        job_name = f"{stack_name_prefix}-%s-hpo-%s-%s" % (
            dataset,
            lr_range.replace(",", "_").replace(".", "_"),
            "_".join(m[:8] for m in timm_chkpts.split(",")),
        )
    return env, job_name


def handler(event, context):
    """
    Execution entrypoint for AWS Lambda.
    Triggers batch jobs with each dataset and hpo combinations.
    ENV variables are set by the AWS CDK infra code.
    An example for HPO task event looks like:
    event = {
        "datasets": [
            "nike"
        ],
        "lr_range": "0.0005,0.001",
        "optim_types": "adamw,sgd",
        "epochs": "1,10",
        "searcher": "bayes",
        "scheduler": "ASHA",
        "num_trials": "3",
        "timm_chkpts": "mobilenetv3_large_100,ghostnet_100",
        "hpo": "true",
        "experiment": "exp1"
    }
    An example for regular task event looks like:
    event = {
        "datasets": [
            "nike"
        ],
        "max_epochs": [10],
        "per_gpu_batch_size": [16],
        "timm_chkpts": "mobilenetv3_large_100,ghostnet_100",
        "experiment": "exp2"
    }
    """
    batch_job_queue = os.environ.get("BATCH_JOB_QUEUE")
    batch_job_definition = os.environ.get("BATCH_JOB_DEFINITION")

    hpo = event.get("hpo", "false")
    datasets = event.get("datasets", [])
    max_epochs = event.get("max_epochs", [])
    per_gpu_batch_size = event.get("per_gpu_batch_size", [])
    batch_size = event.get("batch_size", [])

    if hpo == "true":
        for data in datasets:
            env, batch_job_name = get_container_job_environment(
                event=event, dataset=data
            )
            container_overrides = {"environment": env}
            response = aws_batch.submit_job(
                jobName=batch_job_name,  # jobName should not exceed 128 characters
                jobQueue=batch_job_queue,
                jobDefinition=batch_job_definition,
                containerOverrides=container_overrides,
            )
            print(response)
    else:
        timm_chkpts = event.get("timm_chkpts", "swin_base_patch4_window7_224")
        timm_chkpts = timm_chkpts.split(",")
        for data, epoch, gpu_batch_size, bs, timm_chkpt in product(
            datasets, max_epochs, per_gpu_batch_size, batch_size, timm_chkpts
        ):
            env, batch_job_name = get_container_job_environment(
                event=event,
                dataset=data,
                max_epoch=epoch,
                per_gpu_batch_size=gpu_batch_size,
                batch_size=bs,
                timm_chkpt=timm_chkpt,
            )
            container_overrides = {"environment": env}
            response = aws_batch.submit_job(
                jobName=batch_job_name,
                jobQueue=batch_job_queue,
                jobDefinition=batch_job_definition,
                containerOverrides=container_overrides,
            )
            print(response)

    return "Lambda execution finished"
