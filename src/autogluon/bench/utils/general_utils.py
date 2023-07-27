import json
import logging
import os
import time

import numpy as np
from boto3 import client

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def formatted_time():
    current_time = time.localtime()
    formatted_time = time.strftime("%Y%m%dT%H%M%S", current_time)
    return formatted_time


def upload_to_s3(s3_bucket: str, s3_dir: str, local_path: str):
    """Uploads a file or a directory to an S3 bucket.

    Args:
        s3_bucket (str): The name of the S3 bucket to upload the file or directory to.
        s3_dir (str): The S3 path of the directory to upload the file or directory to.
        local_path (str): The local path of the file or directory to upload.

    Returns:
        The S3 path of the uploaded file or directory.
    """
    import boto3

    logging.info("Saving to S3 Bucket %s...", s3_bucket)
    s3 = boto3.client("s3")

    if os.path.isfile(local_path):
        file_name = os.path.basename(local_path)
        s3_path = os.path.join(s3_dir, file_name)
        s3.upload_file(local_path, s3_bucket, s3_path)
        logging.info(f"File {local_path} has been saved to s3://{s3_bucket}/{s3_path}")
        return f"s3://{s3_bucket}/{s3_path}"
    elif os.path.isdir(local_path):
        if len(os.listdir(local_path)) == 0:
            logger.warning(f"No files under {local_path}.")
            return

        for root, dirs, files in os.walk(local_path):
            for filename in files:
                file_local_path = os.path.join(root, filename)

                relative_path = os.path.relpath(file_local_path, local_path)
                s3_path = os.path.join(s3_dir, relative_path)

                s3.upload_file(file_local_path, s3_bucket, s3_path)

        logging.info("Files under %s have been saved to s3://%s/%s", local_path, s3_bucket, s3_dir)
        return f"s3://{s3_bucket}/{s3_dir}"


def download_file_from_s3(s3_path: str, local_path: str = "/tmp") -> str:
    """Downloads a file from an S3 bucket.

    Args:
        s3_path (str): The S3 path of the file.
        local_path (str): The local path where the file will be downloaded.

    Returns:
        str: The local path of the downloaded file.
    """
    import boto3

    logging.info(f"Downloading file from {s3_path} to {local_path}.")
    s3 = boto3.client("s3")

    bucket = s3_path.strip("s3://").split("/")[0]
    s3_path = s3_path[len(f"s3://{bucket}/") :]

    local_file_path = os.path.join(local_path, s3_path.split("/")[-1])
    s3.download_file(bucket, s3_path, local_file_path)

    return local_file_path


def download_dir_from_s3(s3_path: str, local_path: str) -> str:
    """Downloads a directory from an S3 bucket.

    Args:
        s3_path (str): The S3 path of the directory.
        local_path (str): The local path where the directory will be downloaded.

    Returns:
        str: The local path of the downloaded directory.
    """
    import boto3

    logging.info(f"Downloading dir from {s3_path} to {local_path}.")
    s3 = boto3.client("s3")

    bucket = s3_path.strip("s3://").split("/")[0]
    s3_path = s3_path[len(f"s3://{bucket}/") :]

    response = s3.list_objects(Bucket=bucket, Prefix=s3_path)

    for content in response.get("Contents", []):
        s3_obj_path = content["Key"]
        relative_path = os.path.relpath(s3_obj_path, s3_path)
        local_obj_path = os.path.join(local_path, relative_path)

        os.makedirs(os.path.dirname(local_obj_path), exist_ok=True)
        s3.download_file(bucket, s3_obj_path, local_obj_path)

    return local_path
