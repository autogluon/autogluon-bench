from __future__ import annotations

import boto3

from urllib.parse import urlparse

from autogluon.common.loaders import load_s3
from autogluon.common.utils import s3_utils


def is_s3_url(path: str) -> bool:
    """
    Checks if path is a s3 uri.

    Params:
    -------
    path: str
        The path to check.

    Returns:
    --------
    bool: whether the path is a s3 uri.
    """
    if (path[:2] == "s3") and ("://" in path[:6]):
        return True
    return False


def get_bucket_key(s3_uri: str) -> tuple[str, str]:
    """
    Retrieves the bucket and key from a s3 uri.

    Params:
    -------
    origin_path: str
        The path (s3 uri) to be parsed.

    Returns:
    --------
    bucket_name: str
        the associated bucket name
    object_key: str
        the associated key
    """
    if not is_s3_url(s3_uri):
        raise ValueError("Invalid S3 URI scheme. It should be 's3'.")

    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    object_key = parsed_uri.path.lstrip("/")

    return bucket_name, object_key


def get_s3_paths(path_prefix: str, contains: str | None = None, suffix: str | None = None) -> list[str]:
    """
    Gets all s3 paths in the path_prefix that contain 'contains'
    and end with 'suffix.'

    Params:
    -------
    path_prefix: str
        The path prefix.
    contains : Optional[str], default = None
        Can be specified to limit the returned outputs.
        For example, by specifying the constraint, such as ".1h8c."
    suffix: str, default = None
        Can be specified to limit the returned outputs.
        For example, by specifying "leaderboard.csv" only objects ending
        with this suffix will be included
        If no suffix provided, will save all files in artifact directory.

    Returns:
    --------
    List[str]: All s3 paths that adhere to the conditions passed in.
    """
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = load_s3.list_bucket_prefix_suffix_contains_s3(
        bucket=bucket, prefix=prefix, suffix=suffix, contains=contains
    )
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version="s3") for file in objects]
    return paths_full


def copy_s3_object(origin_path: str, destination_path: str) -> bool:
    """
    Copies s3 object from origin_path to destination_path

    Params:
    -------
    origin_path: str
        The path (s3 uri) to the original location of the object
    destination_path: str
        The path (s3 uri) to the intended destination location of the object

    Returns:
    --------
    bool: whether the copy was successful.
    """
    origin_bucket, origin_key = get_bucket_key(origin_path)
    destination_bucket, destination_key = get_bucket_key(destination_path)

    try:
        s3 = boto3.client("s3")
        s3.copy_object(
            Bucket=destination_bucket, CopySource={"Bucket": origin_bucket, "Key": origin_key}, Key=destination_key
        )
        return True
    except:
        pass

    return False
