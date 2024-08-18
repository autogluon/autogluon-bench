from __future__ import annotations

from autogluon.common.loaders import load_s3
from autogluon.common.utils import s3_utils


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
