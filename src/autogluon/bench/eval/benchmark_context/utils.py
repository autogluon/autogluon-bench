from autogluon.common.loaders import load_s3
from autogluon.common.utils import s3_utils


def get_s3_paths(path_prefix: str, contains=None, suffix=None):
    bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path_prefix)
    objects = load_s3.list_bucket_prefix_suffix_contains_s3(
        bucket=bucket, prefix=prefix, suffix=suffix, contains=contains
    )
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version="s3") for file in objects]
    return paths_full
