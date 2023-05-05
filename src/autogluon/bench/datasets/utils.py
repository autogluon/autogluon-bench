import os


def get_home_dir():
    """Get home directory"""
    home_dir = os.environ.get("AUTOGLUON_BENCH_HOME", os.path.join("~", ".autogluon_bench"))
    # expand ~ to actual path
    home_dir = os.path.expanduser(home_dir)
    return home_dir


def get_data_home_dir():
    """Get home directory for storing the datasets"""
    home_dir = get_home_dir()
    return os.path.join(home_dir, "datasets")


def get_repo_url():
    """Return the base URL for dataset repository"""
    default_repo = "https://automl-mm-bench.s3.amazonaws.com"
    repo_url = os.environ.get("AUTOGLUON_BENCH_REPO", default_repo)
    if repo_url[-1] != "/":
        repo_url = repo_url + "/"
    return repo_url
