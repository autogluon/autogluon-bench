import logging
import os
import shutil
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Benchmark(ABC):
    def __init__(self, benchmark_name: str, benchmark_dir: str):
        self.benchmark_name = benchmark_name
        self.benchmark_dir = benchmark_dir
        self.metrics_dir = os.path.join(self.benchmark_dir, "results")
        self.benchmark_dir_s3 = None

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def upload_metrics(self, s3_bucket: str, s3_dir: str):
        """Uploads benchmark metrics to an S3 bucket.

        Args:
            s3_bucket (str): The name of the S3 bucket to upload the metrics to.
            s3_dir (str): The S3 path of the directory to upload the metrics to.

        Returns:
            None
        """

        import boto3

        logging.info("Saving metrics to S3 Bucket %s...", s3_bucket)
        self.benchmark_dir_s3 = f"s3://{s3_bucket}/{s3_dir}"
        s3 = boto3.client("s3")

        if len(os.listdir(self.metrics_dir)) == 0:
            logger.warning("No metrics were created.")
            return

        for root, dirs, files in os.walk(self.metrics_dir):
            for filename in files:
                local_path = os.path.join(root, filename)

                relative_path = os.path.relpath(local_path, self.metrics_dir)
                s3_path = os.path.join(s3_dir, relative_path)

                # upload the file to S3
                s3.upload_file(local_path, s3_bucket, s3_path)

        logging.info("Metrics under %s have been saved to s3://%s/%s.", self.metrics_dir, s3_bucket, s3_dir)

    def cleanup_metrics(self):
        """
        Remove benchmark metrics from local and S3 storage.

        This method removes the directory specified by `self.benchmark_dir` from the local file system.
        If `self.benchmark_dir_s3` is also specified, it removes the corresponding directory from S3.

        """
        shutil.rmtree(self.benchmark_dir)
        if self.benchmark_dir_s3 is not None:
            import boto3

            s3 = boto3.resource("s3")
            bucket_name = self.benchmark_dir_s3.split("//")[-1].split("/")[0]
            benchmark_dir = self.benchmark_dir_s3.split(bucket_name)[-1].lstrip("/")
            bucket = s3.Bucket(bucket_name)
            bucket.objects.filter(Prefix=benchmark_dir).delete()
            s3.Object(bucket_name, self.benchmark_dir_s3).delete()
